"""Latency of the three stages before generation: question embedding, retrieval, context build.

Same code path as `application.answer.ask` up to its "context" lap, with no LLM call.
Run from the project root (on the Nano, ideally through scripts/nano_measure.sh):

    uv run python scripts/profile_pre_llm.py --model ruri-v3-130m --device cuda
    uv run python scripts/profile_pre_llm.py --fake          # smoke test, no model needed
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

from memrag.api import MemRAG, _retrieval_input, load_memories
from memrag.application.answer import build_context
from memrag.application.evaluation.retrieval_profile import percentiles
from memrag.domain.schemas.retrieval import MemoryFilter

STAGES = ("embed_query", "retrieve", "context")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ruri-v3-130m")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--out", default=None, help="write the summary JSON here")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="pre_llm_"))
    try:
        t = time.perf_counter()
        rag = MemRAG.from_config(args.model, device=args.device, fake=args.fake, store_dir=work, use_cache=False)
        rag.embed(["warm up"])  # first call loads the model on some backends
        cold_load_s = round(time.perf_counter() - t, 2)
        rag.index(load_memories())
        data, _ = _retrieval_input("memories", None)
        cfg = rag._cfg
        budget = cfg["answer"]["context_budget_chars"]

        rows = []
        order = [(rep, i, q) for rep in range(args.repeats) for i, q in enumerate(data.queries)]
        for rep, i, q in order:
            flt = rag.pipeline_filter(MemoryFilter(person_id=q.person_id, category=q.filters.category,
                                                   date_from=q.filters.date_from, date_to=q.filters.date_to))
            t0 = time.perf_counter()
            vector = rag._embedder.embed_queries([q.text])[0]
            t1 = time.perf_counter()
            hits = rag.store.search(vector, args.top_k, flt)
            records = {r.memory_id: r for r in rag.store.get([h.memory_id for h in hits])}
            t2 = time.perf_counter()
            build_context([(records[h.memory_id], h.score) for h in hits], budget)
            t3 = time.perf_counter()
            if i >= args.warmup or rep > 0:
                rows.append({"embed_query": (t1 - t0) * 1000, "retrieve": (t2 - t1) * 1000,
                             "context": (t3 - t2) * 1000, "total": (t3 - t0) * 1000})

        summary = {"model": args.model, "device": args.device, "fake": args.fake, "top_k": args.top_k,
                   "queries": len(data.queries), "memories": rag.store.count(), "samples": len(rows),
                   "store": cfg["retrieval"]["store"], "model_load_s": cold_load_s,
                   **{s: percentiles([r[s] for r in rows]) for s in (*STAGES, "total")}}
        total = sum(r["total"] for r in rows)
        summary["share"] = {s: round(sum(r[s] for r in rows) / total, 4) for s in STAGES}
        rag.close()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"{args.model} on {args.device}, Top-k {args.top_k}, {summary['memories']} memories, "
          f"{summary['queries']} queries x {args.repeats} (warm-up skipped), model load {cold_load_s}s")
    print(f"{'stage':<12}{'p50':>9}{'p95':>9}{'p99':>9}{'mean':>9}{'share':>8}   (ms)")
    for s in (*STAGES, "total"):
        p = summary[s]
        share = f"{summary['share'][s] * 100:.1f}%" if s in summary["share"] else ""
        print(f"{s:<12}{p['p50']:>9.3f}{p['p95']:>9.3f}{p['p99']:>9.3f}{p['mean']:>9.3f}{share:>8}")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
