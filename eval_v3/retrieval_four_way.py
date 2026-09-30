"""Compare BM25, BGE-M3, weighted RRF, and RRF + CPU reranker without API calls."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_v3.metrics import percentile, rank_metrics
from shopping_agent.hybrid import HybridRetriever
from shopping_agent.product_store import ProductLineStore, cache_key
from shopping_agent.reranker import MainAgentReranker
from shopping_agent.rrf import fuse
from shopping_agent.settings import settings

DATASET = Path(__file__).resolve().parent / "datasets/product_250_local.jsonl"
RESULTS = Path(__file__).resolve().parent / "results"
VARIANTS = ("bm25", "vector", "fusion", "fusion_rerank")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("title", "partial", "attribute"), required=True)
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--skip-rerank", action="store_true")
    args = parser.parse_args()
    cases = [json.loads(line) for line in DATASET.read_text(encoding="utf8").splitlines() if line][:args.limit]
    if not cases:
        raise ValueError("No prepared cases")
    path = settings.data_dir / "products.jsonl"
    store = ProductLineStore(path, with_attributes=settings.load_attributes)
    retriever = HybridRetriever(store, bm25_cache=settings.data_dir / "bm25_cache.pkl",
                                bm25_cache_key=cache_key(path))
    reranker = None if args.skip_rerank else MainAgentReranker()
    output = RESULTS / f"retrieval_{args.tier}_{len(cases)}{'_no_rerank' if args.skip_rerank else ''}.json"
    output.parent.mkdir(exist_ok=True)
    rows = []
    for n, case in enumerate(cases, 1):
        query, pid = case["queries"][args.tier], case["product_id"]
        t0 = time.perf_counter()
        bm = retriever.bm25.search(query, 40)
        bm_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        vec = retriever.vector.search(query, 80)
        vec_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        merged = fuse(bm, vec, weights=(1.5, 1.0), top_k=20)
        fusion_ms = (time.perf_counter() - t0) * 1000
        ranked = {"bm25": bm, "vector": vec, "fusion": merged}
        latency = {"bm25": bm_ms, "vector": vec_ms, "fusion": bm_ms + vec_ms + fusion_ms}
        if reranker:
            t0 = time.perf_counter()
            ranked["fusion_rerank"] = reranker.rerank(query, merged, None, 8)
            latency["fusion_rerank"] = latency["fusion"] + (time.perf_counter() - t0) * 1000
        ids = {name: [product.product_id for product, _ in result[:8]] for name, result in ranked.items()}
        rows.append({"case_id": case["case_id"], "product_id": pid, "query": query,
                     "attribute_fallback": case["attribute_fallback"], "ids": ids,
                     "metrics": {name: rank_metrics(value, pid, 8) for name, value in ids.items()},
                     "latency_ms": {key: round(value, 2) for key, value in latency.items()},
                     "candidate_hit": pid in {p.product_id for p, _ in bm + vec}})
        if n % 10 == 0 or n == len(cases):
            metrics = {}
            for name in ranked:
                metrics[name] = {
                    key: round(statistics.fmean(row["metrics"][name][key] for row in rows), 4)
                    for key in ("recall@8", "mrr@8", "ndcg@8")}
                timings = [row["latency_ms"][name] for row in rows]
                metrics[name].update(p50_ms=percentile(timings, .5), p95_ms=percentile(timings, .95))
            payload = {"status": "complete" if n == len(cases) else "partial", "tier": args.tier,
                       "cases": len(cases), "processed": n, "model_api_calls": 0,
                       "candidate_coverage": round(sum(row["candidate_hit"] for row in rows) / len(rows), 4),
                       "metrics": metrics, "rows": rows}
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
            print(f"{args.tier}: {n}/{len(cases)} -> {output}", flush=True)


if __name__ == "__main__":
    main()
