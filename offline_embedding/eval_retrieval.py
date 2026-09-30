"""Evaluate the local two-way recall (BM25 + BGE-M3/FAISS) and rerank pipeline.

Runs on the FULL 2.75M corpus without loading it into RAM: BM25 is built over a
``ProductLineStore`` (lazy, byte-offset backed) so only the inverted index and the
doc lengths stay resident, and products are decoded one at a time as ranking needs them.

Reports Recall@K / MRR / NDCG@K for each stage so the contribution of every component
is visible:

    BM25 only  ->  vector only  ->  RRF fusion  ->  + reranker

    .\\.venv\\Scripts\\python.exe offline_embedding\\eval_retrieval.py
    .\\.venv\\Scripts\\python.exe offline_embedding\\eval_retrieval.py --no-rerank
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "model_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(ROOT))

from shopping_agent.bm25 import BM25Index  # noqa: E402
from shopping_agent.index_contract import EMBEDDING_MAX_LENGTH, EMBEDDING_MODEL  # noqa: E402
from shopping_agent.product_store import ProductLineStore  # noqa: E402
from shopping_agent.rrf import fuse  # noqa: E402


class _Counters(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.GetCurrentProcess.restype = wt.HANDLE
_k32.K32GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_Counters), wt.DWORD]


def rss() -> float:
    c = _Counters()
    c.cb = ctypes.sizeof(_Counters)
    _k32.K32GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1024**3


def available_gb() -> float:
    class _Status(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    s = _Status()
    s.dwLength = ctypes.sizeof(_Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / 1024**3


def metrics(rank: int | None, k: int) -> dict:
    if rank is None or rank > k:
        return {"hit": 0.0, "rr": 0.0, "ndcg": 0.0}
    return {"hit": 1.0, "rr": 1.0 / rank, "ndcg": 1.0 / math.log2(rank + 1)}


def title_tokens(title: str) -> frozenset[str]:
    import re

    return frozenset(re.findall(r"[a-z0-9]+", title.lower()))


def leaf_category(category: str) -> str:
    return (category or "").split(">")[-1].strip().lower()


def build_relevance_sets(store, queries: list[dict], threshold: float) -> dict[str, set[int]]:
    """把"唯一正确答案"换成"相关集合"。

    275 万商品的目录里有大量近似重复品：同一个商品的不同剪向/不同经销商都算正确答案，
    只认一个 product_id 会把正确的检索判成失败。这里把相关集合定义为
    「同一叶子品类 + 标题词 Jaccard >= threshold」，并始终包含标注的目标行本身。
    """
    from shopping_agent.models import Product

    targets: dict[str, dict] = {}
    for case in queries:
        pid = str(case["product_id"])
        if pid not in targets:
            targets[pid] = {"tokens": None, "category": None, "row": None}
    # 先定位每个目标的叶子品类与标题词
    wanted_rows: dict[str, int] = {}
    for row, product in enumerate(store):
        pid = product.product_id
        if pid in targets and targets[pid]["row"] is None:
            targets[pid]["tokens"] = title_tokens(product.title)
            targets[pid]["category"] = leaf_category(product.category)
            targets[pid]["row"] = row
        if all(t["row"] is not None for t in targets.values()):
            break

    categories = {t["category"] for t in targets.values() if t["category"]}
    print(f"  目标品类 {len(categories)} 个，扫描同品类商品…", flush=True)
    buckets: dict[str, list[tuple[int, frozenset[str]]]] = defaultdict(list)
    for row, product in enumerate(store):
        category = leaf_category(product.category)
        if category in categories:
            buckets[category].append((row, title_tokens(product.title)))

    relevance: dict[str, set[int]] = {}
    for pid, info in targets.items():
        tokens, category, row = info["tokens"], info["category"], info["row"]
        relevant: set[int] = set()
        if row is not None:
            relevant.add(row)
        if tokens:
            for candidate_row, candidate_tokens in buckets.get(category, ()):
                union = tokens | candidate_tokens
                if union and len(tokens & candidate_tokens) / len(union) >= threshold:
                    relevant.add(candidate_row)
        relevance[pid] = relevant
    # 行号 -> product_id，供命中判定使用
    return {pid: {store[row].product_id for row in rows} for pid, rows in relevance.items()}


def first_relevant_rank(ids: list[str], relevant_ids: set[str]) -> int | None:
    for position, product_id in enumerate(ids, start=1):
        if product_id in relevant_ids:
            return position
    return None


def evaluate(rows: list[dict], k: int) -> dict:
    if not rows:
        return {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "n": 0}
    return {
        "recall": sum(r["hit"] for r in rows) / len(rows),
        "mrr": sum(r["rr"] for r in rows) / len(rows),
        "ndcg": sum(r["ndcg"] for r in rows) / len(rows),
        "n": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--queries", type=Path, default=ROOT / "eval" / "queries_nl.jsonl")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--branch-k", type=int, default=50)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--relevance", action="store_true", help="用相关集合代替唯一目标（推荐）")
    parser.add_argument("--relevance-threshold", type=float, default=0.5, help="标题词 Jaccard 阈值")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "retrieval_nl.json")
    args = parser.parse_args()

    products_path = args.data_dir / "products.jsonl"
    index_path = args.data_dir / "faiss_bge_m3" / "index.faiss"
    for required in (products_path, index_path, args.queries):
        if not required.is_file():
            raise SystemExit(f"缺少 {required}")

    queries = [json.loads(line) for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"查询 {len(queries)} 条 | 可用内存 {available_gb():.1f} GB | 进程 {rss():.2f} GB", flush=True)

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print("构建懒加载商品存储（只记行偏移，不载入商品）…", flush=True)
    started = time.time()
    store = ProductLineStore(products_path, with_attributes=False)
    print(f"  {len(store):,} 条，耗时 {time.time()-started:.0f}s | 进程 {rss():.2f} GB", flush=True)

    relevance_sets: dict[str, set[str]] = {}
    if args.relevance:
        print("构建相关集合（同叶子品类 + 标题词 Jaccard）…", flush=True)
        started = time.time()
        relevance_sets = build_relevance_sets(store, queries, args.relevance_threshold)
        sizes = sorted(len(v) for v in relevance_sets.values())
        print(f"  完成，耗时 {time.time()-started:.0f}s；相关集合大小 p50={sizes[len(sizes)//2]} max={sizes[-1]}", flush=True)

    print("构建 BM25（懒加载商品，不复制 Product 对象）…", flush=True)
    started = time.time()
    bm25 = BM25Index(store)
    print(f"  词表 {len(bm25.postings):,} 词，耗时 {time.time()-started:.0f}s | 进程 {rss():.2f} GB | 可用 {available_gb():.1f} GB", flush=True)

    print("载入 FAISS 索引…", flush=True)
    index = faiss.read_index(str(index_path))
    index.nprobe = min(64, index.nlist)
    print(f"  ntotal={index.ntotal:,} | 进程 {rss():.2f} GB", flush=True)

    encoder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    encoder.max_seq_length = EMBEDDING_MAX_LENGTH
    print(f"载入 BGE-M3 | 进程 {rss():.2f} GB | 可用 {available_gb():.1f} GB", flush=True)

    reranker = None
    if not args.no_rerank:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu", trust_remote_code=True, max_length=256)
        print(f"载入 reranker | 进程 {rss():.2f} GB | 可用 {available_gb():.1f} GB", flush=True)

    stages = ["bm25", "vector", "rrf"] + ([] if reranker is None else ["rrf+rerank"])
    per_stage: dict[str, list[dict]] = {name: [] for name in stages}
    per_group: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {name: [] for name in stages})
    timings: dict[str, list[float]] = {name: [] for name in stages}
    detail = []

    for case in queries:
        query, expected, group = case["query"], str(case["product_id"]), case.get("group", "ungrouped")

        started = time.perf_counter()
        bm_rows = bm25.search(query, args.branch_k)
        bm_ms = (time.perf_counter() - started) * 1000
        bm_ids = [p.product_id for p, _ in bm_rows]

        started = time.perf_counter()
        vector = np.asarray(encoder.encode([query], normalize_embeddings=True), dtype="float32")
        scores, ids = index.search(vector, args.branch_k)
        vec_rows = []
        for score, row in zip(scores[0], ids[0]):
            if row < 0:
                continue
            vec_rows.append((store[int(row)], float(score)))
        vec_ms = (time.perf_counter() - started) * 1000
        vec_ids = [p.product_id for p, _ in vec_rows]

        merged = fuse(bm_rows, vec_rows, top_k=args.branch_k)
        rrf_ids = [p.product_id for p, _ in merged]

        ranks = {}
        relevant = relevance_sets.get(expected) if args.relevance else None
        for name, id_list in (("bm25", bm_ids), ("vector", vec_ids), ("rrf", rrf_ids)):
            if relevant is not None:
                rank = first_relevant_rank(id_list, relevant)
            else:
                rank = id_list.index(expected) + 1 if expected in id_list else None
            ranks[name] = rank
            row = metrics(rank, args.top_k)
            per_stage[name].append(row)
            per_group[group][name].append(row)
            timings[name].append(bm_ms if name == "bm25" else (vec_ms if name == "vector" else bm_ms + vec_ms))

        if reranker is not None:
            started = time.perf_counter()
            candidates = merged[:20]
            pairs = [(query, f"{p.title} {p.brand} {p.category} {p.description}") for p, _ in candidates]
            if pairs:
                scores_r = reranker.predict(pairs, batch_size=4, show_progress_bar=False)
                ranked = sorted(zip([p for p, _ in candidates], scores_r), key=lambda item: -item[1])[: args.top_k]
            else:
                ranked = []
            rr_ms = (time.perf_counter() - started) * 1000
            rr_ids = [p.product_id for p, _ in ranked]
            if relevant is not None:
                rank = first_relevant_rank(rr_ids, relevant)
            else:
                rank = rr_ids.index(expected) + 1 if expected in rr_ids else None
            ranks["rrf+rerank"] = rank
            row = metrics(rank, args.top_k)
            per_stage["rrf+rerank"].append(row)
            per_group[group]["rrf+rerank"].append(row)
            timings["rrf+rerank"].append(bm_ms + vec_ms + rr_ms)

        detail.append({"query": query, "group": group, "expected": expected, "ranks": ranks})

    print()
    print(f"=== 总体（{len(queries)} 条查询，K={args.top_k}）===")
    print(f"{'阶段':>12} | {'Recall@K':>9} | {'MRR':>7} | {'NDCG@K':>8} | {'平均耗时':>10}")
    print("-" * 60)
    for name in stages:
        agg = evaluate(per_stage[name], args.top_k)
        avg_ms = sum(timings[name]) / max(len(timings[name]), 1)
        print(f"{name:>12} | {agg['recall']:9.3f} | {agg['mrr']:7.3f} | {agg['ndcg']:8.3f} | {avg_ms:7.0f} ms")

    print()
    print("=== 分组 ===")
    print(f"{'分组':>22} | " + " | ".join(f"{n:>10}" for n in stages))
    print("-" * (25 + 13 * len(stages)))
    for group in sorted(per_group):
        cells = []
        for name in stages:
            agg = evaluate(per_group[group][name], args.top_k)
            cells.append(f"{agg['recall']:10.3f}")
        print(f"{group:>22} | " + " | ".join(cells))

    print()
    print("=== 逐条明细（rank = 目标商品在该阶段的排名，- 表示未进前 50）===")
    print(f"{'查询':<52} | " + " | ".join(f"{n:>9}" for n in stages))
    for row in detail:
        cells = " | ".join(f"{(row['ranks'].get(n) or '-'):>9}" for n in stages)
        print(f"{row['query'][:50]:<52} | {cells}")

    report = {
        "queries": len(queries), "top_k": args.top_k, "branch_k": args.branch_k,
        "overall": {name: evaluate(per_stage[name], args.top_k) for name in stages},
        "by_group": {g: {name: evaluate(per_group[g][name], args.top_k) for name in stages} for g in per_group},
        "detail": detail,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"报告已写入 {args.output}")
    print(f"结束进程 {rss():.2f} GB | 可用 {available_gb():.1f} GB")


if __name__ == "__main__":
    main()
