"""BGE-M3 dense IVF and Okapi BM25, fused with two-way RRF."""
from __future__ import annotations

import time
from pathlib import Path

from .bm25 import BM25Index
from .models import Product
from .rrf import fuse
from .semantic_vector import FaissEmbeddingIndex
from .settings import settings


class HybridRetriever:
    def __init__(self, products, *, bm25_cache: Path | None = None, bm25_cache_key: str | None = None):
        """Build the two recall branches.

        ``products`` may be a lazy ``ProductLineStore``; both branches then keep that
        reference instead of copying 2.75M Product objects into a list. The BM25 branch
        is restored from ``bm25_cache`` when the key matches, because building it from
        scratch over the full corpus takes about 8 minutes.
        """
        from .product_store import as_product_sequence

        self.products = as_product_sequence(products)
        self.vector = FaissEmbeddingIndex(products, settings.embedding_model, settings.data_dir / "faiss_bge_m3")
        self.bm25 = self._load_or_build_bm25(products, bm25_cache, bm25_cache_key)

    @staticmethod
    def _load_or_build_bm25(products, cache_path: Path | None, cache_key: str | None) -> BM25Index:
        if cache_path is not None and cache_key is not None:
            started = time.perf_counter()
            cached = BM25Index.load_cache(Path(cache_path), cache_key, products)
            if cached is not None:
                print(f"BM25 从缓存载入：{cached.cache_stats()}，耗时 {time.perf_counter()-started:.1f}s", flush=True)
                return cached
            print("BM25 缓存不可用（缺失或数据已变更），开始构建…", flush=True)
        started = time.perf_counter()
        index = BM25Index(products)
        print(f"BM25 构建完成：{index.cache_stats()}，耗时 {time.perf_counter()-started:.1f}s", flush=True)
        if cache_path is not None and cache_key is not None:
            try:
                started = time.perf_counter()
                index.save_cache(Path(cache_path), cache_key)
                print(f"BM25 缓存已写入 {cache_path}，耗时 {time.perf_counter()-started:.1f}s", flush=True)
            except Exception as exc:  # 缓存只是加速手段，失败不应阻止服务启动
                print(f"BM25 缓存写入失败（不影响使用）：{type(exc).__name__}: {exc}", flush=True)
        return index

    def search(self, query: str, top_k: int = 20, budget: float | None = None) -> tuple[list[tuple[Product, float]], dict]:
        # RRF 只向精排提供 20 条；词法和语义分支分别控制开销。
        bm25_k, vector_k = 40, 80
        started = time.perf_counter()
        bm25 = self.bm25.search(query, bm25_k, budget)
        bm25_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        vector = self.vector.search(query, vector_k, budget)
        vector_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        # On the matched official product-250 set, BM25 1.5 / vector 1.0
        # improved strict Recall@8 from .288 to .328 (paired bootstrap > 0).
        merged = fuse(bm25, vector, weights=(1.5, 1.0), top_k=top_k)
        rrf_ms = (time.perf_counter() - started) * 1000
        return merged, {"bm25_count": len(bm25), "vector_count": len(vector), "rrf_count": len(merged),
                        "rrf_weights": {"bm25": 1.5, "vector": 1.0}, "embedding_version": self.vector.model_version,
                        "timing_ms": {"bm25": round(bm25_ms, 2), "vector": round(vector_ms, 2), "rrf": round(rrf_ms, 2)}}

    def search_bm25(self, query: str, top_k: int = 40,
                    budget: float | None = None) -> tuple[list[tuple[Product, float]], dict]:
        """Fast lexical-only recall for later exploratory searches in one turn."""
        started = time.perf_counter()
        rows = self.bm25.search(query, top_k, budget)
        bm25_ms = (time.perf_counter() - started) * 1000
        return rows, {
            "bm25_count": len(rows),
            "vector_count": 0,
            "rrf_count": len(rows),
            "retrieval_mode": "bm25",
            "timing_ms": {"bm25": round(bm25_ms, 2)},
        }
