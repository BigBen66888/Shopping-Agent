"""Load a Kaggle-built FAISS IVF index; CPU only embeds search queries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .index_contract import EMBEDDING_MAX_LENGTH, INDEX_FORMAT, corpus_signature
from .model_paths import resolve_model
from .models import Product
from .quantize import quantize_enabled
from .settings import settings


class FaissEmbeddingIndex:
    def __init__(self, products: Iterable[Product], model_name: str, cache_dir: Path):
        """A ``ProductLineStore`` is kept as a lazy sequence instead of being copied.

        The index only needs the row order (the FAISS row number equals the position in
        products.jsonl), so holding 2.75M Product objects would be waste: a winning row
        is decoded from disk on demand.
        """
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("请安装 sentence-transformers 和 faiss-cpu") from exc

        from .product_store import as_product_sequence

        self.products = as_product_sequence(products)
        self.cache_dir = Path(cache_dir)
        index_path = self.cache_dir / "index.faiss"
        manifest_path = self.cache_dir / "manifest.json"
        if not index_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"缺少离线向量索引：{self.cache_dir}。请导入与 products.jsonl 匹配的 FAISS 索引。"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = corpus_signature(self.products, model_name)
        if manifest.get("format") != INDEX_FORMAT or manifest.get("model") != model_name or manifest.get("max_seq_length") != EMBEDDING_MAX_LENGTH:
            raise ValueError("离线索引的模型或格式与本地配置不一致")
        if manifest.get("count") != len(self.products) or manifest.get("corpus_signature") != expected:
            raise ValueError("离线索引与 products.jsonl 不匹配；请复制同一次 Kaggle 导出的商品文件和索引")

        self.index = faiss.read_index(str(index_path))
        if self.index.ntotal != len(self.products) or self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("FAISS 索引数量或距离度量与商品数据不匹配")
        self.index.nprobe = min(int(manifest.get("nprobe", 64)), self.index.nlist)
        self.model_version = f"{model_name}+{manifest.get('index_type', 'faiss-ivf')}"
        local_model = self.cache_dir.parent / "models" / "bge-m3"
        self.model = SentenceTransformer(resolve_model(model_name, local_model), device="cpu")
        self.model.max_seq_length = EMBEDDING_MAX_LENGTH
        if quantize_enabled(settings.model_quantize):
            from .quantize import quantize_sentence_transformer

            if quantize_sentence_transformer(self.model):
                self.model_version = f"{self.model_version}+int8"

    def search(self, query: str, top_k: int = 50, budget: float | None = None) -> list[tuple[Product, float]]:
        if not self.products or top_k <= 0:
            return []
        vector = np.asarray(self.model.encode([query], normalize_embeddings=True), dtype="float32")
        fetch = min(len(self.products), max(top_k, top_k * (4 if budget is not None else 1)))
        while True:
            scores, indices = self.index.search(vector, fetch)
            rows: list[tuple[Product, float]] = []
            seen: set[str] = set()
            for score, index in zip(scores[0], indices[0]):
                if index < 0:
                    continue
                product = self.products[int(index)]
                if product.product_id in seen or (budget is not None and product.price > budget):
                    continue
                rows.append((product, float(score)))
                seen.add(product.product_id)
                if len(rows) >= top_k:
                    return rows
            if fetch == len(self.products) or fetch >= 4096:
                return rows
            fetch = min(len(self.products), fetch * 2)
