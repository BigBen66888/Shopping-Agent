"""Standard Okapi BM25 over an inverted index."""
from __future__ import annotations

import heapq
import math
import pickle
import re
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .models import Product

# 修改分词或打分逻辑时必须递增，否则会复用不兼容的缓存。
CACHE_VERSION = "bm25-v2"


def _tokens(text: str) -> list[str]:
    """Keep term frequency, unlike the legacy set-based lexical tokenizer."""
    raw = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", text.lower())
    terms = list(raw)
    for chunk in raw:
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            terms.extend(chunk[i:i + 2] for i in range(len(chunk) - 1))
    return terms


class BM25Index:
    def __init__(self, products: Iterable[Product], k1: float = 1.5, b: float = 0.75):
        """Build the index.

        ``products`` may be any sequence; a ``ProductLineStore`` is kept as-is rather
        than copied into a list, so a 2.75M-product corpus never lands in RAM. Prices
        are cached in a compact array during the build so the budget filter never has
        to decode a product line.
        """
        from .product_store import as_product_sequence

        self.products = as_product_sequence(products)
        self.k1, self.b = k1, b
        self.lengths: array = array("I")
        self.prices = array("d")
        self.postings: dict[str, array] = defaultdict(lambda: array("I"))
        for index, product in enumerate(self.products):
            terms = Counter(_tokens(self._text(product)))
            self.lengths.append(sum(terms.values()))
            self.prices.append(float(product.price or 0.0))
            for term, tf in terms.items():
                self.postings[term].extend((index, tf))
        self.avgdl = sum(self.lengths) / max(len(self.lengths), 1)
        self._compute_idf()

    def _compute_idf(self) -> None:
        n = len(self.lengths)
        self.idf = {term: math.log(1 + (n - len(rows) / 2 + 0.5) / (len(rows) / 2 + 0.5)) for term, rows in self.postings.items()}

    @staticmethod
    def _text(p: Product) -> str:
        attrs = " ".join(f"{k} {v}" for k, v in p.attributes.items())
        return f"{p.title} {p.brand} {p.category} {p.description} {attrs}"

    def search(self, query: str, top_k: int = 32, budget: float | None = None) -> list[tuple[Product, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in _tokens(query):
            rows = self.postings.get(term, ())
            for offset in range(0, len(rows), 2):
                index, tf = rows[offset], rows[offset + 1]
                if budget is not None and self.prices[index] > budget:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / (self.avgdl or 1))
                scores[index] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        best = heapq.nlargest(top_k, scores, key=lambda i: (scores[i], -self.prices[i]))
        return [(self.products[i], scores[i]) for i in best]

    # ---- 磁盘缓存：全量 275 万条构建约 8 分钟，缓存后可降到几十秒 ----

    def save_cache(self, path: Path, key: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        payload = {
            "version": CACHE_VERSION,
            "key": key,
            "k1": self.k1,
            "b": self.b,
            "avgdl": self.avgdl,
            "lengths": self.lengths,
            "prices": self.prices,
            "postings": dict(self.postings),
        }
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)

    @classmethod
    def load_cache(cls, path: Path, key: str, products: Iterable[Product]) -> "BM25Index | None":
        """Return a restored index, or None when the cache is missing or stale.

        ``idf`` is intentionally not stored: it is a pure function of the postings and
        the document count, so recomputing it is cheaper than shipping 1.9M floats.
        """
        path = Path(path)
        if not path.is_file():
            return None
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            return None
        if payload.get("version") != CACHE_VERSION or payload.get("key") != key:
            return None
        instance = cls.__new__(cls)
        instance.products = products
        instance.k1 = payload["k1"]
        instance.b = payload["b"]
        instance.avgdl = payload["avgdl"]
        instance.lengths = payload["lengths"]
        instance.prices = payload["prices"]
        instance.postings = payload["postings"]
        instance._compute_idf()
        return instance

    def cache_stats(self) -> dict:
        return {"documents": len(self.lengths), "terms": len(self.postings), "avgdl": round(self.avgdl, 2)}
