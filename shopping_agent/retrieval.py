from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable
from collections import defaultdict

from .models import Product


def _tokens(text: str) -> set[str]:
    # Keep both Chinese chunks and Latin/number terms.
    raw = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", text.lower())
    tokens: set[str] = set(raw)
    for chunk in raw:
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            tokens.update(chunk[i:i + 2] for i in range(max(0, len(chunk) - 1)))
    return tokens


class ProductRetriever:
    def __init__(self, products: Iterable[Product]):
        self.products = list(products)
        self._docs = [_tokens(self._text(p)) for p in self.products]
        self._postings: dict[str, set[int]] = defaultdict(set)
        for index, terms in enumerate(self._docs):
            for term in terms:
                self._postings[term].add(index)

    @staticmethod
    def _text(p: Product) -> str:
        attrs = " ".join(f"{k} {' '.join(map(str, v)) if isinstance(v, list) else v}" for k, v in p.attributes.items())
        return f"{p.title} {p.brand} {p.category} {p.description} {attrs}"

    def search(self, query: str, top_k: int = 8, budget: float | None = None) -> list[tuple[Product, float]]:
        q = _tokens(query)
        scored: list[tuple[Product, float]] = []
        postings = sorted((self._postings[term] for term in q if term in self._postings), key=len)
        candidate_ids = set().union(*postings[:4]) if postings else set()
        for index in candidate_ids:
            product, terms = self.products[index], self._docs[index]
            if budget is not None and product.price > budget:
                continue
            overlap = len(q & terms)
            # Chinese queries are often shorter phrases than catalog titles.
            chinese_bonus = sum(1 for term in q if any(term in doc_term or doc_term in term for doc_term in terms if re.search(r"[\u4e00-\u9fff]", term)))
            phrase = 1.0 if query.lower() in self._text(product).lower() else 0.0
            popularity = min(product.sold_count / 1000.0, 1.0)
            score = (overlap + chinese_bonus * 0.6) / max(len(q), 1) + phrase * 0.35 + popularity * 0.05
            if score > 0:
                scored.append((product, score))
        scored.sort(key=lambda x: (-x[1], x[0].price))
        return scored[:top_k]

    @classmethod
    def from_jsonl(cls, path: Path) -> "ProductRetriever":
        from .dataset import iter_products
        return cls(iter_products(path))

