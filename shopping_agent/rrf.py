"""Reciprocal Rank Fusion。"""
from __future__ import annotations

from .models import Product


def fuse(*rankings: list[tuple[Product, float]], k: int = 60, top_k: int = 32,
         weights: tuple[float, ...] | None = None) -> list[tuple[Product, float]]:
    if weights is None:
        weights = (1.0,) * len(rankings)
    if len(weights) != len(rankings) or any(weight <= 0 for weight in weights):
        raise ValueError("RRF weights must be positive and match the branch count")
    scores: dict[str, float] = {}
    products: dict[str, Product] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, (product, _score) in enumerate(ranking, start=1):
            products[product.product_id] = product
            scores[product.product_id] = scores.get(product.product_id, 0.0) + weight / (k + rank)
    ordered = sorted(products, key=lambda pid: (-scores[pid], products[pid].price))
    return [(products[pid], scores[pid]) for pid in ordered[:top_k]]
