"""Local metrics for strict one-ID product retrieval and latency."""
from __future__ import annotations

import math


def rank_metrics(ids: list[str], expected: str, k: int = 8) -> dict[str, float]:
    top = ids[:k]
    rank = top.index(expected) + 1 if expected in top else None
    return {
        f"recall@{k}": float(rank is not None),
        f"mrr@{k}": 1.0 / rank if rank else 0.0,
        f"ndcg@{k}": 1.0 / math.log2(rank + 1) if rank else 0.0,
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo), 3)
