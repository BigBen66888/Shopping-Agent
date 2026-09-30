"""Turn the official 250 product cases into reproducible, API-free retrieval inputs.

Only the official product test file and the previously matched local product
records are read. No LLM is used to create or score a query.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "datasets/official_product_250.jsonl"
GOLD = BASE / "datasets/official_gold_products.jsonl"
OUTPUT = BASE / "datasets/product_250_local.jsonl"

STOP = {"a", "an", "and", "the", "with", "for", "of", "in", "on", "to", "by",
        "new", "original", "authentic", "free", "sale", "pcs", "piece", "high",
        "quality", "best", "hot", "fashion", "style", "available", "set"}


def words(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.casefold())
            if len(word) > 1 and word not in STOP]


def partial_title(title: str, brand: str) -> str:
    """A short, deterministic title fragment rather than the exact gold title."""
    tokens = words(title)
    chosen = tokens[: min(5, max(2, len(tokens) // 2))]
    if brand and brand.casefold() not in chosen:
        chosen.insert(0, brand.casefold())
    return " ".join(chosen[:6]) or title[:40]


def attribute_values(product: dict) -> list[str]:
    values: list[str] = []
    for value in product.get("attributes", {}).values():
        values.extend(str(v) for v in value) if isinstance(value, list) else values.append(str(value))
    filtered = (v.casefold().strip() for v in values)
    return list(dict.fromkeys(v for v in filtered if len(v) >= 2 and v not in {"yes", "no", "other", "others"}))[:2]


def attribute_query(title: str, product: dict, fallback: str) -> tuple[str, bool]:
    values = attribute_values(product)
    if not values:
        return fallback, True
    category = product.get("category", "").split(">")[-1].strip()
    title_terms = words(title)[:2]
    parts = ([category] if category else []) + title_terms + values
    return " ".join(dict.fromkeys(str(p).casefold() for p in parts if p)).strip(), False


def build() -> list[dict]:
    # str.splitlines() also splits on U+2028 in a JSON string; JSONL uses only LF.
    source = [json.loads(line) for line in SOURCE.read_text(encoding="utf8").split("\n") if line]
    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf8").split("\n") if line]
    if len(source) != 250 or len(gold) != 250:
        raise ValueError(f"Expected 250 official and matched local cases, got {len(source)} and {len(gold)}")
    rows = []
    for number, (case, local) in enumerate(zip(source, gold), 1):
        reward, product = case["reward"], local["product"]
        pid = str(reward["product_id"])
        if pid != str(product["product_id"]):
            raise ValueError(f"Product ID mismatch at {number}: {pid} != {product['product_id']}")
        title = product["title"].strip()
        partial = partial_title(title, product.get("brand", ""))
        attribute, fallback = attribute_query(title, product, partial)
        rows.append({"case_id": f"O{number:03d}", "product_id": pid,
                     "queries": {"title": title, "partial": partial, "attribute": attribute},
                     "attribute_fallback": fallback,
                     "reward_fields": sorted(reward.keys())})
    return rows


if __name__ == "__main__":
    rows = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf8")
    print(f"Saved {len(rows)} local retrieval cases to {OUTPUT}; "
          f"attribute fallback: {sum(row['attribute_fallback'] for row in rows)}")
