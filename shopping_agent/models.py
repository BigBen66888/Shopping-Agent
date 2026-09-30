from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Product:
    product_id: str
    title: str
    category: str = ""
    price: float = 0.0
    brand: str = ""
    description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    image_url: str = ""
    product_url: str = ""
    sold_count: int = 0
    source: str = "ShoppingBench"

    @classmethod
    def from_record(cls, record: dict[str, Any], *, with_attributes: bool = True) -> "Product":
        p = record.get("product", record)
        return cls(
            product_id=str(p.get("product_id", record.get("id", ""))),
            title=str(p.get("title", "未知商品")),
            category=str(p.get("category", "")),
            price=float(p.get("price") or 0),
            brand=str(p.get("brand", "")),
            description=str(p.get("short_description") or p.get("description") or record.get("contents", "")),
            attributes=(p.get("attributes") or {}) if with_attributes else {},
            image_url=str(p.get("main_image_url", "")),
            product_url=str(p.get("product_url", "")),
            sold_count=int(p.get("sold_count") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Recommendation:
    query: str
    answer: str
    products: list[Product]
    thinking: list[str]
    mode: str = "local"
    evidence: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "answer": self.answer, "products": [p.to_dict() for p in self.products], "thinking": self.thinking, "mode": self.mode, "evidence": self.evidence, "trace": self.trace}
