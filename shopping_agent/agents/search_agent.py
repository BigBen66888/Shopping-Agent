from __future__ import annotations

from typing import Any

from ..hybrid import HybridRetriever
from ..tools.registry import ToolRegistry
from ..tools.search_tools import get_product_details


CATEGORY_ALIASES = {
    "mouse": ("mouse", "mice"),
    "keyboard": ("keyboard",),
    "headphone": ("headphone", "headset", "earphone", "earbud"),
    "backpack": ("backpack", "school bag", "rucksack"),
    "battery": ("battery", "batteries"),
}
COLOR_ALIASES = {"白色": "white", "黑色": "black", "红色": "red", "蓝色": "blue"}


class SearchAgent:
    """All read-only tools, including account state and long-term memory."""

    permissions = frozenset({
        "product_search", "get_product_details", "get_cart", "get_orders",
        "get_favorites", "get_memory",
    })

    def __init__(self, retriever: HybridRetriever, service: Any = None,
                 reranker: Any = None, registry: ToolRegistry | None = None):
        self.retriever = retriever
        self.service = service
        self.reranker = reranker
        self.registry = registry or ToolRegistry()
        if not self.registry.describe():
            self.registry.register("product_search", self._retrieve, read_only=True)
            self.registry.register("get_product_details", get_product_details, read_only=True)
            if service is not None:
                for name, handler in (
                    ("get_cart", service.get_cart), ("get_orders", service.get_orders),
                    ("get_favorites", service.get_favorites), ("get_memory", service.get_memory),
                ):
                    self.registry.register(name, handler, read_only=True)

    def _retrieve(self, query: str, top_k: int, budget: float | None,
                  min_price: float | None = None, brand: str | None = None,
                  category: str | None = None, color: str | None = None,
                  size: str | None = None, requirements: str | None = None,
                  retrieval_mode: str = "hybrid"):
        effective_query = " ".join(str(value) for value in (query, brand, category, color, size, requirements) if value)
        if retrieval_mode == "hybrid":
            rows, stats = self.retriever.search(effective_query, 20, budget)
        elif retrieval_mode == "bm25":
            rows, stats = self.retriever.search_bm25(effective_query, 40, budget)
        else:
            raise ValueError(f"不支持的检索模式: {retrieval_mode}")
        if min_price is not None:
            rows = [(p, score) for p, score in rows if p.price >= min_price]
        if brand:
            rows = [(p, score) for p, score in rows
                    if brand.casefold() in p.brand.casefold() or brand.casefold() in p.title.casefold()]
        if category:
            phrase = category.casefold().strip()
            matched = None
            for name, aliases in CATEGORY_ALIASES.items():
                if name in phrase or any(alias in phrase for alias in aliases):
                    matched = [(p, score) for p, score in rows
                               if any(alias in p.category.split(">")[-1].casefold() for alias in aliases)]
                    break
            if matched is None:
                matched = [(p, score) for p, score in rows if phrase in p.category.casefold()]
            rows = matched
        if color:
            wanted = COLOR_ALIASES.get(color.casefold(), color.casefold())
            rows = [(p, score) for p, score in rows
                    if wanted in f"{p.title} {p.description} {p.attributes}".casefold()]
        if retrieval_mode == "hybrid":
            if self.reranker is None:
                raise RuntimeError("商品搜索缺少 BGE reranker")
            rows = self.reranker.rerank(effective_query, rows, budget, top_k)
        else:
            rows = rows[:top_k]
        return rows, {**stats, "retrieval_mode": retrieval_mode}

    def search(self, query: str, top_k: int = 8, budget: float | None = None,
               retrieval_mode: str = "hybrid", **constraints):
        rows, stats = self.registry.invoke(
            self, "product_search", query, top_k, budget,
            retrieval_mode=retrieval_mode, retries=0, **constraints,
        )[0]
        return rows, {**stats, "retrieval_query": query}

    def details(self, product: Any, include_attributes: bool = False) -> dict:
        return self.registry.invoke(self, "get_product_details", product, include_attributes=include_attributes)[0]

    def cart(self, session_id: str) -> dict:
        return self.registry.invoke(self, "get_cart", session_id)[0]

    def orders(self, session_id: str) -> dict:
        return self.registry.invoke(self, "get_orders", session_id)[0]

    def favorites(self, session_id: str) -> dict:
        return self.registry.invoke(self, "get_favorites", session_id)[0]

    def memory(self, session_id: str) -> dict:
        return self.registry.invoke(self, "get_memory", session_id)[0]

