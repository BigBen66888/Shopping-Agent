from __future__ import annotations

from typing import Any

from ..tools.registry import ToolRegistry


class ShopAgent:
    """Write-only shopping tools; all reads belong to SearchAgent."""

    permissions = frozenset({
        "add_to_cart", "remove_from_cart", "buy_now", "add_favorite",
        "remove_favorite", "create_order", "update_order", "rebuy",
    })

    def __init__(self, service: Any, registry: ToolRegistry | None = None):
        self.service = service
        self.registry = registry or ToolRegistry()
        if not self.registry.describe() and hasattr(service, "add_cart"):
            for name, handler in (
                ("add_to_cart", service.add_cart),
                ("remove_from_cart", service.remove_cart),
                ("buy_now", service.buy_now),
                ("add_favorite", service.add_favorite),
                ("remove_favorite", service.remove_favorite),
                ("create_order", service.create_order),
                ("update_order", service.update_order),
                ("rebuy", service.rebuy),
            ):
                self.registry.register(name, handler, read_only=False)

    def add_to_cart(self, session_id: str, product: dict) -> dict:
        return self.registry.invoke(self, "add_to_cart", session_id, product)[0]

    def remove_from_cart(self, session_id: str, product_id: str) -> dict:
        return self.registry.invoke(self, "remove_from_cart", session_id, product_id)[0]

    def buy_now(self, session_id: str, product: dict) -> dict:
        return self.registry.invoke(self, "buy_now", session_id, product)[0]

    def favorite(self, session_id: str, product: dict) -> dict:
        return self.registry.invoke(self, "add_favorite", session_id, product)[0]

    def unfavorite(self, session_id: str, product_id: str) -> dict:
        return self.registry.invoke(self, "remove_favorite", session_id, product_id)[0]

    def create_order(self, session_id: str) -> dict:
        return self.registry.invoke(self, "create_order", session_id)[0]

    def update_order(self, session_id: str, order_id: str, action: str) -> dict:
        return self.registry.invoke(self, "update_order", session_id, order_id, action)[0]

    def rebuy(self, session_id: str, order_id: str) -> dict:
        return self.registry.invoke(self, "rebuy", session_id, order_id)[0]
