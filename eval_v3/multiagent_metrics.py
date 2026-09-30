"""Evaluate one real shopping task against an isolated service session."""
from __future__ import annotations

import time

from shopping_agent.service import ShoppingService


def evaluate_case(case: dict, service: ShoppingService) -> dict:
    sid = service.create_session(user_id=f"eval_v3_{case['case_id']}")["id"]
    started = time.perf_counter()
    try:
        result = service.recommend_for_session(sid, case["query"])
        state = service.session_state(sid)
        tool_calls = result.evidence.get("tool_calls", [])
        calls = [item["tool"] for item in tool_calls]
        task = case["task"]
        cart_n, order_n = len(state["cart"]), len(state["orders"])
        no_unwanted_buy = bool(case["allow_buy"]) or order_n == 0
        if task.endswith("_read"):
            target = {"cart_read": "cart", "orders_read": "orders",
                      "favorites_read": "favorites", "memory_read": "memory"}[task]
            success = any(target in str(item.get("arguments", {})) for item in tool_calls) and cart_n == 0 and order_n == 0
        elif task == "search":
            success = bool(result.products) and cart_n == 0 and order_n == 0
        elif task == "search_two":
            success = len(result.products) >= 2 and cart_n == 0 and order_n == 0
        elif task == "read_and_search":
            success = bool(result.products) and "read_state" in calls and cart_n == 0 and order_n == 0
        elif task == "cart_two":
            success = cart_n >= 2 and order_n == 0
        elif task == "no_result":
            success = not result.products and cart_n == 0 and order_n == 0
        elif task == "cart_write":
            success = cart_n > 0 and order_n == 0
        else:
            success = order_n > 0
        shown = {product.product_id for product in result.products}
        written = {str(product.get("product_id", "")) for product in state["cart"]}
        for order in state["orders"]:
            written.update(str(product.get("product_id", "")) for product in order.get("items", []))
        batches: dict[int, list[dict]] = {}
        for call in tool_calls:
            if call.get("parallel_batch_ms"):
                batches.setdefault(call.get("batch_step"), []).append(call)
        speedups = [round(sum(item["elapsed_ms"] for item in batch) / batch[0]["parallel_batch_ms"], 3)
                    for batch in batches.values() if len(batch) > 1 and batch[0]["parallel_batch_ms"] > 0]
        row = {"case_id": case["case_id"], "status": "ok", "query": case["query"],
               "task": task, "success": bool(success), "no_unwanted_buy": no_unwanted_buy,
               "product_count": len(result.products), "cart_count": cart_n, "order_count": order_n,
               "calls": calls, "tool_calls": tool_calls, "parallel_speedups": speedups,
               "written_grounded": not (written - shown), "answer": result.answer,
               "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}
    except Exception as exc:
        row = {"case_id": case["case_id"], "status": "error", "task": case["task"],
               "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
               "error": f"{type(exc).__name__}: {str(exc)[:250]}"}
    print("multiagent 1/1", flush=True)
    return row
