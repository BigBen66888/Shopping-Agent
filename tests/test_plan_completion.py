from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from shopping_agent.agents.main_agent import MainAgent, READ_DEFS, WRITE_DEFS
from shopping_agent.agents.search_agent import SearchAgent
from shopping_agent.agents.shop_agent import ShopAgent
from shopping_agent.database import Database
from shopping_agent.memory.context_manager import MemoryExtractor
from shopping_agent.models import Product
from shopping_agent.service import ShoppingService


class RetrieverDouble:
    def __init__(self):
        self.calls = []
        self.bm25_calls = []
        self.products = [
            Product("p1", "black laptop backpack", brand="Acme", category="Bags", price=120),
            Product("p2", "blue laptop backpack", brand="Acme", category="Bags", price=220),
        ]

    def search(self, query, top_k, budget):
        self.calls.append((query, top_k, budget))
        rows = [(p, float(10 - i)) for i, p in enumerate(self.products)
                if budget is None or p.price <= budget]
        return rows, {"bm25_count": 2, "vector_count": 2, "rrf_count": len(rows)}

    def search_bm25(self, query, top_k, budget):
        self.bm25_calls.append((query, top_k, budget))
        rows = [(p, float(10 - i)) for i, p in enumerate(self.products)
                if budget is None or p.price <= budget]
        return rows, {"bm25_count": len(rows), "vector_count": 0,
                      "rrf_count": len(rows), "retrieval_mode": "bm25"}


class RerankerDouble:
    def __init__(self):
        self.calls = []

    def rerank(self, query, rows, budget, top_k):
        self.calls.append((query, budget, len(rows)))
        return rows[:top_k]


def make_service(tmp_path):
    service = ShoppingService(data_dir=tmp_path)
    service.db = Database(tmp_path / "shopping.sqlite3")
    retriever = RetrieverDouble()
    reranker = RerankerDouble()
    service.main_agent = MainAgent(SearchAgent(retriever, service, reranker), ShopAgent(service))
    service.agent = SimpleNamespace(hybrid=retriever, reranker=reranker)
    return service, retriever, reranker


def test_search_always_uses_both_recall_and_rerank_with_constraints(tmp_path):
    service, retriever, reranker = make_service(tmp_path)
    session = service.create_session()["id"]
    handlers, shown, _, _ = service.main_agent._runtime(session, object())
    result = handlers["product_search"](
        query="black laptop backpack", budget=200, min_price=100, brand="Acme",
        category="Bags", color="black", size="15 inch", requirements="for commuting",
    )
    assert retriever.calls == [("black laptop backpack Acme Bags black 15 inch for commuting", 20, 200)]
    assert reranker.calls == [("black laptop backpack Acme Bags black 15 inch for commuting", 200, 1)]
    assert result["count"] == 1 and shown[0].product_id == "p1"
    assert result["constraints"]["color"] == "black"


def test_later_search_cannot_drop_an_explicit_budget(tmp_path):
    service, retriever, _ = make_service(tmp_path)
    session = service.create_session()["id"]
    handlers, shown, _, _ = service.main_agent._runtime(session, object())
    assert handlers["product_search"](query="backpack", budget=0.01)["count"] == 0
    assert handlers["product_search"](query="backpack")["count"] == 0
    assert [call[2] for call in retriever.calls] == [0.01, 0.01]
    assert shown == []


def test_product_search_has_per_turn_hard_limit_and_mode_switch(tmp_path):
    service, retriever, reranker = make_service(tmp_path)
    session = service.create_session()["id"]
    handlers, _, _, _ = service.main_agent._runtime(session, object())

    results = [handlers["product_search"](query=f"category {index}")
               for index in range(1, 6)]
    assert [item["search_call"] for item in results] == [1, 2, 3, 4, 5]
    assert [item["pipeline"] for item in results] == [
        "BGE+BM25+rerank", "BGE+BM25+rerank", "BM25", "BM25", "BM25",
    ]
    assert len(retriever.calls) == 2
    assert len(retriever.bm25_calls) == 3
    assert len(reranker.calls) == 2
    with pytest.raises(ValueError, match="5 次硬上限"):
        handlers["product_search"](query="sixth category")

    # Each user turn creates a new runtime, so the next message starts at one.
    next_handlers, _, _, _ = service.main_agent._runtime(session, object())
    restarted = next_handlers["product_search"](query="next user turn")
    assert restarted["search_call"] == 1
    assert restarted["pipeline"] == "BGE+BM25+rerank"


def test_parallel_requests_serialize_cpu_product_retrieval(tmp_path):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    service, _, _ = make_service(tmp_path)
    session = service.create_session()["id"]
    active = 0
    max_active = 0
    modes = []
    state_lock = Lock()

    def guarded_search(query, top_k, budget, retrieval_mode="hybrid", **constraints):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
            modes.append(retrieval_mode)
        time.sleep(0.01)
        with state_lock:
            active -= 1
        return [], {"rrf_count": 0, "retrieval_mode": retrieval_mode}

    service.main_agent.search_agent.search = guarded_search
    handlers, _, _, _ = service.main_agent._runtime(session, object())
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(
            lambda index: handlers["product_search"](query=f"category {index}"),
            range(5),
        ))

    assert max_active == 1
    assert modes == ["hybrid", "hybrid", "bm25", "bm25", "bm25"]
    assert sorted(item["search_call"] for item in results) == [1, 2, 3, 4, 5]


def test_only_one_search_agent_can_be_dispatched_per_turn(tmp_path):
    service, _, _ = make_service(tmp_path)
    session = service.create_session()["id"]

    class Client:
        def run_tool_loop(self, messages, tools, handlers, **kwargs):
            return {"answer": messages[-1]["content"], "calls": []}

    handlers, _, _, _ = service.main_agent._runtime(session, Client())
    assert handlers["delegate_search"]("搜索登山鞋和背包")["answer"] == "搜索登山鞋和背包"
    with pytest.raises(ValueError, match="只能派发一个 SearchAgent"):
        handlers["delegate_search"]("再搜索冲锋衣")


def test_tool_permissions_and_model_isolation(tmp_path):
    service, _, _ = make_service(tmp_path)
    search, shop = service.main_agent.search_agent, service.main_agent.shop_agent
    assert {"get_cart", "get_orders", "get_favorites", "get_memory"} <= search.permissions
    assert not search.permissions.intersection(shop.permissions)
    assert len(READ_DEFS) + len(WRITE_DEFS) == 6
    assert {"product_search", "get_product_details"} <= search.permissions
    assert {"add_to_cart", "buy_now"} <= shop.permissions


def test_memory_only_request_has_no_product_cards_and_reads_real_memory(tmp_path):
    service, retriever, _ = make_service(tmp_path)
    session = service.create_session()["id"]
    service.remember(session, "color", "喜欢黑色")

    class Client:
        def run_tool_loop_stream(self, messages, tools, handlers, **kwargs):
            memory = handlers["read_state"]("memory")
            assert memory["items"][0]["content"] == "喜欢黑色"
            yield {"type": "tool", "tool": "read_state", "arguments": {"kind": "memory"}, "result": memory}
            return {"answer": "你的长期记忆是喜欢黑色。", "calls": ["read_state"], "usage": {}}

    stream = service.main_agent.recommend_stream(Client(), "查看我的长期记忆", session)
    while True:
        try:
            next(stream)
        except StopIteration as completed:
            result = completed.value
            break
    assert not result.products and not retriever.calls
    assert result.answer == "你的长期记忆是喜欢黑色。"


def test_memory_extractor_uses_llm_every_turn_and_skips_one_off_request():
    extractor = MemoryExtractor()

    class Client:
        def __init__(self, response):
            self.response = response
            self.calls = 0

        def create(self, messages):
            self.calls += 1
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=self.response))])

    stable = Client('[{"kind":"attribute","content":"喜欢黑色"}]')
    one_off = Client("[]")
    assert extractor.extract("我喜欢黑色", "", stable)[0]["content"] == "喜欢黑色"
    assert extractor.extract("我要买黑色背包", "", one_off) == []
    assert stable.calls == one_off.calls == 1
    with pytest.raises(RuntimeError):
        extractor.extract("我喜欢黑色", "", None)


def test_stream_api_and_frontend_show_only_relevant_cards(tmp_path, monkeypatch):
    import shopping_agent.api as api_module
    import shopping_agent.service as service_module

    service, retriever, reranker = make_service(tmp_path)
    service.processed = tmp_path / "missing.jsonl"

    class Client:
        def run_tool_loop_stream(self, messages, tools, handlers, **kwargs):
            if "背包" in messages[-1]["content"]:
                found = handlers["product_search"]("black laptop backpack", budget=200)
                yield {"type": "tool", "tool": "product_search",
                       "arguments": {"query": "black laptop backpack", "budget": 200},
                       "result": found}
                return {"answer": "首选黑色笔记本背包，符合通勤和预算要求。需要下单吗？",
                        "calls": ["product_search"], "usage": {}}
            memory = handlers["read_state"]("memory")
            yield {"type": "tool", "tool": "read_state", "arguments": {"kind": "memory"}, "result": memory}
            return {"answer": "你喜欢黑色。", "calls": ["read_state"], "usage": {}}

        def create(self, messages):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="[]"))])

    monkeypatch.setattr(service_module, "DeepSeekClient", Client)
    monkeypatch.setattr(api_module, "service", service)
    client = TestClient(api_module.app)
    session = client.post("/api/sessions").json()["id"]
    page = client.get("/")
    assert page.status_code == 200
    assert "长期记忆" in page.text and "长期偏好</span>" not in page.text
    assert "renderProducts(done.products||[],e.querySelector('.body'))" in page.text
    for query, expected in [("找通勤背包，预算200", 1), ("查看我的长期记忆", 0)]:
        response = client.post("/api/recommend/stream",
                               json={"session_id": session, "query": query})
        events = [json.loads(line) for line in response.text.splitlines()]
        assert events[-1]["type"] == "done"
        assert len(events[-1]["result"]["products"]) == expected
        assert not any("recommendation" in event for event in events)
    assert len(retriever.calls) == 1 and len(reranker.calls) == 1


def test_api_model_failure_is_reported_without_local_fallback(tmp_path, monkeypatch):
    import shopping_agent.api as api_module
    import shopping_agent.service as service_module

    service, retriever, _ = make_service(tmp_path)
    service.processed = tmp_path / "missing.jsonl"

    class FailingClient:
        def run_tool_loop_stream(self, *args, **kwargs):
            raise RuntimeError("gateway unavailable")
            yield

    monkeypatch.setattr(service_module, "DeepSeekClient", FailingClient)
    monkeypatch.setattr(api_module, "service", service)
    client = TestClient(api_module.app)
    session = client.post("/api/sessions").json()["id"]
    response = client.post("/api/recommend/stream",
                           json={"session_id": session, "query": "找背包"})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]["type"] == "error"
    assert "gateway unavailable" in events[-1]["message"]
    assert not retriever.calls

def test_parallel_child_dispatch_uses_isolated_calls():
    from threading import Barrier
    from shopping_agent.deepseek import DeepSeekClient

    barrier = Barrier(2)
    invoked = []

    def child(role):
        barrier.wait(timeout=2)
        invoked.append(role)
        return {"agent": role, "answer": "ok"}

    calls = [
        SimpleNamespace(id="search", function=SimpleNamespace(name="delegate_search", arguments='{"task":"find backpack and boots"}')),
        SimpleNamespace(id="shop", function=SimpleNamespace(name="delegate_shop", arguments='{"task":"remove first cart item"}')),
    ]
    first = SimpleNamespace(
        content="正在并行处理。",
        tool_calls=calls,
        model_dump=lambda exclude_none=True: {"role": "assistant", "content": "正在并行处理。"},
    )
    second = SimpleNamespace(
        content="完成。",
        tool_calls=[],
        model_dump=lambda exclude_none=True: {"role": "assistant", "content": "完成。"},
    )
    messages = iter((first, second))
    client = object.__new__(DeepSeekClient)
    client.create = lambda *_args, **_kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=next(messages))], usage=None)
    stream = client.run_tool_loop_stream(
        [{"role": "user", "content": "search and write"}], [],
        {"delegate_search": lambda task: child(task),
         "delegate_shop": lambda task: child(task)},
        max_steps=2,
    )
    events = list(stream)
    assert set(invoked) == {"find backpack and boots", "remove first cart item"}
    assert [event["type"] for event in events if event["type"] in {"model", "tool"}] == ["model", "tool", "tool"]
    assert len([event for event in events if event["type"] == "tool_start"]) == 2


def test_one_mountain_search_agent_runs_multiple_searches_alongside_shop(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    service, _, _ = make_service(tmp_path)
    session = service.create_session()["id"]
    service.add_cart(session, {"product_id": "old-first", "title": "First", "price": 10})
    service.add_cart(session, {"product_id": "old-second", "title": "Second", "price": 20})
    barrier = Barrier(2)

    class Client:
        def run_tool_loop(self, messages, tools, handlers, **kwargs):
            task = messages[-1]["content"]
            barrier.wait(timeout=3)
            if task == "删除购物车第一个物品":
                cart = handlers["read_state"]("cart")
                assert [item["product_id"] for item in cart["items"]] == ["old-first", "old-second"]
                handlers["shop_action"]("remove_from_cart", product_id=cart["items"][0]["product_id"])
                return {"answer": "已删除第一个物品", "calls": []}
            assert task == "搜索登山背包、登山鞋和冲锋衣"
            backpack = handlers["product_search"]("hiking backpack", budget=150)
            boots = handlers["product_search"]("hiking boots", budget=250)
            jacket = handlers["product_search"]("hiking jacket", budget=300)
            assert backpack["pipeline"] == boots["pipeline"] == "BGE+BM25+rerank"
            assert jacket["pipeline"] == "BM25"
            return {"answer": task, "calls": []}

    handlers, displayed, _, groups = service.main_agent._runtime(session, Client())
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(handlers[name], task) for name, task in (
            ("delegate_search", "搜索登山背包、登山鞋和冲锋衣"),
            ("delegate_shop", "删除购物车第一个物品"),
        )]
        search, shop = [future.result(timeout=5) for future in futures]

    assert search["agent"] == "SearchAgent"
    assert shop["agent"] == "ShopAgent"
    assert len(search["products"]) <= 8
    assert len(groups) == 3 and displayed
    assert [item["product_id"] for item in service.get_cart(session)["items"]] == ["old-second"]


def test_faster_delegate_result_is_streamed_before_slower_delegate_finishes():
    from threading import Event
    from shopping_agent.deepseek import DeepSeekClient

    release_slow = Event()
    calls = [
        SimpleNamespace(id="slow", function=SimpleNamespace(name="delegate_search", arguments='{"task":"search"}')),
        SimpleNamespace(id="fast", function=SimpleNamespace(name="delegate_shop", arguments='{"task":"cart"}')),
    ]
    first = SimpleNamespace(content="开始", tool_calls=calls,
        model_dump=lambda exclude_none=True: {"role": "assistant", "content": "开始"})
    second = SimpleNamespace(content="完成", tool_calls=[],
        model_dump=lambda exclude_none=True: {"role": "assistant", "content": "完成"})
    replies = iter((first, second))
    client = object.__new__(DeepSeekClient)
    client.create = lambda *_args, **_kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=next(replies))], usage=None)

    def slow(task):
        assert release_slow.wait(timeout=3)
        return {"answer": task}

    stream = client.run_tool_loop_stream([{"role": "user", "content": "two tasks"}], [],
        {"delegate_search": slow, "delegate_shop": lambda task: {"answer": task}},
        max_steps=2)
    first_result = next(event for event in stream if event["type"] == "tool")
    assert first_result["tool"] == "delegate_shop"
    release_slow.set()
    assert any(event["type"] == "tool" and event["tool"] == "delegate_search" for event in stream)


def test_shop_child_reads_through_search_proxy(tmp_path):
    service, _, _ = make_service(tmp_path)
    session = service.create_session()["id"]
    service.remember(session, "brand", "喜欢 Acme")

    class Client:
        def run_tool_loop(self, messages, tools, handlers, **kwargs):
            names = {item["function"]["name"] for item in tools}
            assert "read_state" in names and "shop_action" in names
            memory = handlers["read_state"]("memory")
            assert memory["items"][0]["content"] == "喜欢 Acme"
            return {"answer": "已读取偏好", "calls": ["read_state"]}

    handlers, _, _, _ = service.main_agent._runtime(session, Client())
    result = handlers["delegate_shop"]("读取长期记忆")
    assert result["agent"] == "ShopAgent"


def test_explicit_cart_checkout_can_confirm_new_order(tmp_path):
    service, _, _ = make_service(tmp_path)
    session = service.create_session()["id"]
    service.add_cart(session, {"product_id": "p1", "title": "Test item", "price": 20})
    handlers, _, _, _ = service.main_agent._runtime(session, object())
    draft = handlers["shop_action"]("create_order")
    confirmed = handlers["shop_action"]("update_order", order_id=draft["id"], order_action="confirm")
    assert next(order for order in confirmed["orders"] if order["id"] == draft["id"])["status"] == "CONFIRMED"

def test_deleted_long_term_preference_can_be_learned_again(tmp_path):
    service, _, _ = make_service(tmp_path)
    session = service.create_session()["id"]
    first = service.remember(session, "attribute", "喜欢黑色")
    service.forget(session, first["id"])
    restored = service.remember(session, "attribute", "喜欢黑色")
    assert restored["saved"] and restored["id"] == first["id"]
    assert service.get_memory(session)["count"] == 1
