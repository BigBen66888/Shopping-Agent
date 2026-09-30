from __future__ import annotations

from types import SimpleNamespace

from shopping_agent.agents.delegation_policy import delegation_triggers
from shopping_agent.agents.main_agent import MainAgent
from shopping_agent.deepseek import DeepSeekClient
from shopping_agent.agents.main_agent import tool_result_message
import json


def test_only_independent_multi_task_requests_open_delegation():
    assert delegation_triggers("帮我看看鞋子") == ()
    assert delegation_triggers("帮我找鞋子，然后把它加入购物车") == ()
    assert delegation_triggers("查看购物车") == ()
    assert delegation_triggers("帮我看看鞋子和42码") == ()

    assert "product_and_independent_state" in delegation_triggers(
        "帮我看看鞋子，以及下单购物车的东西"
    )
    assert "multiple_product_searches" in delegation_triggers("帮我看看耳机和鼠标")
    assert "multiple_product_searches" in delegation_triggers("帮我找帽子和鞋子")
    assert "multiple_state_tasks" in delegation_triggers("查看购物车并查看订单")
    assert "product_and_independent_state" in delegation_triggers(
        "帮我看看登山需要买什么，并帮我加入购物车，同时删掉购物车最贵的两个东西"
    )
    assert "product_and_independent_state" in delegation_triggers(
        "我想去登山，帮我看看要买什么，并且删除购物车第一个物品"
    )
    assert delegation_triggers("帮我看看登山需要买什么，并把推荐的东西加入购物车") == ()


def test_main_agent_exposes_direct_tools_only_for_simple_request():
    simple_tools, _, simple_requires_tool = MainAgent._tool_plan("帮我看看鞋子")
    complex_tools, _, complex_requires_tool = MainAgent._tool_plan(
        "帮我看看鞋子，以及下单购物车的东西"
    )
    simple_names = {item["function"]["name"] for item in simple_tools}
    complex_names = {item["function"]["name"] for item in complex_tools}
    assert "product_search" in simple_names
    assert not any(name.startswith("delegate_") for name in simple_names)
    assert complex_names == {"delegate_search", "delegate_shop"}
    assert not simple_requires_tool and complex_requires_tool


def test_runtime_handler_gate_matches_advertised_tools():
    agent = object.__new__(MainAgent)
    handlers = {
        "product_search": lambda **kwargs: None,
        "read_state": lambda **kwargs: None,
        "delegate_search": lambda **kwargs: None,
        "delegate_shop": lambda **kwargs: None,
    }
    agent._runtime = lambda session_id, client, on_progress=None: (handlers, [], [], [])
    observed = []

    class Client:
        def run_tool_loop_stream(self, messages, tools, active_handlers, **kwargs):
            observed.append((
                {item["function"]["name"] for item in tools},
                set(active_handlers),
                kwargs["require_first_tool"],
            ))
            if False:
                yield None
            return {"answer": "完成", "calls": [], "usage": {}}

    list(agent.recommend_stream(Client(), "帮我看看鞋子", "session"))
    list(agent.recommend_stream(Client(), "帮我看看鞋子，以及下单购物车的东西", "session"))
    assert "delegate_search" not in observed[0][0]
    assert "delegate_search" not in observed[0][1]
    assert "product_search" in observed[0][1]
    assert observed[1] == (
        {"delegate_search", "delegate_shop"},
        {"delegate_search", "delegate_shop"},
        True,
    )


def test_direct_main_search_trace_is_not_mislabeled_as_search_agent():
    agent = object.__new__(MainAgent)
    agent._runtime = lambda session_id, client, on_progress=None: ({}, [], [], [])

    class Client:
        def run_tool_loop_stream(self, messages, tools, handlers, **kwargs):
            assert not kwargs["require_first_tool"]
            yield {"type": "tool", "tool": "product_search",
                   "arguments": {"query": "shoes"}, "result": {"count": 8}}
            return {"answer": "找到鞋子。", "calls": [], "usage": {}}

    events = list(agent.recommend_stream(Client(), "帮我看看鞋子", "session"))
    trace = next(item for item in events if item["type"] == "trace")
    assert trace["agent"] == "MainAgent"
    assert trace["content"].startswith("MainAgent · product_search")


def test_delegation_uses_thinking_compatible_auto_choice():
    first_call = SimpleNamespace(
        id="delegate-1",
        function=SimpleNamespace(name="delegate_search", arguments='{"task":"找鞋子"}'),
    )
    first = SimpleNamespace(
        content="", tool_calls=[first_call],
        model_dump=lambda exclude_none=True: {"role": "assistant", "tool_calls": []},
    )
    second = SimpleNamespace(
        content="完成", tool_calls=[],
        model_dump=lambda exclude_none=True: {"role": "assistant", "content": "完成"},
    )
    replies = iter((first, second))
    choices = []
    client = object.__new__(DeepSeekClient)

    def create(*args, **kwargs):
        choices.append(kwargs["tool_choice"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=next(replies))], usage=None,
        )

    client.create = create
    result = client.run_tool_loop(
        [{"role": "user", "content": "帮我看看鞋子，以及下单购物车的东西"}],
        [{"type": "function", "function": {"name": "delegate_search"}}],
        {"delegate_search": lambda task: {"answer": task}},
        require_first_tool=True,
    )
    assert result["answer"] == "完成"
    assert choices == ["auto", "auto"]


def test_first_delegate_is_retried_if_model_answers_without_tool():
    no_tool = SimpleNamespace(content="我来处理", tool_calls=[],
        model_dump=lambda exclude_none=True: {"role": "assistant", "content": "我来处理"})
    call = SimpleNamespace(id="delegate-1",
        function=SimpleNamespace(name="delegate_search", arguments='{"task":"找鞋子"}'))
    with_tool = SimpleNamespace(content="", tool_calls=[call],
        model_dump=lambda exclude_none=True: {"role": "assistant", "tool_calls": []})
    final = SimpleNamespace(content="完成", tool_calls=[],
        model_dump=lambda exclude_none=True: {"role": "assistant", "content": "完成"})
    replies = iter((no_tool, with_tool, final))
    client = object.__new__(DeepSeekClient)
    client.create = lambda *args, **kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=next(replies))], usage=None)
    result = client.run_tool_loop([{"role": "user", "content": "复杂任务"}],
        [{"type": "function", "function": {"name": "delegate_search"}}],
        {"delegate_search": lambda task: {"answer": task}}, require_first_tool=True)
    assert result["answer"] == "完成"
    assert len(result["calls"]) == 1


def test_cart_serializer_keeps_every_price_and_id():
    items = [{"product_id": str(i), "title": f"商品{i}", "price": i * 10,
              "quantity": 1, "description": "irrelevant" * 100} for i in range(18)]
    compact = json.loads(tool_result_message("read_state",
        {"items": items, "count": 18, "total_price": 1530}))
    assert len(compact["items"]) == 18
    assert compact["items"][-1]["product_id"] == "17"
    assert compact["items"][-1]["price"] == 170


def test_delegate_serializer_returns_only_child_output_to_main_agent():
    compact = json.loads(tool_result_message("delegate_search", {
        "agent": "SearchAgent",
        "answer": "找到八件候选商品。",
        "calls": [{"tool": "product_search", "arguments": {"query": "secret context"}}],
        "model_timings_ms": [1234],
        "products": [{"product_id": str(index)} for index in range(8)],
    }))
    assert compact == {"answer": "找到八件候选商品。"}
