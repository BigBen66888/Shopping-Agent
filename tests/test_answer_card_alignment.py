from __future__ import annotations

from shopping_agent.agents.main_agent import MainAgent
from shopping_agent.models import Product


def test_answer_cited_candidate_appears_in_eight_cards():
    agent = object.__new__(MainAgent)
    displayed = [Product(str(1000000000 + i), f"Candidate {i}") for i in range(10)]
    agent._runtime = lambda session_id, client, on_progress=None: ({}, displayed, [], [displayed])

    class Client:
        def run_tool_loop_stream(self, messages, tools, handlers, **kwargs):
            if False:
                yield None
            return {"answer": "推荐商品 ID 1000000009", "calls": [], "usage": {}}

    stream = agent.recommend_stream(Client(), "find product", "session")
    while True:
        try:
            next(stream)
        except StopIteration as completed:
            result = completed.value
            break
    assert len(result.products) == 8
    assert result.products[0].product_id == "1000000009"


def test_written_candidate_appears_in_eight_cards_without_answer_citation():
    agent = object.__new__(MainAgent)
    displayed = [Product(str(1000000000 + i), f"Candidate {i}") for i in range(12)]
    agent._runtime = lambda session_id, client, on_progress=None: ({}, displayed, ["1000000010", "1000000011"], [displayed])

    class Client:
        def run_tool_loop_stream(self, messages, tools, handlers, **kwargs):
            if False:
                yield None
            return {"answer": "两件商品已加入购物车", "calls": [], "usage": {}}

    stream = agent.recommend_stream(Client(), "add two products", "session")
    while True:
        try:
            next(stream)
        except StopIteration as completed:
            result = completed.value
            break
    assert [product.product_id for product in result.products[:2]] == ["1000000010", "1000000011"]


def test_two_independent_searches_both_appear_in_cards():
    agent = object.__new__(MainAgent)
    mice = [Product(str(1000000000 + i), f"Mouse {i}") for i in range(8)]
    keyboards = [Product(str(2000000000 + i), f"Keyboard {i}") for i in range(8)]
    agent._runtime = lambda session_id, client, on_progress=None: ({}, mice + keyboards, [], [mice, keyboards])

    class Client:
        def run_tool_loop_stream(self, messages, tools, handlers, **kwargs):
            if False:
                yield None
            return {"answer": "分别推荐鼠标和键盘", "calls": [], "usage": {}}

    stream = agent.recommend_stream(Client(), "two products", "session")
    while True:
        try:
            next(stream)
        except StopIteration as completed:
            result = completed.value
            break
    assert len(result.products) == 8
    assert sum(product.title.startswith("Mouse") for product in result.products) == 4
    assert sum(product.title.startswith("Keyboard") for product in result.products) == 4
