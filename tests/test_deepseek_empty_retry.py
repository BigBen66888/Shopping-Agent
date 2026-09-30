from types import SimpleNamespace

from shopping_agent.deepseek import DeepSeekClient


def test_empty_model_reply_is_retried(monkeypatch):
    replies = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=None), finish_reason="stop")]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="完成", tool_calls=None), finish_reason="stop")]),
    ]
    calls = []
    client = object.__new__(DeepSeekClient)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: calls.append(kwargs) or replies.pop(0)
    )))
    client.retry_count = 0
    monkeypatch.setattr("shopping_agent.deepseek.time.sleep", lambda _: None)

    response = client.create([{"role": "user", "content": "测试"}])

    assert response.choices[0].message.content == "完成"
    assert len(calls) == 2
    assert client.retry_count == 1


def test_malformed_tool_arguments_are_returned_to_model():
    bad_call = SimpleNamespace(id="call-1", function=SimpleNamespace(name="search", arguments='{"query":'))
    first = SimpleNamespace(content="", tool_calls=[bad_call],
                            model_dump=lambda **_: {"role": "assistant", "tool_calls": []})
    second = SimpleNamespace(content="已修正", tool_calls=[],
                             model_dump=lambda **_: {"role": "assistant", "content": "已修正"})
    replies = [SimpleNamespace(choices=[SimpleNamespace(message=first, finish_reason="tool_calls")], usage=None),
               SimpleNamespace(choices=[SimpleNamespace(message=second, finish_reason="stop")], usage=None)]
    client = object.__new__(DeepSeekClient)
    client.create = lambda *args, **kwargs: replies.pop(0)

    result = client.run_tool_loop([{"role": "user", "content": "搜索"}], [],
                                  {"search": lambda query: (_ for _ in ()).throw(AssertionError("should not run"))})

    assert result["answer"] == "已修正"
    assert result["calls"][0]["tool"] == "search"


def test_length_truncation_expands_output_budget(monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs.copy())
        length = len(calls) == 1
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="" if length else "摘要", tool_calls=None),
            finish_reason="length" if length else "stop")])

    client = object.__new__(DeepSeekClient)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.retry_count = 0
    monkeypatch.setattr("shopping_agent.deepseek.time.sleep", lambda _: None)

    assert client.create([{"role": "user", "content": "摘要"}], max_tokens=400).choices[0].message.content == "摘要"
    assert [call["max_tokens"] for call in calls] == [400, 800]


def test_invalid_product_id_is_returned_to_model():
    call = SimpleNamespace(id="call-2", function=SimpleNamespace(name="details", arguments='{"product_id":"bad"}'))
    first = SimpleNamespace(content="", tool_calls=[call],
                            model_dump=lambda **_: {"role": "assistant", "tool_calls": []})
    second = SimpleNamespace(content="已改用当前候选", tool_calls=[],
                             model_dump=lambda **_: {"role": "assistant", "content": "已改用当前候选"})
    replies = [SimpleNamespace(choices=[SimpleNamespace(message=first, finish_reason="tool_calls")], usage=None),
               SimpleNamespace(choices=[SimpleNamespace(message=second, finish_reason="stop")], usage=None)]
    client = object.__new__(DeepSeekClient)
    client.create = lambda *args, **kwargs: replies.pop(0)

    result = client.run_tool_loop([{"role": "user", "content": "查看商品"}], [],
                                  {"details": lambda product_id: (_ for _ in ()).throw(KeyError("商品不在当前候选中"))})

    assert result["answer"] == "已改用当前候选"
    assert result["calls"][0]["tool"] == "details"
