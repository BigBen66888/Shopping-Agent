"""DeepSeek OpenAI-compatible SDK adapter."""
from __future__ import annotations

from typing import Any
import time
import json

from openai import OpenAI

from .settings import settings


class DeepSeekClient:
    def __init__(self) -> None:
        if not settings.llm_api_key:
            raise ValueError("缺少 DEEPSEEK_API_KEY（也兼容 LLM_API_KEY）")
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or "https://api.deepseek.com",
            timeout=settings.llm_timeout,
            max_retries=1,
        )
        self.last_usage: dict[str, int] = {}
        self.retry_count = 0

    def _request(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                 tool_choice: str = "auto") -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": settings.llm_model or "deepseek-flash",
            "messages": messages,
            "max_tokens": settings.llm_max_output_tokens,
        }
        if tools:
            request.update({"tools": tools, "tool_choice": tool_choice})
        if settings.deepseek_thinking_enabled and settings.deepseek_reasoning_effort:
            request["reasoning_effort"] = settings.deepseek_reasoning_effort
        if settings.deepseek_thinking_enabled:
            request["extra_body"] = {"thinking": {"type": "enabled"}}
        return request

    def create(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, *,
               max_tokens: int | None = None, tool_choice: str = "auto") -> Any:
        request = self._request(messages, tools, tool_choice=tool_choice)
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        for attempt in range(3):
            response = self.client.chat.completions.create(**request, stream=False)
            choice = response.choices[0]
            message = choice.message
            if choice.finish_reason != "length" and ((message.content or "").strip() or getattr(message, "tool_calls", None)):
                return response
            if attempt < 2:
                self.retry_count += 1
                if choice.finish_reason == "length":
                    request["max_tokens"] = min(int(request["max_tokens"]) * 2, 3200)
                time.sleep(0.2 * (attempt + 1))
        return response

    def stream(self, messages: list[dict[str, Any]]):
        """Yield only public answer deltas; private reasoning fields are ignored."""
        request = self._request(messages)
        request["stream_options"] = {"include_usage": True}
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(**request, stream=True)
                for chunk in response:
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        self.last_usage = {
                            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                        }
                    if not chunk.choices:
                        continue
                    content = getattr(chunk.choices[0].delta, "content", None)
                    if content:
                        yield content
                return
            except Exception:
                if attempt == 2:
                    raise
                self.retry_count += 1
                time.sleep(0.15 * (2 ** attempt))

    def preflight(self) -> dict[str, Any]:
        response = self.create([
            {"role": "system", "content": "You are a helpful assistant. Reply with exactly: DEEPSEEK_OK"},
            {"role": "user", "content": "Connectivity check"},
        ])
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return {
            "ok": True,
            "model": getattr(response, "model", settings.llm_model),
            "content": choice.message.content,
            "finish_reason": choice.finish_reason,
            "usage": usage.model_dump() if usage and hasattr(usage, "model_dump") else None,
        }

    def run_tool_loop(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], handlers: dict[str, Any], max_steps: int = 6, *,
                      serializer: Any = None, require_first_tool: bool = False,
                      progress_callback: Any = None) -> dict[str, Any]:
        """Execute model-selected tools until DeepSeek returns a public final answer."""
        stream = self.run_tool_loop_stream(
            messages, tools, handlers, max_steps, serializer=serializer,
            require_first_tool=require_first_tool,
        )
        while True:
            try:
                event = next(stream)
                if progress_callback:
                    progress_callback(event)
            except StopIteration as completed:
                return completed.value

    def run_tool_loop_stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], handlers: dict[str, Any], max_steps: int = 6, *,
                             serializer: Any = None, require_first_tool: bool = False,
                             progress_queue: Any = None):
        """Yield model narration and real tool results between planning steps.

        ``serializer`` 负责把工具原始返回值裁剪成回灌给模型的紧凑 JSON；
        不传则退化为长度截断。裁剪只影响模型看到的副本，前端仍拿完整结果。
        """
        conversation = list(messages)
        calls = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        model_timings_ms = []
        first_tool_retries = 0
        for step in range(max_steps):
            yield {"type": "stage", "phase": "model_wait", "step": step + 1}
            model_started = time.perf_counter()
            response = self.create(
                conversation, tools,
                # DeepSeek Thinking mode accepts auto, but rejects required.
                tool_choice="auto",
            )
            model_ms = round((time.perf_counter() - model_started) * 1000, 2)
            model_timings_ms.append(model_ms)
            yield {"type": "stage", "phase": "model_done", "step": step + 1,
                   "elapsed_ms": model_ms}
            usage = getattr(response, "usage", None)
            if usage:
                for key in total_usage:
                    total_usage[key] += int(getattr(usage, key, 0) or 0)
            message = response.choices[0].message
            assistant = message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else {"role": "assistant", "content": message.content}
            conversation.append(assistant)
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                if require_first_tool and not calls:
                    first_tool_retries += 1
                    if first_tool_retries >= 2:
                        raise RuntimeError("复杂任务需要委派，但模型未调用子 Agent")
                    conversation.append({"role": "user", "content": "请先调用 delegate_search 或 delegate_shop 执行独立任务，再给出总结。"})
                    continue
                return {"answer": message.content or "", "calls": calls,
                        "usage": total_usage, "model_timings_ms": model_timings_ms}
            if message.content and message.content.strip():
                yield {"type": "model", "content": message.content.strip()}
            def execute(call):
                name = call.function.name
                if name not in handlers:
                    raise PermissionError(f"模型请求了未授权工具: {name}")
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    return call, name, {}, {"error": "工具参数不是有效 JSON，请重新调用并修正参数"}, 0.0
                started = time.perf_counter()
                try:
                    result = handlers[name](**arguments)
                except (KeyError, ValueError, TypeError) as exc:
                    result = {"error": f"{type(exc).__name__}: {str(exc)[:150]}; 请核对当前候选或参数后重试"}
                return call, name, arguments, result, round((time.perf_counter() - started) * 1000, 2)

            names = [call.function.name for call in tool_calls]
            parallel = len(tool_calls) > 1 and all(name in {"delegate_search", "delegate_shop"} for name in names)
            batch_started = time.perf_counter()
            # Wait on futures rather than pool.map: each child can report its
            # own progress and each completed result can reach the UI at once.
            from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
            from queue import Empty
            executed = {}
            batches = [list(enumerate(tool_calls))] if parallel else [
                [(index, call)] for index, call in enumerate(tool_calls)]
            for batch in batches:
                for _, call in batch:
                    yield {"type": "tool_start", "tool": call.function.name}
                with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                    futures = {pool.submit(execute, call): index for index, call in batch}
                    pending = set(futures)
                    last_heartbeat = time.perf_counter()
                    while pending:
                        done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                        if progress_queue is not None:
                            while True:
                                try:
                                    yield progress_queue.get_nowait()
                                except Empty:
                                    break
                        for future in done:
                            index = futures[future]
                            call, name, arguments, result, elapsed_ms = future.result()
                            executed[index] = (call, name, arguments, result, elapsed_ms)
                            batch_ms = round((time.perf_counter() - batch_started) * 1000, 2)
                            calls.append({"id": call.id, "tool": name, "arguments": arguments,
                                          "elapsed_ms": elapsed_ms, "batch_step": step,
                                          "parallel_batch_ms": batch_ms if parallel else None})
                            yield {"type": "tool", "id": call.id, "tool": name, "arguments": arguments,
                                   "result": result, "elapsed_ms": elapsed_ms,
                                   "parallel_batch_ms": batch_ms if parallel else None}
                        if pending and time.perf_counter() - last_heartbeat >= 5:
                            yield {"type": "heartbeat", "elapsed_ms": round((time.perf_counter() - batch_started) * 1000, 2),
                                   "active_tools": [tool_calls[futures[future]].function.name for future in pending]}
                            last_heartbeat = time.perf_counter()
            # Keep tool responses in the model's original call order.
            for index in range(len(tool_calls)):
                call, name, arguments, result, elapsed_ms = executed[index]
                payload = serializer(name, result) if serializer else json.dumps(result, ensure_ascii=False, default=str)[:8000]
                conversation.append({"role": "tool", "tool_call_id": call.id, "content": payload})
        # Give the model one tool-free turn to summarize completed operations.
        # Without this, a valid tool call on the last allowed step becomes an
        # unexplained stream failure instead of a final response.
        conversation.append({"role": "user", "content": "工具执行阶段已结束。请依据已有工具结果总结；未完成的操作请如实说明。"})
        yield {"type": "stage", "phase": "model_wait", "step": max_steps + 1}
        model_started = time.perf_counter()
        response = self.create(conversation, None)
        model_ms = round((time.perf_counter() - model_started) * 1000, 2)
        model_timings_ms.append(model_ms)
        yield {"type": "stage", "phase": "model_done", "step": max_steps + 1,
               "elapsed_ms": model_ms}
        usage = getattr(response, "usage", None)
        if usage:
            for key in total_usage:
                total_usage[key] += int(getattr(usage, key, 0) or 0)
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError("模型在工具调用上限后仍未返回总结")
        return {"answer": answer, "calls": calls, "usage": total_usage,
                "model_timings_ms": model_timings_ms}
