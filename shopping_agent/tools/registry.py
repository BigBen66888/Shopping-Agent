from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable


class ToolPermissionError(PermissionError):
    """Raised when an agent tries to call a tool outside its allow-list."""


class ToolRateLimitError(RuntimeError):
    """Raised when a single runtime exceeds its tool-call window."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    handler: Callable[..., Any]
    read_only: bool
    description: str = ""


class ToolRegistry:
    """Single runtime registry with deny-by-default agent permissions."""

    def __init__(self, rate_limit_per_minute: int = 120) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.permission_violations = 0
        self.failures = 0
        self.rate_limit_per_minute = rate_limit_per_minute
        self.rate_limit_violations = 0
        self._calls: deque[float] = deque()

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        *,
        read_only: bool,
        description: str = "",
    ) -> None:
        if name in self._tools:
            raise ValueError(f"工具已注册: {name}")
        self._tools[name] = ToolSpec(name, handler, read_only, description)

    def invoke(
        self,
        agent: Any,
        name: str,
        /,
        *args: Any,
        retries: int = 1,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        allowed = getattr(agent, "permissions", frozenset())
        if name not in allowed:
            self.permission_violations += 1
            raise ToolPermissionError(f"{type(agent).__name__} 无权调用工具 {name}")
        spec = self._tools.get(name)
        if spec is None:
            raise KeyError(f"工具未注册: {name}")
        now = time.monotonic()
        while self._calls and now - self._calls[0] > 60:
            self._calls.popleft()
        if len(self._calls) >= self.rate_limit_per_minute:
            self.rate_limit_violations += 1
            raise ToolRateLimitError("工具调用超过每分钟限额")
        self._calls.append(now)
        started = time.perf_counter()
        last_error: Exception | None = None
        attempts = 0
        for attempts in range(1, max(retries, 0) + 2):
            try:
                result = spec.handler(*args, **kwargs)
                count = len(result) if isinstance(result, (list, tuple)) else 1
                return result, {
                    "tool": name,
                    "read_only": spec.read_only,
                    "attempts": attempts,
                    "result_count": count,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            except Exception as exc:  # retry boundary is intentionally centralized
                last_error = exc
                self.failures += 1
        assert last_error is not None
        raise last_error

    def describe(self) -> list[dict[str, Any]]:
        return [
            {"name": item.name, "read_only": item.read_only, "description": item.description}
            for item in self._tools.values()
        ]
