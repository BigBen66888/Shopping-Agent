"""用真实 DeepSeek API 做最小连通性检查，不输出密钥。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shopping_agent.deepseek import DeepSeekClient
from shopping_agent.settings import settings


def main() -> int:
    summary = {
        "base_url": settings.llm_base_url,
        "model": settings.llm_model,
        "api_key_configured": bool(settings.llm_api_key),
        "reasoning_effort": settings.deepseek_reasoning_effort,
        "thinking_enabled": settings.deepseek_thinking_enabled,
    }
    try:
        summary.update(DeepSeekClient().preflight())
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        summary.update({
            "ok": False,
            "error_type": type(exc).__name__,
            "status_code": getattr(exc, "status_code", None),
            "error_code": getattr(exc, "code", None),
            "message": str(exc)[:500],
        })
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
