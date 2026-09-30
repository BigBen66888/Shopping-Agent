"""Probe the real model's first routing decision without loading product models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shopping_agent.agents.main_agent import MAIN_SYSTEM_PROMPT, READ_DEFS, WRITE_DEFS, DELEGATE_DEFS
from shopping_agent.deepseek import DeepSeekClient

BASE = Path(__file__).resolve().parent
SOURCES = (BASE / "datasets/multiagent_complex_6.jsonl",
           BASE / "datasets/multiagent_parallel_3.jsonl")


def main() -> None:
    client = DeepSeekClient()
    cases = [json.loads(line) for path in SOURCES for line in path.read_text(encoding="utf8").split("\n") if line]
    rows = []
    for case in cases:
        response = client.create([{"role": "system", "content": MAIN_SYSTEM_PROMPT},
                                  {"role": "user", "content": case["query"]}],
                                 READ_DEFS + WRITE_DEFS + DELEGATE_DEFS)
        message = response.choices[0].message
        calls = [{"name": call.function.name, "arguments": call.function.arguments}
                 for call in (message.tool_calls or [])]
        delegated = sum(call["name"] == "delegate_search" for call in calls)
        rows.append({"case_id": case["case_id"], "query": case["query"], "calls": calls,
                     "dual_delegate_first_step": delegated >= 2,
                     "content": message.content or ""})
        print(case["case_id"], [call["name"] for call in calls], flush=True)
    output = BASE / "results/delegation_first_step.json"
    output.write_text(json.dumps({"cases": len(rows), "dual_delegate_rate": sum(r["dual_delegate_first_step"] for r in rows)/len(rows),
                                  "rows": rows}, ensure_ascii=False, indent=2), encoding="utf8")
    print(output)


if __name__ == "__main__":
    main()
