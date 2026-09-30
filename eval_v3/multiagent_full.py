"""Run the actual agent and persist natural versus explicitly parallel cases."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_v3.multiagent_metrics import evaluate_case
from eval_v3.metrics import percentile
from shopping_agent.database import Database
from shopping_agent.deepseek import DeepSeekClient
from shopping_agent.service import ShoppingService

BASE = Path(__file__).resolve().parent
GROUPS = {"basic": BASE / "datasets/multiagent_20.jsonl",
          "natural_complex": BASE / "datasets/multiagent_complex_6.jsonl",
          "explicit_parallel": BASE / "datasets/multiagent_parallel_3.jsonl"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=tuple(GROUPS) + ("all",), default="all")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    results = BASE / "results"
    results.mkdir(exist_ok=True)
    for label, path in GROUPS.items():
        if args.group not in {"all", label}:
            continue
        cases = [json.loads(line) for line in path.read_text(encoding="utf8").split("\n") if line]
        if args.limit:
            cases = cases[:args.limit]
        # Each evaluation run starts with an empty, isolated cart and order history.
        # Reusing the same SQLite file made earlier runs look like ungrounded writes.
        db_path = results / f"multiagent_{label}_{args.limit or 'all'}.sqlite3"
        db_path.unlink(missing_ok=True)
        service = ShoppingService()
        service.db = Database(db_path)
        service.prepare()
        original_run_tool_loop = DeepSeekClient.run_tool_loop
        rows = []
        for case in cases:
            child_calls = []
            def capture(self, *args, **kwargs):
                result = original_run_tool_loop(self, *args, **kwargs)
                child_calls.extend(result.get("calls", []))
                return result
            DeepSeekClient.run_tool_loop = capture
            try:
                row = evaluate_case(case, service)
            finally:
                DeepSeekClient.run_tool_loop = original_run_tool_loop
            row["child_calls"] = child_calls
            if row.get("task") == "read_and_search":
                row["success"] = (row.get("product_count", 0) > 0
                                  and ("read_state" in row.get("calls", [])
                                       or any(call.get("tool") == "read_state" for call in child_calls))
                                  and row.get("cart_count", 0) == 0 and row.get("order_count", 0) == 0)
            rows.append(row)
        ok = [row for row in rows if row.get("status") == "ok"]
        speedups = [value for row in ok for value in row.get("parallel_speedups", [])]
        write_cases = [row for row in ok if row.get("cart_count") or row.get("order_count")]
        payload = {"cases": len(cases), "completed": len(ok), "rows": rows,
                   "metrics": {"task_success": round(sum(row.get("success", False) for row in ok)/len(cases), 4),
                               "delegation_rate": round(sum(any(name.startswith("delegate_") for name in row["calls"]) for row in ok)/len(ok), 4),
                               "unwanted_buy_count": sum(not row["no_unwanted_buy"] for row in ok),
                               "parallel_batches": len(speedups),
                               "mean_parallel_speedup": round(statistics.fmean(speedups), 3) if speedups else None,
                               "tool_steps_mean": round(statistics.fmean(len(row["calls"]) for row in ok), 2),
                               "grounded_write_rate": round(sum(row["written_grounded"] for row in write_cases)/len(write_cases), 4) if write_cases else None,
                               "p50_ms": percentile([row["elapsed_ms"] for row in rows], .5),
                               "p95_ms": percentile([row["elapsed_ms"] for row in rows], .95)}}
        output = results / f"multiagent_after_{label}{f'_{args.limit}' if args.limit else ''}.json"
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
        print(label, payload["metrics"], output, flush=True)


if __name__ == "__main__":
    main()
