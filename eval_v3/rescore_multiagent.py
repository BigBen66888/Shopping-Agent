"""Recompute read-plus-search success from saved main and child tool calls."""
import json
from pathlib import Path

path = Path(__file__).resolve().parent / "results/multiagent_after_natural_complex.json"
data = json.loads(path.read_text(encoding="utf8"))
for row in data["rows"]:
    if row.get("status") != "ok" or row.get("task") != "read_and_search":
        continue
    observed = "read_state" in row.get("calls", []) or any(
        call.get("tool") == "read_state" for call in row.get("child_calls", []))
    row["success"] = (row.get("product_count", 0) > 0 and observed
                      and row.get("cart_count", 0) == 0 and row.get("order_count", 0) == 0)
data["metrics"]["task_success"] = round(sum(bool(row.get("success")) for row in data["rows"]) / data["cases"], 4)
output = path.with_name("multiagent_after_natural_complex_rescored.json")
output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
print(data["metrics"], output)
