"""Verify that every annotated multi-turn target exists in the local catalog."""
from __future__ import annotations

import json
from pathlib import Path

from context_end_to_end import condition

BASE = Path(__file__).resolve().parent
cases = [json.loads(line) for line in (BASE / "datasets/context_multiturn.jsonl").read_text(encoding="utf8").splitlines() if line]
targets = [(case["case_id"], i, target["checks"])
           for case in cases for i, target in enumerate(case.get("targets", [{"checks": case.get("checks", [])}]))]
stats = {(cid, i): {"count": 0, "example": None} for cid, i, _ in targets}
with (BASE.parent / "data/products.jsonl").open(encoding="utf8") as source:
    for line in source:
        product = json.loads(line)
        title = str(product.get("title", "")).casefold()
        for cid, i, checks in targets:
            if not any(term.casefold() in title for term in checks[0]["any_terms"]):
                continue
            if all(condition(product, check) for check in checks):
                row = stats[cid, i]
                row["count"] += 1
                if row["example"] is None:
                    row["example"] = {key: product.get(key) for key in ("product_id", "title", "price")}
output = BASE / "results/context_feasibility.json"
output.write_text(json.dumps({f"{cid}_target{i+1}": row for (cid, i), row in stats.items()},
                             ensure_ascii=False, indent=2), encoding="utf8")
print(output)
