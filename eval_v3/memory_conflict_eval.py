"""Exercise persistent memory conflict handling with an isolated SQLite file."""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shopping_agent.database import Database
from shopping_agent.service import ShoppingService

RESULTS = Path(__file__).resolve().parent / "results"
CASES = [
    ("color_backpack", "attribute", "用户只买黑色背包", "用户改为只买白色背包", "用户只买黑色运动鞋"),
    ("color_headphones", "attribute", "用户常买黑色耳机", "以后耳机优先白色", "用户只买黑色背包"),
    ("brand_mouse", "brand", "用户长期偏好罗技鼠标", "用户改为偏好雷蛇鼠标", "用户长期偏好索尼耳机"),
    ("budget_headphones", "budget", "用户通常购买 300 元以下的耳机", "用户以后耳机预算为 500 元", "用户键盘预算 800 元"),
    ("size_shoes", "attribute", "用户一直买 42 码鞋", "用户现在改穿 43 码鞋", "用户背包只买黑色"),
    ("cross_kind_color", "attribute", "用户只买黑色背包", "用户改为只买白色背包", "用户长期偏好罗技鼠标"),
]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    service = ShoppingService()
    service.db = Database(RESULTS / "memory_conflicts.sqlite3")
    rows = []
    for name, kind, old, new, unrelated in CASES:
        sid = service.create_session(user_id=f"memory_conflict_{name}_{uuid.uuid4().hex[:8]}")["id"]
        unrelated_kind = "brand" if "罗技" in unrelated or "索尼" in unrelated else "budget" if "预算" in unrelated else "attribute"
        unrelated_saved = service.remember(sid, unrelated_kind, unrelated)
        old_saved = service.remember(sid, kind, old)
        new_kind = "constraint" if name == "cross_kind_color" else kind
        new_saved = service.remember(sid, new_kind, new)
        active = service.long_term_memory(sid)
        contents = {item["content"] for item in active}
        passed = new in contents and old not in contents and unrelated in contents and active[0]["content"] == new
        rows.append({"case_id": name, "passed": passed,
                     "old_id": old_saved["id"], "new_id": new_saved["id"],
                     "unrelated_id": unrelated_saved["id"], "active": active})
    payload = {"cases": len(rows), "passed": sum(row["passed"] for row in rows), "rows": rows}
    output = RESULTS / "memory_conflicts.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"{payload['passed']}/{payload['cases']} -> {output}")


if __name__ == "__main__":
    main()
