"""Recompute context metrics from saved products with the current field labels."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from context_end_to_end import score_case
from shopping_agent.database import Database
from shopping_agent.service import ShoppingService

BASE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    args = parser.parse_args()
    source = BASE / f"results/context_{args.label}.json"
    data = json.loads(source.read_text(encoding="utf8"))
    cases = {row["case_id"]: row for row in (
        json.loads(line) for line in (BASE / "datasets/context_multiturn.jsonl").read_text(encoding="utf8").splitlines() if line)}
    service = ShoppingService()
    service.db = Database(BASE / f"results/context_{args.label}.sqlite3")
    completed = []
    for row in data["rows"]:
        case = cases[row["case_id"]]
        final = row["turns"][-1] if row["turns"] else {}
        row["score"] = score_case(final.get("products", []), case)
        row["complete"] = row["completed_turns"] == row["turns_expected"] and "error" not in final
        if row["complete"]:
            state = service.session_state(row["session_id"])
            row["unwanted_order"] = bool(state["orders"])
            completed.append(row)
    data["metric_definition"] = "Top-card or one-card-per-target field checks; colors may appear in listed description variants."
    data["status"] = "complete" if len(completed) == len(data["rows"]) else "partial"
    data["metrics_completed_only"] = {
        "completion_rate": round(len(completed) / len(data["rows"]), 4) if data["rows"] else 0,
        "asr": round(statistics.fmean(float(row["score"]["asr"]) for row in completed), 4) if completed else None,
        "car": round(statistics.fmean(row["score"]["car"] for row in completed), 4) if completed else None,
        "candidate_asr": round(statistics.fmean(float(row["score"].get("any_card_asr", row["score"]["asr"])) for row in completed), 4) if completed else None,
        "compression_rate": round(statistics.fmean(float(row["compressed"]) for row in completed), 4) if completed else None,
        "mean_retained_ratio": round(statistics.fmean(row["turns"][-1]["retained_ratio"] for row in completed), 4) if completed else None,
        "unwanted_order_count": sum(bool(row["unwanted_order"]) for row in completed),
    }
    output = BASE / f"results/context_{args.label}_rescored.json"
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
    print(data["metrics_completed_only"], output)


if __name__ == "__main__":
    main()
