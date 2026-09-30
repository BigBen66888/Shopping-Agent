"""Run actual persisted shopping sessions turn by turn; score final products.

ASR is the fraction of scenarios for which every explicitly annotated product
condition is met. CAR is the mean fraction of conditions met. These use the
benchmark's concepts, but are engineering labels, not official ShoppingBench
scores (which also use a learned title similarity model).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shopping_agent.database import Database
from shopping_agent.memory.context_manager import ContextManager, estimate_tokens
from shopping_agent.service import ShoppingService

BASE = Path(__file__).resolve().parent
CASES = BASE / "datasets/context_multiturn.jsonl"


def condition(product: dict, check: dict) -> bool:
    if not product:
        return False
    if "max_price" in check:
        price = product.get("price")
        return price is not None and float(price) > 0 and float(price) <= check["max_price"]
    title = str(product.get("title", "")).casefold()
    if any(phrase.casefold() in title for phrase in check.get("reject_title_phrases", [])):
        return False
    for key, blocked in check.get("reject_attributes", {}).items():
        values = product.get("attributes", {}).get(key, [])
        values = values if isinstance(values, list) else [values]
        if any(term.casefold() in str(value).casefold() for value in values for term in blocked):
            return False
    # Descriptions often mention compatible accessories and unrelated search
    # terms; scoring those as the product itself produced false positives.
    fields = check.get("fields", ("title", "brand"))
    text = " ".join(str(product.get(key, "")) for key in fields).casefold()
    return any(term.casefold() in text for term in check["any_terms"])


def target_score(products: list[dict], checks: list[dict]) -> dict:
    scored = []
    for product in products:
        values = [condition(product, check) for check in checks]
        scored.append({"product_id": product.get("product_id"), "title": product.get("title"),
                       "price": product.get("price"), "checks": dict(zip((c["name"] for c in checks), values)),
                       "car": sum(values) / len(values), "asr": all(values)})
    return max(scored, key=lambda x: (x["asr"], x["car"]), default={"product_id": None, "car": 0.0, "asr": False, "checks": {}})


def score_case(products: list[dict], case: dict) -> dict:
    if "targets" in case:
        targets = [target_score(products, target["checks"]) for target in case["targets"]]
        distinct = len({item["product_id"] for item in targets if item["product_id"]}) == len(targets)
        return {"asr": bool(distinct and all(item["asr"] for item in targets)),
                "car": statistics.fmean(item["car"] for item in targets), "targets": targets}
    checks = case["checks"]
    first = target_score(products[:1], checks)
    any_card = target_score(products, checks)
    return {"asr": bool(first["asr"]), "car": first["car"], "first": first,
            "any_card_asr": bool(any_card["asr"]), "best_card": any_card}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--context-mode", choices=("compressed", "full"), default="compressed")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-id")
    args = parser.parse_args()
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf8").split("\n") if line]
    if args.case_id:
        cases = [case for case in cases if case["case_id"] == args.case_id]
        if not cases:
            raise ValueError(f"Unknown case ID: {args.case_id}")
    if args.limit:
        cases = cases[:args.limit]
    results = BASE / "results"
    results.mkdir(exist_ok=True)
    service = ShoppingService()
    service.db = Database(results / f"context_{args.label}.sqlite3")
    if args.context_mode == "full":
        service.context_manager = ContextManager(recent_limit=200, token_budget=100000)
    service.prepare()
    rows = []
    output = results / f"context_{args.label}.json"
    for case in cases:
        sid = service.create_session(user_id=f"context_{args.label}_{case['case_id']}_{int(time.time())}")["id"]
        turns = []
        for number, query in enumerate(case["turns"], 1):
            start = time.perf_counter()
            try:
                result = service.recommend_for_session(sid, query)
                state = service.session_state(sid)
                shopping = state["shopping_state"]
                with service.db.connect() as db:
                    messages = db.execute("SELECT id,content FROM messages WHERE session_id=? ORDER BY id", (sid,)).fetchall()
                full_tokens = sum(estimate_tokens(row["content"]) for row in messages)
                retained_tokens = estimate_tokens(shopping.get("context_summary", "")) + sum(
                    estimate_tokens(row["content"]) for row in messages
                    if row["id"] > int(shopping.get("context_compacted_through_id", 0)))
                turns.append({"number": number, "query": query, "answer": result.answer,
                              "products": [p.to_dict() for p in result.products],
                              "tool_calls": result.evidence.get("tool_calls", []),
                              "memory_capture_error": result.evidence.get("memory_capture_error"),
                              "compacted_through_id": shopping.get("context_compacted_through_id", 0),
                              "history_tokens": full_tokens, "retained_tokens": retained_tokens,
                              "retained_ratio": round(retained_tokens / full_tokens, 4) if full_tokens else 1.0,
                              "elapsed_ms": round((time.perf_counter()-start)*1000, 1)})
            except Exception as exc:
                turns.append({"number": number, "query": query, "error": f"{type(exc).__name__}: {str(exc)[:250]}",
                              "elapsed_ms": round((time.perf_counter()-start)*1000, 1)})
                break
            print(f"{case['case_id']} turn {number}/{len(case['turns'])}", flush=True)
        final_products = turns[-1].get("products", []) if turns else []
        scores = score_case(final_products, case)
        rows.append({"case_id": case["case_id"], "session_id": sid,
                     "completed_turns": sum("error" not in turn for turn in turns), "turns_expected": len(case["turns"]),
                     "compressed": any(t.get("compacted_through_id", 0) > 0 for t in turns),
                     "score": scores, "turns": turns})
        all_completed = len(rows) == len(cases) and all(
            row["completed_turns"] == row["turns_expected"] for row in rows)
        output.write_text(json.dumps({"status": "complete" if all_completed else "partial",
                                      "cases": len(cases), "processed": len(rows), "context_mode": args.context_mode,
                                      "asr": statistics.fmean(float(r["score"]["asr"]) for r in rows),
                                      "car": statistics.fmean(r["score"]["car"] for r in rows),
                                      "rows": rows}, ensure_ascii=False, indent=2), encoding="utf8")
    print(output)


if __name__ == "__main__":
    main()
