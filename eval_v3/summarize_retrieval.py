"""Summarize saved strict-ID retrieval ranks without another corpus scan."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent / "results"
FILES = {
    "title": BASE / "retrieval_title_250_no_rerank.json",
    "partial": BASE / "retrieval_partial_250.json",
    "attribute": BASE / "retrieval_attribute_250_no_rerank.json",
}
summary = {}
for tier, path in FILES.items():
    data = json.loads(path.read_text(encoding="utf8"))
    if data["status"] != "complete" or data["processed"] != 250:
        raise ValueError(f"Incomplete retrieval results: {path}")
    variants = {}
    for name in data["metrics"]:
        variants[name] = {
            f"recall@{k}": round(statistics.fmean(float(row["product_id"] in row["ids"][name][:k])
                                                 for row in data["rows"]), 4)
            for k in (1, 5, 8)
        }
        variants[name].update(data["metrics"][name])
    summary[tier] = {"cases": data["cases"], "candidate_coverage": data["candidate_coverage"],
                     "variants": variants}
output = BASE / "retrieval_summary.json"
output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
print(output)
