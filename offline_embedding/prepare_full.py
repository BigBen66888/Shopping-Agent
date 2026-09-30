"""Normalize the entire ShoppingBench corpus; reject an incomplete export."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shopping_agent.dataset import OFFICIAL_ARCHIVE, SPLIT_PARTS, locate_parts, preprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--min-products", type=int, default=2_400_000)
    args = parser.parse_args()
    raw = args.work_dir / "raw"
    sources = locate_parts(raw)
    if not sources:
        parser.error(f"缺少原始数据：{OFFICIAL_ARCHIVE}，或完整的 {', '.join(SPLIT_PARTS)}")
    print("使用原始数据：" + ", ".join(map(str, sources)), flush=True)
    output = args.work_dir / "export" / "products.jsonl"
    temporary = output.with_suffix(".jsonl.part")
    result = preprocess(raw, temporary)
    print(result, flush=True)
    if result["records"] < args.min_products:
        raise SystemExit(f"只有 {result['records']} 条，低于 --min-products={args.min_products}；请核对数据集版本")
    temporary.replace(output)
    print(f"全量商品文件：{output}", flush=True)


if __name__ == "__main__":
    main()
