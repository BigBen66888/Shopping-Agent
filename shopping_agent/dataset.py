from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Iterator

from .models import Product

OFFICIAL_ARCHIVE = "documents.jsonl.gz"
SPLIT_PARTS = ("documents.part1.jsonl.gz", "documents.part2.jsonl.gz")
OFFICIAL_URL = "https://media.githubusercontent.com/media/yjwjy/ShoppingBench/main/resources/documents.jsonl.gz"
OFFICIAL_SIZE = 1_471_795_629
OFFICIAL_SHA256 = "a0ba9a8618ce72a7b383a6de31acf6e5bf138873da0d13ad8a1a6b0271e8e9d1"


def iter_records(source: Path) -> Iterator[dict]:
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def locate_parts(data_dir: Path) -> list[Path]:
    candidates = [data_dir, data_dir.parent]
    for base in candidates:
        official = base / OFFICIAL_ARCHIVE
        if official.is_file():
            return [official]
        split = [base / name for name in SPLIT_PARTS]
        if all(path.is_file() for path in split):
            return split
    return []


def download_dataset(data_dir: Path, url: str = OFFICIAL_URL, expected_size: int = OFFICIAL_SIZE, expected_sha256: str = OFFICIAL_SHA256) -> list[Path]:
    """Download the original Git LFS archive, resuming partial HTTP downloads."""
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = locate_parts(data_dir)
    if existing and len(existing) == 2:
        return existing
    if existing and len(existing) == 1 and existing[0].name == OFFICIAL_ARCHIVE and existing[0].stat().st_size == expected_size and _sha256(existing[0]) == expected_sha256:
        return existing
    target = data_dir / OFFICIAL_ARCHIVE
    if target.is_file() and target.stat().st_size == expected_size:
        if _sha256(target) == expected_sha256:
            return [target]
        raise ValueError(f"已有文件校验失败：{target}")
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= expected_size:
        offset = 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        append = offset > 0 and response.status == 206
        with partial.open("ab" if append else "wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
    if partial.stat().st_size != expected_size:
        raise IOError(f"下载未完成：{partial.stat().st_size}/{expected_size} 字节；重新运行会续传")
    if _sha256(partial) != expected_sha256:
        raise ValueError(f"官方数据 SHA-256 校验失败：{partial}")
    partial.replace(target)
    return [target]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(data_dir: Path, output: Path, limit: int | None = None) -> dict:
    """Normalize raw records into compact JSONL used by the local retriever."""
    parts = locate_parts(data_dir)
    if not parts:
        raise FileNotFoundError(f"未找到 {OFFICIAL_ARCHIVE} 或完整的两个本地拆分文件：{data_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    invalid = 0
    with output.open("w", encoding="utf-8") as out:
        for part in parts:
            for record in iter_records(part):
                try:
                    product = Product.from_record(record)
                except (ValueError, TypeError, OverflowError):
                    invalid += 1
                    continue
                if not product.product_id or not product.title:
                    invalid += 1
                    continue
                out.write(json.dumps(product.to_dict(), ensure_ascii=False) + "\n")
                count += 1
                if limit and count >= limit:
                    return {"records": count, "invalid": invalid, "output": str(output), "parts": [str(p) for p in parts]}
    return {"records": count, "invalid": invalid, "output": str(output), "parts": [str(p) for p in parts]}


def iter_products(processed: Path, *, with_attributes: bool = True) -> Iterator[Product]:
    """Stream products. Pass with_attributes=False to save ~0.85 KB per product in RAM.

    写 corpus 时（preprocess）必须保持默认 True，否则 products.jsonl 会丢失 attributes。
    """
    for record in iter_records(processed):
        yield Product.from_record(record, with_attributes=with_attributes)
