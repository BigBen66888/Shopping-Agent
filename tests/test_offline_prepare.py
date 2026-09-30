import gzip
import json
import sys

from offline_embedding import prepare_full


def test_full_prepare_requires_both_shards_and_publishes_complete_file(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    for number in (1, 2):
        with gzip.open(raw / f"documents.part{number}.jsonl.gz", "wt", encoding="utf-8") as output:
            output.write(json.dumps({"product": {"product_id": str(number), "title": f"item {number}"}}) + "\n")
    monkeypatch.setattr(sys, "argv", ["prepare_full.py", "--work-dir", str(tmp_path), "--min-products", "2"])
    prepare_full.main()
    products = tmp_path / "export" / "products.jsonl"
    assert len(products.read_text(encoding="utf-8").splitlines()) == 2
    assert not products.with_suffix(".jsonl.part").exists()


def test_full_prepare_accepts_official_single_archive(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    with gzip.open(raw / "documents.jsonl.gz", "wt", encoding="utf-8") as output:
        output.write(json.dumps({"product": {"product_id": "one", "title": "item one"}}) + "\n")
    monkeypatch.setattr(sys, "argv", ["prepare_full.py", "--work-dir", str(tmp_path), "--min-products", "1"])
    prepare_full.main()
    assert len((tmp_path / "export" / "products.jsonl").read_text(encoding="utf-8").splitlines()) == 1
