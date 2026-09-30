"""Exercise the Colab artifact contract with real FAISS and model doubles."""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from shopping_agent.bm25 import BM25Index
from shopping_agent.models import Product


def test_bm25_uses_term_frequency():
    rows = BM25Index([Product("a", "backpack backpack"), Product("b", "backpack")]).search("backpack", 2)
    assert rows[0][0].product_id == "a"
    assert rows[0][1] > rows[1][1]


def test_gpu_export_loads_on_cpu_and_resumes(tmp_path, monkeypatch):
    class Encoder:
        def __init__(self, *args, **kwargs):
            pass

        def get_embedding_dimension(self):
            return 8

        def half(self):
            return self

        def encode(self, texts, **kwargs):
            result = []
            for text in texts:
                number = int(text.split("item ")[-1].split()[0]) if "item " in text else 1
                result.append([1, number / 100, (number % 3) / 10, (number % 5) / 10, 0.1, 0.2, 0.3, 0.4])
            vectors = np.array(result, dtype="float32")
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    class Ranker:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs, **kwargs):
            assert len(pairs) in (8, 20)
            return np.arange(len(pairs), dtype="float32")

    class Autocast:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    # encoder.py 需要 device_count / float16 / autocast / empty_cache / backends.cudnn，
    # 替身必须覆盖生产代码实际用到的全部接口。
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, empty_cache=lambda: None),
        float16="float16",
        autocast=Autocast,
        backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=False)),
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Encoder, CrossEncoder=Ranker))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    import faiss
    from offline_embedding import build_gpu
    import shopping_agent.hybrid as hybrid
    import shopping_agent.reranker as reranker
    from dataclasses import replace

    products = [Product(str(i), f"backpack item {i}", price=float(i)) for i in range(100)]
    export = tmp_path / "export"
    export.mkdir()
    (export / "models" / "bge-m3").mkdir(parents=True)
    (export / "models" / "bge-reranker-v2-m3").mkdir(parents=True)
    for name in ("bge-m3", "bge-reranker-v2-m3"):
        (export / "models" / name / "config.json").write_text("{}")
    with (export / "products.jsonl").open("w", encoding="utf-8") as output:
        for product in products:
            output.write(json.dumps(product.to_dict()) + "\n")
    checkpoint = tmp_path / "checkpoints"
    monkeypatch.setattr(sys, "argv", ["build_gpu.py", "--work-dir", str(tmp_path), "--checkpoint-dir", str(checkpoint), "--train-size", "64", "--nlist", "1", "--pq-m", "2", "--pq-bits", "4", "--encode-chunk", "20", "--checkpoint-every", "20"])
    build_gpu.main()
    manifest = json.loads((export / "faiss_bge_m3" / "manifest.json").read_text())
    assert manifest["count"] == 100 and manifest["index_type"] == "faiss-ivfpq-ip"
    assert faiss.read_index(str(export / "faiss_bge_m3" / "index.faiss")).ntotal == 100

    monkeypatch.setattr(hybrid, "settings", replace(hybrid.settings, data_dir=export))
    retriever = hybrid.HybridRetriever(products)
    rows, stats = retriever.search("backpack", 20)
    assert len(rows) == stats["rrf_count"] == 20
    assert len({p.product_id for p, _ in rows}) == 20
    assert len(reranker.MainAgentReranker().rerank("backpack", rows, top_k=8)) == 8

    with pytest.raises(ValueError, match="不匹配"):
        hybrid.HybridRetriever(products[:-1])

    partial = faiss.read_index(str(checkpoint / "index.faiss"))
    partial.reset()
    partial.add(Encoder().encode([f"backpack item {i}" for i in range(20)]))
    faiss.write_index(partial, str(checkpoint / "index.faiss"))
    progress_path = checkpoint / "progress.json"
    progress = json.loads(progress_path.read_text())
    progress["indexed"] = 20
    progress_path.write_text(json.dumps(progress))
    (export / "faiss_bge_m3" / "manifest.json").unlink()
    build_gpu.main()
    assert faiss.read_index(str(export / "faiss_bge_m3" / "index.faiss")).ntotal == 100
    from offline_embedding import verify_local
    monkeypatch.setattr(sys, "argv", ["verify_local.py", "--data-dir", str(export)])
    verify_local.main()
