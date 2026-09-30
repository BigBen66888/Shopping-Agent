"""Stream-verify a downloaded Kaggle build artifact without building the full BM25 index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shopping_agent.dataset import iter_products
from shopping_agent.index_contract import EMBEDDING_MAX_LENGTH, EMBEDDING_MODEL, INDEX_FORMAT, corpus_signature


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--query", default="lightweight backpack")
    args = parser.parse_args()
    products = args.data_dir / "products.jsonl"
    index_dir = args.data_dir / "faiss_bge_m3"
    model_dir = args.data_dir / "models" / "bge-m3"
    reranker_dir = args.data_dir / "models" / "bge-reranker-v2-m3"
    # 服务本身在 data/models 不存在时会回退到 Hugging Face 缓存
    # （settings.py 把 HF_HOME 指向 data/model_cache），这里保持同样的回退口径，
    # 否则缓存里已有模型却会被误判为“缺少文件”。
    required = [products, index_dir / "index.faiss", index_dir / "manifest.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("缺少文件：" + ", ".join(missing))
    model_source = str(model_dir) if model_dir.is_dir() else EMBEDDING_MODEL
    reranker_source = str(reranker_dir) if reranker_dir.is_dir() else "BAAI/bge-reranker-v2-m3"
    print(f"编码模型：{model_source}")
    print(f"精排模型：{reranker_source}")

    import faiss
    from sentence_transformers import CrossEncoder, SentenceTransformer

    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    count = 0

    def counted():
        nonlocal count
        for product in iter_products(products):
            count += 1
            yield product

    signature = corpus_signature(counted(), EMBEDDING_MODEL)
    if manifest.get("format") != INDEX_FORMAT or manifest.get("model") != EMBEDDING_MODEL or manifest.get("max_seq_length") != EMBEDDING_MAX_LENGTH:
        raise ValueError("索引格式或模型参数不匹配")
    if manifest.get("count") != count or manifest.get("corpus_signature") != signature:
        raise ValueError("products.jsonl 与离线索引不是同一批数据")
    index = faiss.read_index(str(index_dir / "index.faiss"))
    if index.ntotal != count or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("FAISS 索引数量或距离度量不正确")
    index.nprobe = min(int(manifest.get("nprobe", 64)), index.nlist)
    encoder = SentenceTransformer(model_source, device="cpu")
    encoder.max_seq_length = EMBEDDING_MAX_LENGTH
    vector = np.asarray(encoder.encode([args.query], normalize_embeddings=True), dtype="float32")
    scores, ids = index.search(vector, 20)
    valid = [int(i) for i in ids[0] if i >= 0]
    if len(valid) < 8:
        raise ValueError(f"向量搜索只返回 {len(valid)} 条")
    reranker = CrossEncoder(reranker_source, device="cpu", trust_remote_code=True, max_length=256)
    pair_scores = reranker.predict([(args.query, f"candidate {i}") for i in valid[:8]], batch_size=4, show_progress_bar=False)
    if len(pair_scores) != 8:
        raise ValueError("reranker 未返回 8 个分数")
    print(json.dumps({"verified": True, "products": count, "index_type": manifest.get("index_type"), "dimension": index.d, "search_results": len(valid), "rerank_scores": len(pair_scores)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
