"""Build a CPU-readable FAISS IVF-PQ index with BGE-M3 on Kaggle's dual T4 GPUs."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shopping_agent.dataset import iter_products
from shopping_agent.index_contract import (
    EMBEDDING_MAX_LENGTH, EMBEDDING_MODEL, INDEX_FORMAT, corpus_signature, product_text,
)

# runpy.run_path() (used by the notebook) does not put this directory on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoder import build_model, make_encoder  # noqa: E402


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def count_and_signature(products_path: Path) -> tuple[int, str]:
    count = 0

    def counted():
        nonlocal count
        for product in iter_products(products_path):
            count += 1
            yield product

    signature = corpus_signature(counted(), EMBEDDING_MODEL)
    return count, signature


def training_texts(products_path: Path, size: int) -> list[str]:
    rng = random.Random(42)
    sample: list[str] = []
    for row, product in enumerate(iter_products(products_path)):
        text = product_text(product)
        if row < size:
            sample.append(text)
        else:
            replacement = rng.randrange(row + 1)
            if replacement < size:
                sample[replacement] = text
    return sample


def checkpoint(index, checkpoint_dir: Path, scratch_dir: Path, state: dict) -> None:
    import faiss

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch = scratch_dir / "index.faiss.tmp"
    faiss.write_index(index, str(scratch))
    staged = checkpoint_dir / "index.faiss.part"
    shutil.copy2(scratch, staged)
    staged.replace(checkpoint_dir / "index.faiss")
    atomic_json(checkpoint_dir / "progress.json", {**state, "indexed": index.ntotal, "trained": bool(index.is_trained)})
    print(f"已保存进度：{index.ntotal}/{state['count']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Kaggle 下用 /kaggle/working 内的目录")
    parser.add_argument("--batch-size", type=int, default=32, help="这是每张 GPU 的批量，不是总量")
    parser.add_argument("--encode-chunk", type=int, default=8192)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16", help="T4 的 FP32 仅 8.1 TFLOPS，FP16 TensorCore 为 65 TFLOPS")
    parser.add_argument("--attn", choices=("sdpa", "eager"), default="sdpa", help="sdpa 使用融合注意力内核，明显快于 eager")
    parser.add_argument("--parallel", choices=("auto", "thread", "mp", "single"), default="auto", help="thread=每卡一份模型+多线程；mp=sentence-transformers 多进程；single=只用 cuda:0")
    parser.add_argument("--gpus", type=int, default=0, help="使用的 GPU 数量，0 表示全部可用")
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--pq-m", type=int, default=64)
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=200_000, help="越小越安全，但每次都要写一份索引到磁盘")
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()
    if min(args.batch_size, args.encode_chunk, args.train_size, args.nlist, args.pq_m, args.pq_bits, args.nprobe, args.checkpoint_every, args.progress_every) <= 0:
        parser.error("所有数值参数必须为正数")
    try:
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("先安装 torch、sentence-transformers 和 faiss-cpu") from exc
    if not torch.cuda.is_available():
        raise SystemExit("未检测到 CUDA GPU；请在 Kaggle 的设置里把 Accelerator 选为 GPU T4 x2")

    products_path = args.work_dir / "export" / "products.jsonl"
    model_path = args.work_dir / "export" / "models" / "bge-m3"
    if not products_path.is_file() or not model_path.is_dir():
        parser.error("缺少 products.jsonl 或 BGE-M3 模型；先运行 download.py 和 prepare_full.py")
    count, signature = count_and_signature(products_path)
    state = {
        "format": INDEX_FORMAT, "model": EMBEDDING_MODEL, "max_seq_length": EMBEDDING_MAX_LENGTH,
        "count": count, "corpus_signature": signature, "nlist": args.nlist,
        "pq_m": args.pq_m, "pq_bits": args.pq_bits,
        "precision": args.precision, "attn": args.attn, "batch_size": args.batch_size,
    }
    index_dir = args.work_dir / "export" / "faiss_bge_m3"
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = index_dir / "manifest.json"
    if manifest_path.is_file() and (index_dir / "index.faiss").is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(previous.get(key) == value for key, value in state.items()):
            finished = faiss.read_index(str(index_dir / "index.faiss"))
            if finished.ntotal == count:
                print(f"索引已完成：{count} 条", flush=True)
                return

    # T4 (Turing) 没有 TF32；FP32 只有 8.1 TFLOPS，而 FP16 TensorCore 有 65 TFLOPS。
    # 默认 fp32 会让 GPU 空转，是编码慢的主要原因。
    parallel = args.parallel
    if parallel == "auto":
        parallel = "thread" if torch.cuda.device_count() > 1 else "single"
    encoder, devices = make_encoder(model_path, args.precision, args.attn, args.gpus, args.batch_size, parallel)
    print(f"GPU {torch.cuda.device_count()} 块，使用 {devices}；后端={parallel}；{encoder.describe()}", flush=True)

    # 用一小批真实文本核对 fp16 与 fp32 的向量是否一致，避免静默降质。
    # 参照模型是 fp32（约 2.3 GB），用完必须显式释放，否则会和正式编码抢显存。
    probe = training_texts(products_path, 64)
    fast = encoder.encode(probe)
    reference_model = build_model(model_path, "fp32", args.attn, devices[0])
    try:
        with torch.autocast("cuda", enabled=False):
            reference = np.asarray(
                reference_model.encode(probe, batch_size=args.batch_size, normalize_embeddings=True),
                dtype="float32",
            )
    finally:
        del reference_model
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    similarity = float(np.mean(np.sum(fast * reference, axis=1)))
    print(f"精度自检：{args.precision} 与 fp32 的平均余弦相似度 = {similarity:.6f}", flush=True)
    if similarity < 0.999:
        encoder.close()
        raise SystemExit(f"精度自检未通过（{similarity:.6f} < 0.999）；请改用 --precision fp32")
    del probe, fast, reference

    dimension = encoder.dimension
    if dimension % args.pq_m:
        parser.error(f"向量维度 {dimension} 必须能被 --pq-m={args.pq_m} 整除")
    if count < args.nlist * 39:
        parser.error(f"商品数不足以训练 nlist={args.nlist} 的 IVF；减小 --nlist")
    faiss.omp_set_num_threads(min(8, os.cpu_count() or 1))

    progress_path = args.checkpoint_dir / "progress.json"
    checkpoint_path = args.checkpoint_dir / "index.faiss"
    if progress_path.is_file() and checkpoint_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if any(progress.get(key) != value for key, value in state.items()):
            raise SystemExit("已有断点与当前数据或参数不一致；请换一个 --checkpoint-dir")
        index = faiss.read_index(str(checkpoint_path))
        if index.d != dimension:
            raise SystemExit("断点索引维度与模型不一致")
        if index.ntotal != progress.get("indexed"):
            print(f"进度文件记录 {progress.get('indexed')} 条，按已保存的 FAISS 索引 {index.ntotal} 条续建", flush=True)
        print(f"从第 {index.ntotal} 条继续", flush=True)
    else:
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatIP(dimension), dimension, args.nlist, args.pq_m,
            args.pq_bits, faiss.METRIC_INNER_PRODUCT,
        )

    if not index.is_trained:
        print(f"抽样 {args.train_size} 条训练 IVF-PQ", flush=True)
        sample = training_texts(products_path, args.train_size)
        vectors = encoder.encode(sample)
        index.train(vectors)
        del vectors, sample
        checkpoint(index, args.checkpoint_dir, args.work_dir, state)

    stream = itertools.islice(iter_products(products_path), index.ntotal, None)
    last_checkpoint = time.monotonic()
    last_report = index.ntotal
    report_start = time.monotonic()
    encode_seconds = 0.0
    add_seconds = 0.0
    while True:
        batch = list(itertools.islice(stream, args.encode_chunk))
        if not batch:
            break
        texts = [product_text(product) for product in batch]
        started = time.monotonic()
        vectors = encoder.encode(texts)
        encode_seconds += time.monotonic() - started
        started = time.monotonic()
        index.add(vectors)
        add_seconds += time.monotonic() - started
        if index.ntotal - last_report >= args.progress_every:
            elapsed = max(time.monotonic() - report_start, 1e-6)
            rate = (index.ntotal - last_report) / elapsed
            remaining = (count - index.ntotal) / max(rate, 1e-6) / 3600
            print(
                f"已编码并加入索引：{index.ntotal}/{count}；最近 {index.ntotal - last_report} 条耗时 {elapsed:.0f}s"
                f"（编码 {encode_seconds:.0f}s，FAISS {add_seconds:.0f}s）；{rate:.0f} 条/s，预计还需 {remaining:.1f} 小时",
                flush=True,
            )
            last_report = index.ntotal
            report_start = time.monotonic()
            encode_seconds = 0.0
            add_seconds = 0.0
        if index.ntotal % args.checkpoint_every < len(batch) or time.monotonic() - last_checkpoint >= 1800:
            checkpoint(index, args.checkpoint_dir, args.work_dir, state)
            last_checkpoint = time.monotonic()
    if index.ntotal != count:
        encoder.close()
        raise RuntimeError(f"索引只有 {index.ntotal} 条，商品文件有 {count} 条")
    encoder.close()
    index.nprobe = min(args.nprobe, args.nlist)
    checkpoint(index, args.checkpoint_dir, args.work_dir, state)
    shutil.copy2(checkpoint_path, index_dir / "index.faiss")
    atomic_json(manifest_path, {**state, "index_type": "faiss-ivfpq-ip", "dimension": dimension, "nprobe": index.nprobe})
    print(f"完成：{index.ntotal} 条；CPU 可读取索引位于 {index_dir}", flush=True)


if __name__ == "__main__":
    main()
