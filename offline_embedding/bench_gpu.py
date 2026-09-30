"""Measure BGE-M3 encoding throughput before committing to a full build.

Your T4 baseline was 52 docs/s in fp32 and 201 docs/s in fp16 with sdpa, and raising
the batch size did not help -- the single GPU is already saturated. The remaining
lever is the second T4, so this compares the parallel backends on your own data.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shopping_agent.dataset import iter_products
from shopping_agent.index_contract import EMBEDDING_MAX_LENGTH, product_text

from encoder import Encoder, MultiProcessEncoder


def load_texts(products_path: Path, limit: int) -> list[str]:
    return [product_text(p) for p in itertools.islice(iter_products(products_path), limit)]


def timed(encoder, texts: list[str]) -> tuple[float, float]:
    import torch

    encoder.encode(texts[: min(64, len(texts))])  # warmup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.monotonic()
    encoder.encode(texts)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = max(time.monotonic() - started, 1e-6)
    return elapsed, len(texts) / elapsed


def run(label: str, factory, texts: list[str], total: int, results: list) -> None:
    encoder = None
    try:
        encoder = factory()
        elapsed, rate = timed(encoder, texts)
        hours = total / rate / 3600
        results.append((label, rate, hours))
        print(f"{label:34s} -> {elapsed:6.1f}s  {rate:7.1f} 条/s  全量 {total} 条约 {hours:5.2f} 小时", flush=True)
    except Exception as exc:  # OOM, unsupported pool, etc.
        print(f"{label:34s} -> 失败：{type(exc).__name__}: {str(exc).splitlines()[0][:90]}", flush=True)
    finally:
        try:
            if encoder is not None:
                encoder.close()
        except Exception:
            pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--total", type=int, default=2_746_368)
    parser.add_argument("--batch-size", type=int, default=32, help="每张 GPU 的批量")
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("先安装 torch、sentence-transformers 和 faiss-cpu") from exc
    if not torch.cuda.is_available():
        raise SystemExit("未检测到 CUDA GPU；请在 Kaggle 设置里把 Accelerator 选为 GPU T4 x2")

    products_path = args.work_dir / "export" / "products.jsonl"
    model_path = args.work_dir / "export" / "models" / "bge-m3"
    if not products_path.is_file() or not model_path.is_dir():
        parser.error("缺少 products.jsonl 或 BGE-M3 模型；先运行 download.py 和 prepare_full.py")

    gpus = torch.cuda.device_count()
    print(f"GPU: {gpus} x {torch.cuda.get_device_name(0)}", flush=True)
    print(f"基准样本: {args.sample} 条，max_seq_length={EMBEDDING_MAX_LENGTH}，每卡 batch={args.batch_size}", flush=True)
    texts = load_texts(products_path, args.sample)
    print(f"样本字符长度 p50={np.median([len(t) for t in texts]):.0f}", flush=True)

    results: list[tuple[str, float, float]] = []
    # 基线：单卡 fp16 sdpa，也就是之前在单张 T4 上测得最快的那组。
    run("single fp16 sdpa bs32", lambda: Encoder(model_path, "fp16", "sdpa", ["cuda:0"], args.batch_size), texts, args.total, results)

    if gpus >= 2:
        devices = [f"cuda:{i}" for i in range(gpus)]
        run(f"thread 双卡 fp16 bs{args.batch_size}", lambda: Encoder(model_path, "fp16", "sdpa", devices, args.batch_size), texts, args.total, results)
        run(f"thread 双卡 fp16 bs{args.batch_size * 2}", lambda: Encoder(model_path, "fp16", "sdpa", devices, args.batch_size * 2), texts, args.total, results)
        run(f"mp 双进程 fp16 bs{args.batch_size}", lambda: MultiProcessEncoder(model_path, "fp16", "sdpa", devices, args.batch_size), texts, args.total, results)

    if not results:
        raise SystemExit("所有配置都失败了；请检查 GPU 与显存")
    print("\n=== 汇总 ===", flush=True)
    base = results[0][1]
    for label, rate, hours in results:
        print(f"{label:34s}: {rate:7.1f} 条/s（{rate / base:.2f}x 单卡）全量 {hours:5.2f} 小时", flush=True)
    best = max(results, key=lambda row: row[1])
    print(f"\n最快：{best[0]}，{best[1]:.1f} 条/s，相当于单卡的 {best[1] / base:.2f} 倍", flush=True)
    if best[0].startswith("thread"):
        print(f"建议：--parallel thread --batch-size {best[0].split('bs')[-1]} --precision fp16 --attn sdpa", flush=True)
    elif best[0].startswith("mp"):
        print(f"建议：--parallel mp --batch-size {args.batch_size} --precision fp16 --attn sdpa", flush=True)
    else:
        print("建议：--parallel single（多卡没有收益，检查是否卡在数据读取）", flush=True)


if __name__ == "__main__":
    main()
