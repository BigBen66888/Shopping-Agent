"""Measure int8 dynamic quantization for BGE-M3 / bge-reranker-v2-m3 on CPU.

Each variant runs in its own process so the reported RSS is the real footprint of
that variant alone (Python's allocator does not return freed pages, so measuring
two variants in one process understates the saving).

    .\\.venv\\Scripts\\python.exe offline_embedding\\measure_quantization.py            # 跑全部变体
    .\\.venv\\Scripts\\python.exe offline_embedding\\measure_quantization.py --mode fp32  # 只跑一个
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "model_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(ROOT))

BGE = ROOT / "data" / "model_cache" / "hub" / "models--BAAI--bge-m3" / "snapshots" / "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER = ROOT / "data" / "model_cache" / "hub" / "models--BAAI--bge-reranker-v2-m3" / "snapshots" / "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

TEXTS = [
    "wireless bluetooth earbuds with charging case",
    "laptop battery replacement for acer aspire",
    "cheap plastic storage box for kitchen",
    "孩子用的安全无毒水彩笔套装",
    "waterproof hiking backpack 40l lightweight",
]


class _Counters(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.GetCurrentProcess.restype = wt.HANDLE
_k32.K32GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_Counters), wt.DWORD]


def rss() -> float:
    c = _Counters()
    c.cb = ctypes.sizeof(_Counters)
    _k32.K32GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1024**3


def run_embedding(mode: str) -> dict:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from shopping_agent.index_contract import EMBEDDING_MAX_LENGTH
    from shopping_agent.quantize import quantize_sentence_transformer

    base = rss()
    model = SentenceTransformer(str(BGE), device="cpu")
    model.max_seq_length = EMBEDDING_MAX_LENGTH
    reference = np.asarray(model.encode(TEXTS, normalize_embeddings=True), dtype="float32")
    del model
    import gc

    gc.collect()

    started = time.time()
    model = SentenceTransformer(str(BGE), device="cpu")
    model.max_seq_length = EMBEDDING_MAX_LENGTH
    quantized = False
    if mode != "fp32":
        quantized = quantize_sentence_transformer(model, include_embedding=False)
    load_seconds = time.time() - started
    loaded_rss = rss()

    output = np.asarray(model.encode(TEXTS, normalize_embeddings=True), dtype="float32")
    drift = float(np.mean(np.sum(reference * output, axis=1)))

    bench = TEXTS * 40
    started = time.time()
    model.encode(bench, batch_size=32, normalize_embeddings=True)
    rate = len(bench) / (time.time() - started)
    return {"mode": mode, "quantized": quantized, "rss": loaded_rss - base, "similarity": drift, "rate": rate, "load_s": load_seconds}


def run_reranker(mode: str) -> dict:
    import numpy as np
    from sentence_transformers import CrossEncoder

    from shopping_agent.quantize import quantize_cross_encoder

    base = rss()
    reference_model = CrossEncoder(str(RERANKER), device="cpu", trust_remote_code=True, max_length=256)
    pairs = [(TEXTS[i % len(TEXTS)], f"candidate product number {i} with a short description") for i in range(16)]
    reference = reference_model.predict(pairs, batch_size=4, show_progress_bar=False)
    del reference_model
    import gc

    gc.collect()

    started = time.time()
    model = CrossEncoder(str(RERANKER), device="cpu", trust_remote_code=True, max_length=256)
    quantized = False
    if mode != "fp32":
        quantized = quantize_cross_encoder(model, include_embedding=False)
    load_seconds = time.time() - started
    loaded_rss = rss()

    scores = model.predict(pairs, batch_size=4, show_progress_bar=False)
    order_match = list(np.argsort(-np.asarray(reference))) == list(np.argsort(-np.asarray(scores)))
    max_delta = float(np.max(np.abs(np.asarray(reference) - np.asarray(scores))))

    started = time.time()
    for _ in range(3):
        model.predict(pairs, batch_size=4, show_progress_bar=False)
    ms = (time.time() - started) / 3 * 1000
    return {"mode": mode, "quantized": quantized, "rss": loaded_rss - base, "order_match": order_match, "max_delta": max_delta, "ms": ms, "load_s": load_seconds}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fp32", "int8", "int8+emb", "all"), default="all")
    parser.add_argument("--target", choices=("embedding", "reranker", "both"), default="both")
    parser.add_argument("--child", choices=("fp32", "int8", "int8+emb"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        payload = {}
        if args.target in {"embedding", "both"}:
            payload["embedding"] = run_embedding(args.child)
        if args.target in {"reranker", "both"}:
            payload["reranker"] = run_reranker(args.child)
        import json

        print("RESULT " + json.dumps(payload, ensure_ascii=False))
        return

    modes = ("fp32", "int8", "int8+emb") if args.mode == "all" else (args.mode,)
    results: dict[str, dict] = {}
    for mode in modes:
        print(f"--- 运行变体 {mode} ---", flush=True)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__)), "--child", mode, "--target", args.target],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        line = next((l for l in completed.stdout.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            print(completed.stdout[-1500:], completed.stderr[-800:])
            continue
        import json

        for key, value in json.loads(line[len("RESULT "):]).items():
            results[f"{key}/{mode}"] = value

    print()
    if args.target in {"embedding", "both"}:
        print("=== BGE-M3（查询编码器，仅影响查询侧）===")
        print(f"{'变体':>10} | {'进程内存':>9} | {'vs fp32 相似度':>14} | {'吞吐':>9}")
        print("-" * 56)
        for mode in modes:
            row = results.get(f"embedding/{mode}")
            if not row:
                continue
            print(f"{mode:>10} | {row['rss']:6.2f} GB | {row['similarity']:14.6f} | {row['rate']:6.1f} 条/s")
    if args.target in {"reranker", "both"}:
        print()
        print("=== bge-reranker-v2-m3 ===")
        print(f"{'变体':>10} | {'进程内存':>9} | {'排序一致':>8} | {'最大偏差':>9} | {'延迟':>10}")
        print("-" * 62)
        for mode in modes:
            row = results.get(f"reranker/{mode}")
            if not row:
                continue
            print(f"{mode:>10} | {row['rss']:6.2f} GB | {str(row['order_match']):>8} | {row['max_delta']:9.4f} | {row['ms']:6.1f} ms")

    fp32_e = results.get("embedding/fp32")
    int8_e = results.get("embedding/int8+emb") or results.get("embedding/int8")
    fp32_r = results.get("reranker/fp32")
    int8_r = results.get("reranker/int8+emb") or results.get("reranker/int8")
    if fp32_e and int8_e and fp32_r and int8_r:
        saved = (fp32_e["rss"] - int8_e["rss"]) + (fp32_r["rss"] - int8_r["rss"])
        print()
        print(f"两模型合计节省约 {saved:.2f} GB")


if __name__ == "__main__":
    main()
