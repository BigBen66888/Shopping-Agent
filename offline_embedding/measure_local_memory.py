"""Estimate the local memory cost of the BM25 + FAISS stack from small samples.

Loads only ``--limit`` products (default 200k) at several sizes, fits the per-product
cost, and extrapolates. Never loads the full corpus, so it is safe on a 16 GB laptop.
Model footprints are read from the files on disk instead of loading them.

    .\\.venv\\Scripts\\python.exe offline_embedding\\measure_local_memory.py
    .\\.venv\\Scripts\\python.exe offline_embedding\\measure_local_memory.py --sizes 100000,200000,400000
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import gc
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
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


def process_gb() -> float:
    c = _Counters()
    c.cb = ctypes.sizeof(_Counters)
    _k32.K32GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1024**3


def available_gb() -> float:
    s = _MemoryStatus()
    s.dwLength = ctypes.sizeof(_MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / 1024**3


def total_gb() -> float:
    s = _MemoryStatus()
    s.dwLength = ctypes.sizeof(_MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullTotalPhys / 1024**3


def dir_size_gb(path: Path) -> float:
    if not path.is_dir():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--sizes", default="100000,200000,300000", help="抽样规模，逗号分隔")
    parser.add_argument("--target", type=int, default=2_746_368, help="要外推到的全量条数")
    args = parser.parse_args()

    products_path = args.data_dir / "products.jsonl"
    if not products_path.is_file():
        raise SystemExit(f"缺少 {products_path}")

    from shopping_agent.bm25 import BM25Index
    from shopping_agent.dataset import iter_products

    print(f"系统总内存 {total_gb():.1f} GB | 当前可用 {available_gb():.1f} GB")
    print()
    print(f"{'样本':>9} | {'products':>9} | {'BM25':>8} | {'合计':>8} | {'每条':>9} | {'可用':>7}")
    print("-" * 68)

    rows: list[tuple[int, float, float]] = []
    baseline = process_gb()
    for size in (int(s) for s in args.sizes.split(",")):
        if available_gb() < 1.5:
            print(f"  可用内存不足 1.5 GB，跳过 {size}")
            continue
        started = time.time()
        products = list(itertools.islice(iter_products(products_path), size))
        after_products = process_gb()
        bm25 = BM25Index(products)
        after_bm25 = process_gb()
        n = len(products)
        prod = after_products - baseline
        bm = after_bm25 - after_products
        rows.append((n, prod, bm))
        print(
            f"{n:>9,} | {prod:6.2f} GB | {bm:5.2f} GB | {prod+bm:5.2f} GB | "
            f"{(prod+bm)/n*1e6:6.2f} KB | {available_gb():5.2f} GB   ({time.time()-started:.0f}s)"
        )
        del products, bm25
        gc.collect()

    if not rows:
        raise SystemExit("没有取得有效样本")

    # 用最后一个（最大）样本的每条约数外推，最接近真实规模
    n_last, prod_last, bm_last = rows[-1]
    prod_kb, bm_kb = prod_last / n_last * 1e6, bm_last / n_last * 1e6
    target = args.target
    prod_gb, bm_gb = target * prod_kb / 1e6, target * bm_kb / 1e6

    index_dir = args.data_dir / "faiss_bge_m3"
    faiss_gb = (index_dir / "index.faiss").stat().st_size / 1024**3 if (index_dir / "index.faiss").is_file() else 0.0

    # 模型不加载，直接按磁盘体积估算（CPU 上是 fp32，常驻内存≈权重文件大小）
    bge_dir = args.data_dir / "model_cache" / "hub" / "models--BAAI--bge-m3"
    rer_dir = args.data_dir / "model_cache" / "hub" / "models--BAAI--bge-reranker-v2-m3"
    bge_gb, rer_gb = dir_size_gb(bge_dir), dir_size_gb(rer_dir)

    print()
    print(f"=== 外推到全量 {target:,} 条（按每条约数 {prod_kb:.2f} + {bm_kb:.2f} KB）===")
    print(f"  products 列表        {prod_gb:5.1f} GB")
    print(f"  BM25 索引            {bm_gb:5.1f} GB")
    print(f"  FAISS 索引           {faiss_gb:5.1f} GB")
    print(f"  BGE-M3 (CPU)         {bge_gb:5.1f} GB   [磁盘 {bge_dir.name}]")
    print(f"  bge-reranker-v2-m3   {rer_gb:5.1f} GB   [磁盘 {rer_dir.name}]")
    python_gb = baseline
    print(f"  Python 基线          {python_gb:5.1f} GB")
    total = prod_gb + bm_gb + faiss_gb + bge_gb + rer_gb + python_gb
    print(f"  {'-'*22}")
    print(f"  合计峰值约           {total:5.1f} GB")
    print()
    headroom = total_gb() - total
    print(f"  本机总内存 {total_gb():.1f} GB -> 余量 {headroom:+.1f} GB")
    if headroom < 0.5:
        print("  !! 余量不足，服务启动后极易换页或被 OOM Kill。")
        print(f"     可行规模上限约 {int((total_gb()*0.75 - faiss_gb - bge_gb - rer_gb - python_gb)/((prod_kb+bm_kb)/1e6)):,} 条（按 75% 内存占用）")
    else:
        print("  内存够用，但启动前仍建议关掉占内存的程序。")


if __name__ == "__main__":
    main()
