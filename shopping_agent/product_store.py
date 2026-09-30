"""Lazy product access by row index, backed by byte offsets into products.jsonl.

Loading all 2.75M products as Python objects costs about 4.4 GB (1.61 KB each once the
``attributes`` field is skipped). Retrieval only ever needs the few dozen products that
survive ranking, so instead of holding them all we record each line's byte offset once
(about 22 MB) and decode a line only when a row index is actually requested.

The object is sequence-like, so it can be passed straight to code that indexes or
iterates products.
"""
from __future__ import annotations

import json
import threading
from array import array
from pathlib import Path
from typing import Iterator

from .models import Product


class ProductLineStore:
    """Sequence of Products decoded on demand from a JSONL file.

    Offsets are kept in an ``array("q")`` (signed 64-bit) rather than a Python list:
    a list of 2.75M ints costs about 133 MB because every value above 256 is its own
    28-byte object, while the packed array is 22 MB with no per-element overhead.
    """

    def __init__(self, path: Path, *, with_attributes: bool = False, offsets: "array | None" = None):
        self.path = Path(path)
        self.with_attributes = with_attributes
        self.offsets = offsets if offsets is not None else self._scan()
        # Parallel SearchAgent delegates share this store. A shared file handle
        # races between seek() and readline(), so each worker needs its own.
        self._thread_state = threading.local()

    def _scan(self) -> array:
        return array("q", build_offsets(self.path))

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> Product:
        if index < 0:
            index += len(self.offsets)
        if not 0 <= index < len(self.offsets):
            raise IndexError(index)
        handle = getattr(self._thread_state, "handle", None)
        if handle is None:
            handle = self.path.open("rb")
            self._thread_state.handle = handle
        handle.seek(self.offsets[index])
        record = json.loads(handle.readline().decode("utf-8"))
        return Product.from_record(record, with_attributes=self.with_attributes)

    def __iter__(self) -> Iterator[Product]:
        # 逐行流式读取，避免把整份语料同时放进内存
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield Product.from_record(json.loads(line), with_attributes=self.with_attributes)

    def __del__(self) -> None:
        try:
            handle = getattr(self._thread_state, "handle", None)
            if handle is not None:
                handle.close()
        except Exception:
            pass


def as_product_sequence(products):
    """Keep a sequence-like product source as-is; materialize plain iterators.

    ``ProductLineStore`` implements ``__len__`` and ``__getitem__``, so it is used
    directly and the 2.75M products are never copied into a list. Generators such as
    ``iter_products()`` expose neither, so they are materialized exactly as before.

    Doing this by duck-typing instead of a ``materialize`` keyword keeps every existing
    call site -- including test doubles -- working unchanged.
    """
    if hasattr(products, "__len__") and hasattr(products, "__getitem__"):
        return products
    return list(products)


def build_offsets(path: Path) -> array:
    """Record the starting byte position of every non-empty line.

    A plain generator of ints; the caller wraps it in ``array("q")`` so the whole table
    stays at 8 bytes per line instead of paying Python int object overhead.
    """
    offsets = array("q")
    with Path(path).open("rb") as handle:
        position = 0
        for line in handle:
            if line.strip():
                offsets.append(position)
            position += len(line)
    return offsets


def cache_key(path: Path) -> str:
    """Cheap identity for a corpus file, used to invalidate derived caches.

    Deliberately based on size + mtime + line count instead of ``corpus_signature``,
    because hashing the whole corpus takes about a minute.
    """
    path = Path(path)
    stat = path.stat()
    return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
