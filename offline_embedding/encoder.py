"""BGE-M3 GPU encoding backends for the offline index build.

Why not ``torch.nn.DataParallel``: it calls ``replicate()`` on *every* forward pass,
which re-copies ~1.1 GB of fp16 weights to the second GPU for each batch. On a T4
that is a hidden per-batch tax of roughly 100 ms. Keeping one model replica per GPU
and driving them from Python threads avoids it: CUDA ops release the GIL, so the two
GPUs genuinely run concurrently while the weights stay resident.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

import numpy as np


def build_model(model_path: Path, precision: str, attn: str, device: str):
    """Load BGE-M3 onto one device in the requested precision."""
    import torch
    from sentence_transformers import SentenceTransformer

    def load(**kwargs):
        return SentenceTransformer(str(model_path), device=device, model_kwargs=kwargs)

    kwargs = {"attn_implementation": attn}
    if precision == "fp16":
        kwargs["dtype"] = torch.float16
    try:
        model = load(**kwargs)
    except TypeError:
        # transformers < 4.56 spells it torch_dtype; a truly unknown kwarg would also land here.
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = torch.float16
        model = load(**kwargs)
    model.max_seq_length = _max_length()
    if precision == "fp16":
        model.half()
    return model


def _max_length() -> int:
    from shopping_agent.index_contract import EMBEDDING_MAX_LENGTH

    return EMBEDDING_MAX_LENGTH


def split_contiguous(texts: Sequence[str], parts: int) -> list[list[str]]:
    """Split in order so concatenating the results reproduces the input order."""
    total = len(texts)
    size = -(-total // parts)  # ceil
    return [list(texts[i : i + size]) for i in range(0, total, size)] or [[] for _ in range(parts)]


class Encoder:
    """Encode batches across one or more GPUs, returning float32 normalized vectors."""

    def __init__(self, model_path: Path, precision: str, attn: str, devices: Sequence[str], batch_size: int):
        import torch

        self.torch = torch
        self.precision = precision
        self.batch_size = batch_size
        self.devices = list(devices)
        self.models = [build_model(model_path, precision, attn, device) for device in self.devices]
        torch.backends.cudnn.benchmark = True
        dimension = self.models[0].get_embedding_dimension() if hasattr(self.models[0], "get_embedding_dimension") else self.models[0].get_sentence_embedding_dimension()
        self.dimension = int(dimension)

    def describe(self) -> str:
        return f"{self.precision}/{len(self.devices)}卡/每卡bs{self.batch_size}"

    def _encode_one(self, index: int, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.precision == "fp16"):
            encoded = self.models[index].encode(
                list(texts), batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
            )
        return np.asarray(encoded, dtype="float32")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(self.models) == 1:
            return self._encode_one(0, texts)
        chunks = split_contiguous(texts, len(self.models))
        results: list[np.ndarray | None] = [None] * len(self.models)
        errors: list[BaseException] = []

        def run(index: int) -> None:
            try:
                results[index] = self._encode_one(index, chunks[index])
            except BaseException as exc:  # surfaced to the caller after join
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(len(self.models))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        return np.concatenate(results, axis=0)

    def close(self) -> None:
        del self.models
        self.models = []


class MultiProcessEncoder:
    """sentence-transformers' own multi-process pool as an alternative backend.

    Useful when the Python/tokenizer side, not the GPU, is the bottleneck: each worker
    gets its own interpreter and therefore its own GIL.
    """

    def __init__(self, model_path: Path, precision: str, attn: str, devices: Sequence[str], batch_size: int):
        import torch

        self.torch = torch
        self.precision = precision
        self.batch_size = batch_size
        self.devices = list(devices)
        self.model = build_model(model_path, precision, attn, devices[0])
        self.dimension = int(
            self.model.get_embedding_dimension()
            if hasattr(self.model, "get_embedding_dimension")
            else self.model.get_sentence_embedding_dimension()
        )
        self.pool = self.model.start_multi_process_pool(target_devices=self.devices)

    def describe(self) -> str:
        return f"{self.precision}/mp{len(self.devices)}进程/bs{self.batch_size}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        encoded = self.model.encode_multi_process(
            list(texts), self.pool, batch_size=self.batch_size, normalize_embeddings=True
        )
        return np.asarray(encoded, dtype="float32")

    def close(self) -> None:
        try:
            self.model.stop_multi_process_pool(self.pool)
        finally:
            del self.model


def make_encoder(model_path: Path, precision: str, attn: str, gpus: int, batch_size: int, parallel: str):
    """Return an encoder plus the device list actually used."""
    import torch

    available = torch.cuda.device_count()
    if parallel == "single" or available <= 1:
        return Encoder(model_path, precision, attn, ["cuda:0"], batch_size), ["cuda:0"]
    count = max(1, min(gpus or available, available))
    devices = [f"cuda:{i}" for i in range(count)]
    if parallel == "mp":
        return MultiProcessEncoder(model_path, precision, attn, devices, batch_size), devices
    return Encoder(model_path, precision, attn, devices, batch_size), devices
