"""Resolve a model reference to a local path without touching the network.

The application loads BGE-M3 and bge-reranker-v2-m3 on CPU at startup. Passing a bare
Hugging Face repo id to SentenceTransformer / CrossEncoder makes ``transformers`` issue
a HEAD request (it probes for a PEFT adapter config even when the weights are already
cached), so a machine without internet -- or behind a blocking proxy -- fails at startup.

Resolution order:

1. an explicit local directory (``data/models/<name>``)
2. the Hugging Face cache snapshot, located by walking ``$HF_HOME/hub`` directly
3. ``snapshot_download(..., local_files_only=True)``
4. the bare repo id, which may need the network

Step 2 exists because ``snapshot_download`` refuses a cache that is missing optional
repo files (``.gitattributes``, ``imgs/``, ``colbert_linear.pt``), even though every
weight and tokenizer file the encoder needs is present. Walking the cache layout is
what ``transformers`` does internally, so it always succeeds offline.
"""
from __future__ import annotations

import os
from pathlib import Path


def _cache_snapshot(reference: str) -> str | None:
    home = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    if not home:
        return None
    repo_dir = Path(home) / "hub" / ("models--" + reference.replace("/", "--"))
    revision = repo_dir / "refs" / "main"
    candidates: list[Path] = []
    if revision.is_file():
        candidates.append(repo_dir / "snapshots" / revision.read_text(encoding="utf-8").strip())
    snapshots = repo_dir / "snapshots"
    if snapshots.is_dir():
        candidates.extend(sorted((p for p in snapshots.iterdir() if p.is_dir()), reverse=True))
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return str(candidate)
    return None


def resolve_model(reference: str, local_dir: Path | None = None) -> str:
    if local_dir is not None and Path(local_dir).is_dir():
        return str(local_dir)
    cached = _cache_snapshot(reference)
    if cached is not None:
        return cached
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(reference, local_files_only=True)
    except Exception:
        # 未缓存时回退到 repo id；此时确实需要联网，由调用方承担。
        return reference
