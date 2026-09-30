"""Download the two local retrieval models without storing them in Git."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

MODELS = (
    ("BAAI/bge-m3", "bge-m3"),
    ("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("请先安装 requirements.txt") from exc

    model_root = args.data_dir / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    for repo_id, directory in MODELS:
        target = model_root / directory
        print(f"下载 {repo_id} -> {target}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            allow_patterns=["*.json", "*.txt", "*.model", "*.bin", "*.safetensors", "*.py"],
            ignore_patterns=["*onnx*", "*openvino*", "*.gguf"],
            max_workers=2,
        )
    print("模型下载完成。", flush=True)


if __name__ == "__main__":
    main()
