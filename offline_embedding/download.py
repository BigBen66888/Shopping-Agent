"""Download the official ShoppingBench archive and BGE-M3 for GPU embedding."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shopping_agent.dataset import OFFICIAL_ARCHIVE, OFFICIAL_URL, SPLIT_PARTS, download_dataset, locate_parts

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

KAGGLE_INPUT = Path("/kaggle/input")


def _rglob_named(*names: str) -> list[Path]:
    """在 /kaggle/input 下按文件名递归查找。

    Kaggle 的挂载点可能是 /kaggle/input/<slug>/ 也可能是
    /kaggle/input/datasets/<owner>/<slug>/，深度不固定，所以必须用 rglob 而不是定深 glob。
    只按确切文件名搜索，避免遍历整个 input 目录。
    """
    if not KAGGLE_INPUT.is_dir():
        return []
    found: list[Path] = []
    for name in names:
        found.extend(path for path in KAGGLE_INPUT.rglob(name) if path.is_file())
    return sorted(set(found))


def stage_from_kaggle_input(raw: Path) -> list[Path]:
    """Kaggle 没有 Google Drive；上传的文件在只读的 /kaggle/input/... 下。"""
    names = {OFFICIAL_ARCHIVE, *SPLIT_PARTS}
    found = [path for path in _rglob_named(*names) if path.name in names]
    if not found:
        return []
    raw.mkdir(parents=True, exist_ok=True)
    staged = []
    for source in sorted(found):
        target = raw / source.name
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            print(f"从 Kaggle 输入复制：{source} -> {target}", flush=True)
            shutil.copy2(source, target)
        else:
            print(f"已存在：{target}", flush=True)
        staged.append(target)
    return staged


def find_model_in_kaggle_input(models: Path, target: Path) -> bool:
    """允许把 bge-m3 整个目录放进 Kaggle 数据集，避免每次联网下载 2.3 GB。"""
    if not KAGGLE_INPUT.is_dir():
        return False
    for candidate in sorted(p for p in KAGGLE_INPUT.rglob("bge-m3") if p.is_dir()):
        if (candidate / "modules.json").is_file() and (candidate / "config.json").is_file():
            if target.exists():
                shutil.rmtree(target)
            print(f"从 Kaggle 输入复制模型：{candidate} -> {target}", flush=True)
            shutil.copytree(candidate, target)
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("先安装 huggingface_hub") from exc

    raw = args.work_dir / "raw"
    models = args.work_dir / "export" / "models"
    raw.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    sources = stage_from_kaggle_input(raw) or locate_parts(raw)
    if sources:
        print("使用原始数据：" + ", ".join(map(str, sources)), flush=True)
    else:
        print(f"检查或下载官方原始文件：{OFFICIAL_URL}", flush=True)
        try:
            sources = download_dataset(raw)
        except Exception as exc:
            raise RuntimeError(
                f"GitHub 原始数据下载失败：{type(exc).__name__}: {exc}。"
                "可把 documents.jsonl.gz 或两个 documents.part*.jsonl.gz 作为私有 Kaggle 数据集上传后重试。"
            ) from exc
        print("数据文件：" + ", ".join(map(str, sources)), flush=True)
    repo = "BAAI/bge-m3"
    target = models / "bge-m3"
    required = [target / name for name in ("config.json", "tokenizer.json", "pytorch_model.bin", "modules.json")]
    if all(path.is_file() for path in required):
        print(f"已存在，跳过模型下载：{target}", flush=True)
        return
    if find_model_in_kaggle_input(models, target):
        return
    print(f"正在下载模型：{repo}", flush=True)
    try:
        path = snapshot_download(
            repo_id=repo, local_dir=str(target),
            allow_patterns=["*.json", "*.txt", "*.model", "*.bin", "*.safetensors", "*.py"],
            ignore_patterns=["*onnx*", "*openvino*", "*.gguf"],
            max_workers=2,
        )
    except Exception as exc:
        raise RuntimeError(f"下载模型 {repo} 失败：{type(exc).__name__}: {exc}") from exc
    print(f"模型：{repo} -> {path}", flush=True)


if __name__ == "__main__":
    main()
