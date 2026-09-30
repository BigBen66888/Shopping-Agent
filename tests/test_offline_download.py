import hashlib
import io
import sys

from offline_embedding import download
from shopping_agent import dataset


def _existing_models(work_dir):
    models = work_dir / "export" / "models"
    target = models / "bge-m3"
    target.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "pytorch_model.bin", "modules.json"):
        (target / name).write_bytes(b"existing model")


def test_download_reuses_users_two_split_shards(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in dataset.SPLIT_PARTS:
        (raw / name).write_bytes(b"existing shard")
    _existing_models(tmp_path)

    import huggingface_hub
    monkeypatch.setattr(download, "download_dataset", lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected data download")))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected model download")))
    monkeypatch.setattr(sys, "argv", ["download.py", "--work-dir", str(tmp_path)])
    download.main()
    assert not (tmp_path / "export" / "models" / "bge-reranker-v2-m3").exists()


def test_official_git_lfs_archive_resumes_and_checks_sha(tmp_path, monkeypatch):
    archive = b"original compressed archive"
    target = tmp_path / dataset.OFFICIAL_ARCHIVE
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_bytes(archive[:8])

    class Response(io.BytesIO):
        status = 206

    def urlopen(request, **_kwargs):
        assert request.headers["Range"] == "bytes=8-"
        return Response(archive[8:])

    monkeypatch.setattr(dataset.urllib.request, "urlopen", urlopen)
    paths = dataset.download_dataset(tmp_path, url="https://example.test/archive", expected_size=len(archive), expected_sha256=hashlib.sha256(archive).hexdigest())
    assert paths == [target]
    assert target.read_bytes() == archive
