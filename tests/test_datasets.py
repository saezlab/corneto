import hashlib
import io
import tarfile
from pathlib import Path

import pytest

import corneto.datasets._catalog as catalog
import corneto.datasets._fetch as fetch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_dataset(root: Path, name: str, version: str, files: dict[str, str]) -> Path:
    directory = root / name / version
    directory.mkdir(parents=True)
    for filename, content in files.items():
        (directory / filename).write_text(content, encoding="utf-8")
    return directory


def _make_remote_spec(tmp_path: Path, monkeypatch, *, unsafe_member: str | None = None):
    files = {
        "measurements.tsv": "value\n1\n",
        "condition_manifest.csv": "condition\nexample\n",
        "README.md": "example dataset\n",
    }
    payload = tmp_path / "payload"
    payload.mkdir()
    for filename, content in files.items():
        (payload / filename).write_text(content, encoding="utf-8")

    archive = tmp_path / "dataset.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        if unsafe_member is None:
            for filename in files:
                handle.add(payload / filename, arcname=filename)
        else:
            info = tarfile.TarInfo(unsafe_member)
            content = b"unsafe\n"
            info.size = len(content)
            handle.addfile(info, fileobj=io.BytesIO(content))

    spec = catalog.DatasetSpec(
        name="example",
        version="v1",
        remote_url="https://data.example.org/example-v1.tar.gz",
        archive_sha256=_sha256(archive),
        file_sha256={
            filename: hashlib.sha256(content.encode("utf-8")).hexdigest() for filename, content in files.items()
        },
    )
    monkeypatch.setitem(catalog.DATASETS, (spec.name, spec.version), spec)

    def open_archive(request, timeout):
        assert request.full_url == spec.remote_url
        return archive.open("rb")

    monkeypatch.setattr(fetch, "urlopen", open_archive)
    return spec


def test_default_version_uses_local_repository_dataset(tmp_path):
    repository_root = Path(__file__).parents[1]

    result = fetch.fetch_dataset("sachs", data_home=tmp_path, auto_download=False)

    assert result == (repository_root / "datasets" / "sachs" / "v1").resolve()
    assert (result / "measurements.tsv").is_file()
    assert (result / "condition_manifest.csv").is_file()


def test_explicit_future_version_can_use_local_source(tmp_path, monkeypatch):
    files = {
        "measurements.tsv": "value\n2\n",
        "condition_manifest.csv": "condition\nfuture\n",
        "README.md": "future dataset\n",
    }
    directory = _write_dataset(tmp_path / "datasets", "example", "v2", files)
    spec = catalog.DatasetSpec(
        name="example",
        version="v2",
        remote_url=None,
        archive_sha256="unused",
        file_sha256={filename: hashlib.sha256(content.encode()).hexdigest() for filename, content in files.items()},
    )
    monkeypatch.setitem(catalog.DATASETS, (spec.name, spec.version), spec)

    result = fetch.fetch_dataset("example", version="v2", source_dir=tmp_path / "datasets", auto_download=False)

    assert result == directory.resolve()


def test_local_source_wins_over_remote(tmp_path, monkeypatch):
    spec = _make_remote_spec(tmp_path, monkeypatch)
    local = _write_dataset(
        tmp_path / "checkout" / "datasets",
        spec.name,
        spec.version,
        {
            "measurements.tsv": "value\nlocal\n",
            "condition_manifest.csv": "condition\nlocal\n",
            "README.md": "local dataset\n",
        },
    )
    local_spec = catalog.DatasetSpec(
        name=spec.name,
        version=spec.version,
        remote_url=spec.remote_url,
        archive_sha256=spec.archive_sha256,
        file_sha256={filename: _sha256(local / filename) for filename in spec.file_sha256},
    )
    monkeypatch.setitem(catalog.DATASETS, (spec.name, spec.version), local_spec)

    def fail_download(*args, **kwargs):
        raise AssertionError("remote source should not be used")

    monkeypatch.setattr(fetch, "urlopen", fail_download)
    result = fetch.fetch_dataset(spec.name, source_dir=tmp_path / "checkout", data_home=tmp_path / "cache")

    assert result == local.resolve()


def test_remote_download_is_cached_and_reused(tmp_path, monkeypatch):
    spec = _make_remote_spec(tmp_path, monkeypatch)
    source_dir = tmp_path / "missing-source"
    data_home = tmp_path / "cache"

    result = fetch.fetch_dataset(spec.name, source_dir=source_dir, data_home=data_home)
    assert result == (data_home / spec.name / spec.version).resolve()
    assert (result / ".complete.json").is_file()

    def fail_download(*args, **kwargs):
        raise AssertionError("cached source should be used")

    monkeypatch.setattr(fetch, "urlopen", fail_download)
    cached = fetch.fetch_dataset(spec.name, source_dir=source_dir, data_home=data_home)

    assert cached == result


def test_auto_download_false_does_not_use_remote(tmp_path, monkeypatch):
    spec = _make_remote_spec(tmp_path, monkeypatch)

    with pytest.raises(FileNotFoundError, match="auto_download is false"):
        fetch.fetch_dataset(
            spec.name,
            source_dir=tmp_path / "missing",
            data_home=tmp_path / "cache",
            auto_download=False,
        )


def test_checksum_mismatch_is_rejected(tmp_path, monkeypatch):
    spec = _make_remote_spec(tmp_path, monkeypatch)
    broken = catalog.DatasetSpec(
        name=spec.name,
        version=spec.version,
        remote_url=spec.remote_url,
        archive_sha256="0" * 64,
        file_sha256=spec.file_sha256,
    )
    monkeypatch.setitem(catalog.DATASETS, (broken.name, broken.version), broken)

    with pytest.raises(fetch.DatasetIntegrityError, match="Checksum mismatch"):
        fetch.fetch_dataset(spec.name, source_dir=tmp_path / "missing", data_home=tmp_path / "cache")


def test_unsafe_archive_member_is_rejected(tmp_path, monkeypatch):
    spec = _make_remote_spec(tmp_path, monkeypatch, unsafe_member="../measurements.tsv")

    with pytest.raises(fetch.DatasetIntegrityError, match="unsafe member"):
        fetch.fetch_dataset(spec.name, source_dir=tmp_path / "missing", data_home=tmp_path / "cache")

    assert not (tmp_path / "measurements.tsv").exists()


def test_incomplete_cache_is_not_accepted(tmp_path, monkeypatch):
    spec = _make_remote_spec(tmp_path, monkeypatch)
    incomplete = tmp_path / "cache" / spec.name / spec.version
    incomplete.mkdir(parents=True)
    (incomplete / "measurements.tsv").write_text("partial\n", encoding="utf-8")

    result = fetch.fetch_dataset(
        spec.name,
        source_dir=tmp_path / "missing",
        data_home=tmp_path / "cache",
        auto_download=True,
    )

    assert (result / ".complete.json").is_file()
    assert (result / "measurements.tsv").read_text(encoding="utf-8") == "value\n1\n"
