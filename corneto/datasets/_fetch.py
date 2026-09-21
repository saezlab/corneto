"""Source-first fetching and caching for small CORNETO datasets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from ._catalog import DatasetSpec, get_dataset_spec

DEFAULT_VERSION = "v1"
_DATASETS_ENV = "CORNETO_DATASETS_DIR"
_DATA_HOME_ENV = "CORNETO_DATA_HOME"
_CACHE_MARKER = ".complete.json"
_DOWNLOAD_TIMEOUT = 30


class DatasetError(RuntimeError):
    """Base error for dataset discovery and fetching failures."""


class DatasetIntegrityError(DatasetError):
    """Raised when a dataset does not match its recorded checksums."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _default_data_home() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "corneto" / "Cache"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches" / "corneto"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "corneto"
    return root / "datasets"


def _cache_root(data_home: Path | str | None) -> Path:
    if data_home is not None:
        return Path(data_home).expanduser()
    configured = os.environ.get(_DATA_HOME_ENV)
    if configured:
        return Path(configured).expanduser()
    return _default_data_home()


def _as_dataset_root(path: Path) -> Path:
    path = path.expanduser()
    if (path / "datasets").is_dir():
        return path / "datasets"
    return path


def _source_roots(source_dir: Path | str | None) -> list[Path]:
    roots: list[Path] = []

    def add(path: Path) -> None:
        candidate = _as_dataset_root(path).resolve()
        if candidate not in roots:
            roots.append(candidate)

    if source_dir is not None:
        add(Path(source_dir))
    configured = os.environ.get(_DATASETS_ENV)
    if configured:
        add(Path(configured))

    for base in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for parent in (base, *base.parents):
            add(parent / "datasets")
    return roots


def _dataset_dir(root: Path, spec: DatasetSpec) -> Path:
    return root / spec.relative_path


def _validate_files(directory: Path, spec: DatasetSpec) -> bool:
    missing = []
    for filename in spec.file_sha256:
        path = directory / filename
        if not path.is_file() or path.is_symlink():
            missing.append(filename)
    if missing:
        return False

    for filename, expected in spec.file_sha256.items():
        actual = _sha256(directory / filename)
        if actual != expected:
            raise DatasetIntegrityError(
                f"Checksum mismatch for {spec.name} {spec.version} file {filename!r}: "
                f"expected {expected}, got {actual}."
            )
    return True


def _marker_payload(spec: DatasetSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "version": spec.version,
        "archive_sha256": spec.archive_sha256,
        "files": dict(spec.file_sha256),
    }


def _valid_cache(directory: Path, spec: DatasetSpec) -> bool:
    marker = directory / _CACHE_MARKER
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if payload != _marker_payload(spec):
        return False
    return _validate_files(directory, spec)


def _write_marker(directory: Path, spec: DatasetSpec) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        prefix=".complete-",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(descriptor.name)
    try:
        with descriptor:
            json.dump(_marker_payload(spec), descriptor, indent=2, sort_keys=True)
            descriptor.write("\n")
            descriptor.flush()
            os.fsync(descriptor.fileno())
        os.replace(temporary, directory / _CACHE_MARKER)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_archive(archive_path: Path, target: Path, spec: DatasetSpec) -> Path:
    extraction_dir = Path(tempfile.mkdtemp(prefix=f".{spec.name}-{spec.version}-", dir=target.parent))
    try:
        seen: set[str] = set()
        expected = set(spec.file_sha256)
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                if member.name not in expected or member.name in seen or not member.isfile():
                    raise DatasetIntegrityError(
                        f"Archive for {spec.name} {spec.version} contains an unexpected or unsafe member "
                        f"{member.name!r}."
                    )
                seen.add(member.name)
                source = archive.extractfile(member)
                if source is None:
                    raise DatasetIntegrityError(f"Could not read archive member {member.name!r}.")
                destination = extraction_dir / member.name
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)

        if seen != expected or not _validate_files(extraction_dir, spec):
            missing = sorted(expected - seen)
            raise DatasetIntegrityError(f"Archive for {spec.name} {spec.version} is missing files: {missing}.")

        target.mkdir(parents=True, exist_ok=True)
        (target / _CACHE_MARKER).unlink(missing_ok=True)
        for filename in expected:
            os.replace(extraction_dir / filename, target / filename)
        _write_marker(target, spec)
        return target
    finally:
        shutil.rmtree(extraction_dir, ignore_errors=True)


def _download_to_cache(target: Path, spec: DatasetSpec) -> Path:
    if not spec.remote_url:
        raise DatasetError(f"Dataset {spec.name!r} {spec.version!r} has no configured remote source.")

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{spec.name}-{spec.version}-",
        suffix=".download",
        delete=False,
    )
    archive_path = Path(descriptor.name)
    digest = hashlib.sha256()
    try:
        with descriptor:
            request = Request(spec.remote_url, headers={"User-Agent": "corneto-datasets"})
            with urlopen(request, timeout=_DOWNLOAD_TIMEOUT) as response:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    descriptor.write(chunk)
            descriptor.flush()
            os.fsync(descriptor.fileno())
        actual = digest.hexdigest()
        if actual != spec.archive_sha256:
            raise DatasetIntegrityError(
                f"Checksum mismatch for downloaded {spec.name} {spec.version}: "
                f"expected {spec.archive_sha256}, got {actual}."
            )
        return _extract_archive(archive_path, target, spec)
    finally:
        archive_path.unlink(missing_ok=True)


def fetch_dataset(
    name: str,
    version: str = DEFAULT_VERSION,
    *,
    source_dir: Path | str | None = None,
    data_home: Path | str | None = None,
    auto_download: bool = True,
) -> Path:
    """Locate or fetch a versioned CORNETO dataset.

    Local repository data takes precedence over the persistent cache and the
    configured remote source. Dataset files are validated by SHA-256 before
    they are returned.

    Args:
        name: Registered dataset name.
        version: Immutable dataset revision. Defaults to ``"v1"``.
        source_dir: Optional directory containing dataset directories, or a
            repository root containing a ``datasets`` directory.
        data_home: Optional persistent cache directory.
        auto_download: Whether to fetch a missing dataset from its remote URL.

    Returns:
        The path containing the validated dataset files.

    Raises:
        DatasetIntegrityError: If available data do not match their checksums.
        DatasetError: If no local data exists and the dataset has no remote URL.
        FileNotFoundError: If data are missing and ``auto_download`` is false.
        ValueError: If the dataset revision is not registered.
    """
    if not isinstance(auto_download, bool):
        raise TypeError("auto_download must be a boolean.")
    spec = get_dataset_spec(name, version)

    for root in _source_roots(source_dir):
        candidate = _dataset_dir(root, spec)
        if candidate.exists() and _validate_files(candidate, spec):
            return candidate

    cache_path = _dataset_dir(_cache_root(data_home), spec)
    if _valid_cache(cache_path, spec):
        return cache_path

    if not auto_download:
        raise FileNotFoundError(
            f"Dataset {name!r} {version!r} was not found locally or in the cache, and auto_download is false."
        )
    return _download_to_cache(cache_path, spec)
