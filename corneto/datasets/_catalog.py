"""Built-in dataset metadata for the CORNETO dataset loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable metadata required to locate and validate one dataset revision."""

    name: str
    version: str
    remote_url: str | None
    archive_sha256: str
    file_sha256: Mapping[str, str]
    provenance: str = ""
    license: str = ""

    @property
    def relative_path(self) -> Path:
        return Path(self.name) / self.version


_SACHS_SPEC = DatasetSpec(
    name="sachs",
    version="v1",
    # This path becomes available when the dataset revision is published. The
    # checksum prevents a mutable hosting path from silently changing data.
    remote_url="https://raw.githubusercontent.com/saezlab/corneto/main/datasets/sachs/v1/sachs-v1.tar.gz",
    archive_sha256="78c23eb79223a9b03cea9f4ce4e832151be3c99c54b31e2ee3ea7ec983bdaca3",
    file_sha256={
        "measurements.tsv": "a40650c0fe6aa6581dca556202cd4e94fb0dfa1e6d121c710123bbc164509bcd",
        "condition_manifest.csv": "9f3083b5a2b20b8b2d4cbfab1f00153765c400fd76c94c5392c81b004ebf6fcb",
        "README.md": "979d829cbc105873c38b00d99733d0563c190bbb44f873b4697f13d8acccd777",
    },
    provenance="https://doi.org/10.5281/zenodo.7681811",
    license="CC BY 4.0",
)

DATASETS: dict[tuple[str, str], DatasetSpec] = {
    (_SACHS_SPEC.name, _SACHS_SPEC.version): _SACHS_SPEC,
}


def _validate_component(value: str, label: str) -> None:
    if not isinstance(value, str) or not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{label} must be a single non-empty path component.")


def get_dataset_spec(name: str, version: str) -> DatasetSpec:
    """Return the registered specification for a dataset revision."""
    _validate_component(name, "name")
    _validate_component(version, "version")
    try:
        return DATASETS[(name, version)]
    except KeyError as error:
        known = ", ".join(f"{dataset}:{revision}" for dataset, revision in sorted(DATASETS))
        raise ValueError(f"Unknown dataset revision {name!r}, {version!r}; known revisions: {known}.") from error
