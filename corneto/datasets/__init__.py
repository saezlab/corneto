"""Small, versioned datasets fetched on demand."""

from corneto.datasets._fetch import DatasetError, DatasetIntegrityError, fetch_dataset

__all__ = ["DatasetError", "DatasetIntegrityError", "fetch_dataset"]
