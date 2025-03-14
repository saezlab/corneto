import pickle
from typing import Optional

from corneto.graph import Graph


def load_corneto_graph(filename: str) -> Graph:
    """Load graph from a saved file.

    Supports various compression formats:
    - .gz (gzip)
    - .bz2 (bzip2)
    - .xz (LZMA)
    - .zip (zip archive, reads first file)

    Args:
        filename: Path to saved graph file

    Returns:
        Loaded graph instance
    """
    if filename.endswith(".gz"):
        import gzip

        opener = gzip.open
    elif filename.endswith(".bz2"):
        import bz2

        opener = bz2.open
    elif filename.endswith(".xz"):
        import lzma

        opener = lzma.open
    elif filename.endswith(".zip"):
        import zipfile

        def opener(file, mode="r"):
            # Supports only reading the first file in a zip archive
            with zipfile.ZipFile(file, "r") as z:
                return z.open(z.namelist()[0], mode=mode)
    else:
        opener = open

    with opener(filename, "rb") as f:
        return pickle.load(f)


def save_corneto_graph(self, filename: str, compressed: Optional[bool] = True) -> None:
    """Save graph to file.

    Args:
        filename: Path to save graph to
        compressed: Whether to use compression. If True, uses LZMA compression.

    Raises:
        ValueError: If filename is empty

    Note:
        If compressed=True, '.xz' extension is added if not present
        If '.pkl' extension is missing, it will be added
    """
    if not filename:
        raise ValueError("Filename must not be empty.")

    if not filename.endswith(".pkl"):
        filename += ".pkl"

    if compressed:
        import lzma

        if not filename.endswith(".xz"):
            filename += ".xz"
        with lzma.open(filename, "wb", preset=9) as f:
            pickle.dump(self, f)
    else:
        with open(filename, "wb") as f:
            pickle.dump(self, f)
