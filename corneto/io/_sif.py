"""Readers and graph conversion for Simple Interaction Format (SIF)."""

import os
from typing import BinaryIO, Iterable, Optional, Sequence, TextIO, Tuple

from corneto._types import TupleSIF
from corneto.graph import Graph

from ._base import _open_text


def _read_sif(
    sif_file: str | os.PathLike | BinaryIO | TextIO,
    delimiter: str = "\t",
    has_header: bool = False,
    discard_self_loops: Optional[bool] = True,
    column_order: Sequence[int] = (0, 1, 2),
    *,
    compression: str | None = "auto",
    timeout: float = 30.0,
    encoding: str = "utf-8",
) -> list[TupleSIF]:
    import csv

    reactions = set()
    with _open_text(sif_file, compression=compression, timeout=timeout, encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, line in enumerate(reader):
            if has_header and i == 0:
                continue
            if len(line) != 3:
                raise ValueError(f"Invalid SIF line: {line}: expected 3 columns")
            s, d, t = [line[idx] for idx in column_order]
            if discard_self_loops and s == t:
                continue
            reactions |= set([(s, int(d), t)])
    return list(reactions)


def _read_sif_iter(
    sif_file: str | os.PathLike | BinaryIO | TextIO,
    delimiter: str = "\t",
    has_header: bool = False,
    discard_self_loops: Optional[bool] = True,
    column_order: Sequence[int] = (0, 1, 2),
    *,
    compression: str | None = "auto",
    timeout: float = 30.0,
    encoding: str = "utf-8",
) -> Iterable[TupleSIF]:
    """Yield source, interaction, and target tuples from a SIF source."""
    import csv

    with _open_text(sif_file, compression=compression, timeout=timeout, encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, line in enumerate(reader):
            if has_header and i == 0:
                continue
            if len(line) <= max(column_order):
                raise ValueError(f"Invalid SIF line: {line}: expected at least 3 columns")
            source, interaction, target = [line[idx] for idx in column_order]
            if discard_self_loops and source == target:
                continue
            yield source, int(interaction), target


def load_graph_from_sif(
    sif_file: str | os.PathLike | BinaryIO | TextIO,
    delimiter: str = "\t",
    has_header: bool = False,
    discard_self_loops: Optional[bool] = True,
    column_order: Sequence[int] = (0, 1, 2),
    *,
    compression: str | None = "auto",
    timeout: float = 30.0,
    encoding: str = "utf-8",
):
    """Create a graph from a local, remote, or file-like SIF source.

    Args:
        sif_file: SIF path, HTTP(S) URL, binary stream, or text stream
        delimiter: Column delimiter in file
        has_header: Whether file has a header row
        discard_self_loops: Whether to ignore self-loops
        column_order: Order of source, interaction, target columns
        compression: ``"auto"`` detects gzip, bz2, or xz by magic bytes;
            ``None`` disables decompression, and an explicit codec forces it.
        timeout: HTTP request timeout in seconds.
        encoding: Encoding used for binary sources.

    Returns:
        New Graph loaded from SIF file
    """
    it = _read_sif_iter(
        sif_file,
        delimiter=delimiter,
        has_header=has_header,
        discard_self_loops=discard_self_loops,
        column_order=column_order,
        compression=compression,
        timeout=timeout,
        encoding=encoding,
    )
    return load_graph_from_sif_tuples(it)


def load_graph_from_sif_tuples(tuples: Iterable[Tuple]):
    """Create graph from iterable of SIF tuples.

    Args:
        tuples: Iterable of (source, interaction, target) tuples

    Returns:
        New Graph created from SIF data
    """
    g = Graph()
    for s, v, t in tuples:
        g.add_edge(s, t, interaction=v)
    return g
