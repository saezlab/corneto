from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

from corneto._types import TupleSIF
from corneto.graph import Graph


def _read_sif(
    sif_file: Union[str, Path],
    delimiter: str = "\t",
    has_header: bool = False,
    discard_self_loops: Optional[bool] = True,
    column_order: List[int] = [0, 1, 2],  # source interaction target
) -> List[TupleSIF]:
    import csv

    reactions = set()
    with open(sif_file, "r") as f:
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


@staticmethod
def load_graph_from_sif(
    sif_file: str,
    delimiter: str = "\t",
    has_header: bool = False,
    discard_self_loops: Optional[bool] = True,
    column_order: List[int] = [0, 1, 2],
):
    """Create graph from Simple Interaction Format (SIF) file.

    Args:
        sif_file: Path to SIF file
        delimiter: Column delimiter in file
        has_header: Whether file has a header row
        discard_self_loops: Whether to ignore self-loops
        column_order: Order of source, interaction, target columns

    Returns:
        New Graph loaded from SIF file
    """
    from corneto._io import _read_sif_iter

    it = _read_sif_iter(
        sif_file,
        delimiter=delimiter,
        has_header=has_header,
        discard_self_loops=discard_self_loops,
        column_order=column_order,
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
