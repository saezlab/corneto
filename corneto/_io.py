from typing import List, Iterable, Tuple, Optional, Dict, Set
from corneto._typing import StrOrInt, TupleSIF
import numpy as np


def _read_sif(
    sif_file: str,
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


def _read_sif_iter(
    sif_file: str,
    delimiter: str = "\t",
    has_header: bool = False,
    discard_self_loops: Optional[bool] = True,
    column_order: List[int] = [0, 1, 2],  # source interaction target
) -> Iterable[TupleSIF]:
    import csv

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
            yield (s, int(d), t)


def _reaction_stoichiometry(reaction: List[TupleSIF]) -> Dict[str, int]:
    if len(reaction) == 1:
        # This reaction contains coarse-grained interaction,
        # coeff not stoichiometry but type of interaction
        # TODO: Consider a better unification
        s, d, t = reaction[0]
        return {s: -1, t: 1}
    else:
        appearances: dict = {}
        for s, _, t in reaction:
            appearances[s] = appearances.get(s, 0) + 1
            appearances[t] = appearances.get(t, 0) + 1
        rid = None
        for k, v in appearances.items():
            if v == len(reaction):
                if rid is None:
                    rid = k
                else:
                    raise ValueError(f"Malformed reaction, options are: {rid}, {k}")
        if rid is None:
            raise ValueError(f"Malformed reaction")
        reactants = {s: -int(d) for s, d, t in reaction if t == rid}
        products = {t: int(d) for s, d, t in reaction if s == rid}
        return {**reactants, **products}


def _get_reaction_species(reactions: Dict[str, Dict[str, int]]) -> Set[str]:
    species: Set[str] = set()
    for v in reactions.values():
        species.update(v.keys())
    return species


def _stoichiometry(
    reactions: Dict[str, Dict[str, int]]
) -> Tuple[np.ndarray, List[str], List[str]]:
    reactions_ids = list(reactions.keys())
    compounds_ids = list(_get_reaction_species(reactions))
    S = np.zeros((len(compounds_ids), len(reactions_ids)))
    for i, r in enumerate(reactions_ids):
        for c, coeff in reactions[r].items():
            S[compounds_ids.index(c), i] = coeff
    return S, reactions_ids, compounds_ids


def _index_reactions(list_reactions: List[Tuple[str, int, str]]):
    rxn: dict = {}
    for s, d, t in list_reactions:
        # Check if any symbol is a reaction
        rxn_id = None
        if s.startswith("R:"):
            rxn_id = s
        elif t.startswith("R:"):
            rxn_id = t
        if rxn_id is None:
            rxn_id = f"{s}--({d})--{t}"
        rxn[rxn_id] = rxn.get(rxn_id, []) + [(s, d, t)]
    return rxn


def load_sif_from_tuples(tpl: List[TupleSIF]):
    indexed_reactions = _index_reactions(tpl)
    reactions = {k: _reaction_stoichiometry(v) for k, v in indexed_reactions.items()}
    reaction_values: Dict[int, float] = dict()
    S, rxn_ids, species_ids = _stoichiometry(reactions)
    for i in range(len(rxn_ids)):
        v = indexed_reactions[rxn_ids[i]]
        if len(v) == 1:
            s, d, t = v[0]
            reaction_values[i] = d
    return S, species_ids, rxn_ids, reaction_values


def load_sif(
    sif_file: str,
    delimiter: str = "\t",
    has_header: bool = False,
    discard_self_loops: Optional[bool] = True,
    column_order: List[int] = [0, 1, 2],
):
    reaction_tpls = _read_sif(
        sif_file,
        delimiter=delimiter,
        has_header=has_header,
        discard_self_loops=discard_self_loops,
        column_order=column_order,
    )
    return load_sif_from_tuples(reaction_tpls)
