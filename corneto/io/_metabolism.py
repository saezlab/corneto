r"""I/O (:mod:`corneto.io.metabolism`)
====================================

.. currentmodule:: corneto.io.metabolism

This module provides the implementations of the various methods used in CORNETO.
It is organized into several functional areas.

"""

from typing import Dict, List, Set, Tuple, Union

import numpy as np

from corneto import suppress_output
from corneto._types import CobraModel
from corneto.graph import Graph

from ._base import graph_from_vertex_incidence
from ._util import _download, _is_url


def import_cobra_model(path: str, quiet: bool = True) -> Graph:
    """Import a COBRA model from an SBML file and convert it to a CORNETO graph.

    Args:
        path: Path to SBML file

    Returns:
        Graph: A CORNETO graph representing the metabolic network
    """
    try:
        from cobra.io import read_sbml_model
    except ImportError as e:
        raise ImportError("COBRApy not installed.", e)
    if quiet:
        with suppress_output(suppress_stdout=True):
            model = read_sbml_model(str(path))
    else:
        model = read_sbml_model(str(path))
    return cobra_model_to_graph(model)


def _load_compressed_gem(url_or_filepath):
    # https://github.com/MetExplore/miom
    import pathlib

    file = url_or_filepath
    if _is_url(url_or_filepath):
        file = _download(url_or_filepath)
    ext = pathlib.Path(file).suffix
    if ext == ".xz" or ext == ".miom":
        import lzma
        from io import BytesIO

        with lzma.open(file, "rb") as f_in:
            M = np.load(BytesIO(f_in.read()), allow_pickle=True)
    else:
        M = np.load(file)
    return M["S"], M["reactions"], M["metabolites"]


def _get_reaction_species(reactions: Dict[str, Dict[str, int]]) -> Set[str]:
    species: Set[str] = set()
    for v in reactions.values():
        species.update(v.keys())
    return species


def _stoichiometry(
    reactions: Dict[str, Dict[str, int]],
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


def parse_cobra_model(model: CobraModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a COBRA metabolic model and extract its core components.

    This function takes a COBRA model and extracts its stoichiometric matrix,
    reaction data, and metabolite data.

    Args:
        model (:class:`CobraModel`): A COBRA model object to import.

    Returns:
        Tuple[:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`]:
        A tuple containing:

        - **S** (:class:`numpy.ndarray`): The stoichiometric matrix.
        - **R** (:class:`numpy.ndarray`): A structured numpy array containing reaction data with fields:

          - **id** (:class:`str`): Reaction identifier.
          - **name** (:class:`str`): Reaction name.
          - **lb** (:class:`float`): Lower bound.
          - **ub** (:class:`float`): Upper bound.
          - **subsystem** (:class:`str` | :class:`List[str]`): Reaction subsystem(s).
          - **gpr** (:class:`str`): Gene-protein-reaction rule.

        - **M** (:class:`numpy.ndarray`): A structured numpy array containing metabolite data with fields:

          - **id** (:class:`str`): Metabolite identifier.
          - **name** (:class:`str`): Metabolite name.
          - **formula** (:class:`str`): Chemical formula.

    Raises:
        ImportError: If COBRApy is not installed.

    """
    # From MIOM: https://github.com/MetExplore/miom/blob/main/miom/mio.py
    try:
        from cobra.util.array import create_stoichiometric_matrix  # type: ignore
    except ImportError as e:
        raise ImportError("COBRApy not installed.", e)
    S = create_stoichiometric_matrix(model)
    subsystems = []
    for rxn in model.reactions:
        subsys = rxn.subsystem
        list_subsystem_rxn = []
        # For .mat models, the subsystem can be loaded as a
        # string repr of a numpy array
        if isinstance(subsys, str) and (subsys.startswith("array(") or subsys.startswith("[array(")):
            try:
                subsys = eval(subsys.strip())
            except Exception:
                # Try to create a list
                import re

                subsys = re.findall(r"\['(.*?)'\]", subsys)
                if len(subsys) == 0:
                    subsys = rxn.subsystem
            # A list containing a numpy array?
            for s in subsys:
                if "tolist" in dir(s):
                    list_subsystem_rxn.extend(s.tolist())
                else:
                    list_subsystem_rxn.append(s)
            if len(list_subsystem_rxn) == 1:
                list_subsystem_rxn = list_subsystem_rxn[0]
            subsystems.append(list_subsystem_rxn)

        elif "tolist" in dir(rxn.subsystem):
            subsystems.append(rxn.subsystem.tolist())
        else:
            subsystems.append(rxn.subsystem)

    rxn_data = [
        (
            rxn.id,
            rxn.name,
            rxn.lower_bound,
            rxn.upper_bound,
            subsystem,
            rxn.gene_reaction_rule,
        )
        for rxn, subsystem in zip(model.reactions, subsystems)
    ]
    met_data = [(met.id, met.name, met.formula) for met in model.metabolites]
    R = np.array(
        rxn_data,
        dtype=[
            ("id", "object"),
            ("name", "object"),
            ("lb", "float"),
            ("ub", "float"),
            ("subsystem", "object"),
            ("gpr", "object"),
        ],
    )
    M = np.array(
        met_data,
        dtype=[("id", "object"), ("name", "object"), ("formula", "object")],
    )
    return S, R, M


def cobra_model_to_graph(model: CobraModel) -> Graph:
    """Create graph from COBRA metabolic model.

    Args:
        model: COBRA model instance

    Returns:
        New Graph representing the metabolic network
    """
    S, R, M = parse_cobra_model(model)
    G = graph_from_vertex_incidence(S, M["id"], R["id"])
    # Add metadata to the graph, such as default lb/ub for reactions
    for i in range(G.num_edges):
        attr = G.get_attr_edge(i)
        attr["default_lb"] = R["lb"][i]
        attr["default_ub"] = R["ub"][i]
        attr["GPR"] = R["gpr"][i]
    return G


def import_miom_model(model_or_path: Union[str, np.ndarray]) -> Graph:
    """Create graph from MIOM metabolic model.

    Args:
        model_or_path: MIOM model instance or path to compressed model file

    Returns:
        New Graph representing the metabolic network
    """
    if isinstance(model_or_path, str):
        S, R, M = _load_compressed_gem(model_or_path)
    else:
        S = model_or_path.S, M = model_or_path.M, R = model_or_path.R
    G = graph_from_vertex_incidence(S, M["id"], R["id"])
    # Add metadata to the graph, such as default lb/ub for reactions
    for i in range(G.num_edges):
        attr = G.get_attr_edge(i)
        attr["default_lb"] = R["lb"][i]
        attr["default_ub"] = R["ub"][i]
        attr["GPR"] = R["gpr"][i]
    return G
