r"""I/O (:mod:`corneto.io.metabolism`)
====================================

.. currentmodule:: corneto.io.metabolism

This module provides the implementations of the various methods used in CORNETO.
It is organized into several functional areas.

"""

import os
import zipfile
from typing import BinaryIO, Dict, List, Set, Tuple, Union

import numpy as np

from corneto import suppress_output
from corneto._types import CobraModel
from corneto.graph import Graph

from ._base import _as_local_path, _open_binary
from ._sbml import MetabolicModel, read_sbml


def graph_from_vertex_incidence(
    A: np.ndarray,
    vertex_ids: Union[List[str], np.ndarray],
    edge_ids: Union[List[str], np.ndarray],
):
    """Create graph from vertex incidence matrix and labels.

    Args:
        A: Vertex incidence matrix. Rows are vertices, columns are edges.
            Non-zero entries indicate edge-vertex connections.
        vertex_ids: Labels for vertices corresponding to matrix rows
        edge_ids: Labels for edges corresponding to matrix columns

    Returns:
        Graph instance constructed from incidence matrix

    Raises:
        ValueError: If dimensions of inputs don't match
    """
    g = Graph()
    if len(vertex_ids) != A.shape[0]:
        raise ValueError(
            """The number of rows in A matrix is different from
            the number of vertex ids"""
        )
    if len(edge_ids) != A.shape[1]:
        raise ValueError(
            """The number of columns in A matrix is different from
            the number of edge ids"""
        )
    for v in vertex_ids:
        g.add_vertex(v)
    for j, v in enumerate(edge_ids):
        values = A[:, j]
        idx = np.flatnonzero(values)
        coeffs = values[idx]
        v_names = [vertex_ids[i] for i in idx]
        s = {n: val for n, val in zip(v_names, coeffs) if val < 0}
        t = {n: val for n, val in zip(v_names, coeffs) if val > 0}
        g.add_edge(s, t, id=v)
    return g


def import_cobra_model(
    path: str | os.PathLike | BinaryIO,
    quiet: bool = True,
    *,
    compression: str | None = "auto",
    timeout: float = 30.0,
) -> Graph:
    """Import a COBRA model from an SBML file and convert it to a CORNETO graph.

    Args:
        path: SBML path, HTTP(S) URL, or binary readable stream
        quiet: Suppress output from the COBRApy reader.
        compression: ``"auto"`` detects and decompresses gzip, bz2, or xz by
            magic bytes; ``None`` disables decompression, and an explicit codec
            forces it. ZIP archives are passed to COBRApy's SBML reader, which
            reads only the first member of a multi-file archive.
        timeout: HTTP request timeout in seconds.

    Paths, URLs, and streams are materialized in a temporary file for COBRApy;
    caller-owned sources are not modified or closed.

    Returns:
        Graph: A CORNETO graph representing the metabolic network
    """
    try:
        from cobra.io import read_sbml_model
    except ImportError as e:
        raise ImportError("COBRApy not installed.", e)
    with _as_local_path(path, compression=compression, timeout=timeout, suffix=".xml") as local_path:
        reader_path = local_path
        if zipfile.is_zipfile(local_path):
            # COBRApy selects its SBML reader partly from the filename suffix.
            # Only rename our temporary materialization; the caller's source is untouched.
            reader_path = local_path.with_suffix(".zip")
            local_path.rename(reader_path)
        if quiet:
            with suppress_output(suppress_stdout=True):
                model = read_sbml_model(str(reader_path))
        else:
            model = read_sbml_model(str(reader_path))
    return cobra_model_to_graph(model)


def sbml_model_to_graph(model: MetabolicModel) -> Graph:
    """Convert parsed SBML records to a CORNETO metabolic hypergraph.

    Raw SBML species and reaction identifiers are preserved. Boundary and
    constant species are kept in the graph's ``sbml_excluded_species``
    metadata and excluded from mass-balance vertices. The active FBC objective,
    when present, is stored as ``fba_objective`` in the graph metadata. Pass
    that mapping explicitly as ``objectives`` when building
    :class:`corneto.methods.fba.MultiSampleFBA`.

    Args:
        model: Parsed records returned by :func:`read_sbml`.

    Returns:
        A graph with one hyperedge per SBML reaction.
    """
    active_objective = model.objectives.get(model.active_objective)
    fba_objective = None
    if active_objective is not None:
        sign = -1.0 if active_objective["sense"] == "maximize" else 1.0
        fba_objective = {
            reaction_id: sign * coefficient for reaction_id, coefficient in active_objective["coefficients"].items()
        }

    excluded_species = {
        identifier: species
        for identifier, species in model.species.items()
        if species["boundary_condition"] or species["constant"]
    }
    graph = Graph(
        sbml_model=model.model_info,
        sbml_compartments=model.compartments,
        sbml_unit_definitions=model.unit_definitions,
        sbml_excluded_species=excluded_species,
        sbml_parameters=model.parameters,
        sbml_gene_products=model.gene_products,
        sbml_objectives=model.objectives,
        sbml_groups=model.groups,
        sbml_active_objective=model.active_objective,
        fba_objective=fba_objective,
    )

    for species_id, species in model.species.items():
        if species_id in excluded_species:
            continue
        vertex = {
            "name": species["name"],
            "compartment": species["compartment"],
            "boundaryCondition": species["boundary_condition"],
            "constant": species["constant"],
            "hasOnlySubstanceUnits": species["has_only_substance_units"],
            "initialAmount": species["initial_amount"],
            "initialConcentration": species["initial_concentration"],
            "substanceUnits": species["substance_units"],
            "formula": species["formula"],
            "charge": species["charge"],
            "meta_id": species["meta_id"],
            "sbo_term": species["sbo_term"],
        }
        if "annotations" in species:
            vertex["annotations"] = species["annotations"]
        graph.add_vertex(species_id, **vertex)

    for reaction_id, reaction in model.reactions.items():
        net: dict[str, float] = {}
        for species_id, coefficient in reaction["reactants"].items():
            net[species_id] = net.get(species_id, 0.0) - coefficient
        for species_id, coefficient in reaction["products"].items():
            net[species_id] = net.get(species_id, 0.0) + coefficient
        net = {species_id: coefficient for species_id, coefficient in net.items() if coefficient}

        sources: dict[str, float] = {}
        targets: dict[str, float] = {}
        for species_id, coefficient in net.items():
            species = model.species[species_id]
            if species["boundary_condition"] or species["constant"]:
                continue
            (sources if coefficient < 0 else targets)[species_id] = abs(coefficient)

        edge = {
            "id": reaction_id,
            "name": reaction["name"],
            "default_lb": reaction["lower_bound"],
            "default_ub": reaction["upper_bound"],
            "GPR": reaction["gpr"],
            "reversible": reaction["reversible"],
            "fast": reaction["fast"],
            "modifiers": reaction["modifiers"],
            "stoichiometry": net,
            "reactants": reaction["reactants"],
            "products": reaction["products"],
            "meta_id": reaction["meta_id"],
            "sbo_term": reaction["sbo_term"],
        }
        if "annotations" in reaction:
            edge["annotations"] = reaction["annotations"]
        graph.add_edge(sources, targets, **edge)

    return graph


def import_sbml_model(
    source: str | os.PathLike | BinaryIO,
    *,
    annotations: bool = False,
    compression: str | None = "auto",
    timeout: float = 30.0,
) -> Graph:
    """Read supported SBML and convert it to a CORNETO metabolic graph.

    The native reader supports SBML Level 3 Version 1 Core and FBC Version 2,
    with optional Groups Version 1. It uses only the standard library to parse
    SBML and does not require COBRApy. Raw SBML identifiers are preserved;
    boundary and constant species remain available in graph metadata but are
    omitted from mass-balance vertices. If the document has an active FBC
    objective, its minimization-convention coefficient map is stored as
    ``graph.get_graph_attributes()["fba_objective"]``. Pass that mapping
    explicitly to ``MultiSampleFBA.build(..., objectives=...)`` to use it.

    Args:
        source: SBML path, HTTP(S) URL, or binary readable stream. Caller-owned
            streams are read from their current position and remain open.
        annotations: Include embedded RDF resource URIs in record metadata.
            Annotation URIs are not fetched.
        compression: ``"auto"`` detects gzip, bz2, or xz by magic bytes;
            ``None`` disables decompression, and an explicit codec forces it.
        timeout: HTTP request timeout in seconds.

    Returns:
        A CORNETO graph containing the parsed metabolic network.

    Raises:
        ValueError: If the document uses unsupported dynamic semantics or
            contains invalid or unresolved model references.
    """
    return sbml_model_to_graph(read_sbml(source, annotations=annotations, compression=compression, timeout=timeout))


def _load_compressed_gem(
    source,
    *,
    compression: str | None = "auto",
    timeout: float = 30.0,
):
    """Load one MIOM archive while its source and archive remain open."""
    with _open_binary(source, compression=compression, timeout=timeout, seekable=True) as stream:
        archive = np.load(stream, allow_pickle=True)
        try:
            S = archive["S"]
            R = archive["reactions"]
            M = archive["metabolites"]
        finally:
            close = getattr(archive, "close", None)
            if close is not None:
                close()
    return S, R, M


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
        rxn[rxn_id] = [*rxn.get(rxn_id, []), (s, d, t)]
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


def import_miom_model(
    model_or_path: Union[str, os.PathLike, BinaryIO, np.ndarray],
    *,
    compression: str | None = "auto",
    timeout: float = 30.0,
) -> Graph:
    """Create graph from MIOM metabolic model.

    Args:
        model_or_path: MIOM model, path, HTTP(S) URL, or binary readable stream
        compression: ``"auto"`` detects gzip, bz2, or xz by magic bytes;
            ``None`` disables decompression, and an explicit codec forces it.
        timeout: HTTP request timeout in seconds.

    Returns:
        New Graph representing the metabolic network
    """
    if isinstance(model_or_path, (str, os.PathLike)) or callable(getattr(model_or_path, "read", None)):
        S, R, M = _load_compressed_gem(model_or_path, compression=compression, timeout=timeout)
    else:
        S, R, M = model_or_path.S, model_or_path.R, model_or_path.M
    G = graph_from_vertex_incidence(S, M["id"], R["id"])
    # Add metadata to the graph, such as default lb/ub for reactions
    for i in range(G.num_edges):
        attr = G.get_attr_edge(i)
        attr["default_lb"] = R["lb"][i]
        attr["default_ub"] = R["ub"][i]
        attr["GPR"] = R["gpr"][i]
    return G
