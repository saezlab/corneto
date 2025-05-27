"""Tests for metabolism-related I/O functionality.

This module contains tests for loading and processing metabolic models,
specifically testing the functionality in corneto.io._metabolism module.
"""

from pathlib import Path

import pytest

from corneto.io import import_cobra_model


@pytest.fixture
def compressed_model_path():
    """Provide path to compressed test metabolic model file.

    Returns:
        Path: Path object pointing to a compressed metabolic model file
            used for testing. The file is expected to be in .xz format.
    """
    return Path(__file__).parent.joinpath("data", "mitocore_v1.01.xz")


@pytest.fixture
def xml_model_path():
    """Provide path to XML test metabolic model file.

    Returns:
        Path: Path object pointing to a metabolic model file
            used for testing in XML format.
    """
    return Path(__file__).parent.joinpath("data", "mitocore_v1.01.xml")


def test_load_compressed_gem(compressed_model_path):
    """Test loading of compressed genome-scale metabolic model.

    Tests if the _load_compressed_gem function correctly loads and parses
    a compressed metabolic model file, returning matrices of expected dimensions.

    Args:
        compressed_model_path (Path): Pytest fixture providing path to test model file.

    Checks:
        - Stoichiometric matrix (S) has correct dimensions (441, 555)
        - Reaction vector (R) has correct length (555)
        - Metabolite vector (M) has correct length (441)
    """
    from corneto.io._metabolism import _load_compressed_gem

    S, R, M = _load_compressed_gem(compressed_model_path)
    assert S.shape == (441, 555)
    assert R.shape == (555,)
    assert M.shape == (441,)


def test_cobra_model_to_graph(xml_model_path):
    """Test conversion from COBRA model to CORNETO graph via import_cobra_model.

    Tests if the import_cobra_model function correctly loads an SBML file and converts it
    into a CORNETO graph, preserving the correct number of nodes and edges.

    Args:
        xml_model_path (Path): Pytest fixture providing path to test model file.

    Checks:
        - Graph has correct number of nodes (441 metabolites)
        - Graph has correct number of edges (555 reactions)
        - Graph edges have correct metadata (default_lb, default_ub, GPR)
    """
    G = import_cobra_model(str(xml_model_path))

    # Check dimensions
    assert G.num_vertices == 441
    assert G.num_edges == 555

    # Check edge attributes are present
    edge_attr = G.get_attr_edge(0)  # Check first edge
    assert "default_lb" in edge_attr
    assert "default_ub" in edge_attr
    assert "GPR" in edge_attr
