"""Tests for metabolism-related I/O functionality.

This module contains tests for loading and processing metabolic models,
specifically testing the functionality in corneto.io._metabolism module.
"""

import bz2
import gzip
import io
import lzma
import zipfile
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


@pytest.mark.parametrize(
    ("encoding", "suffix"),
    [("xml", ".xml"), ("gzip", ".gz"), ("bz2", ".bz2"), ("xz", ".xz"), ("zip", ".zip")],
)
@pytest.mark.parametrize("as_stream", [False, True], ids=["path", "stream"])
def test_import_cobra_model_accepts_sbml_encodings(tmp_path, xml_model_path, encoding, suffix, as_stream):
    xml_bytes = xml_model_path.read_bytes()
    if encoding == "gzip":
        payload = gzip.compress(xml_bytes)
    elif encoding == "bz2":
        payload = bz2.compress(xml_bytes)
    elif encoding == "xz":
        payload = lzma.compress(xml_bytes)
    elif encoding == "zip":
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            archive.writestr("model.xml", xml_bytes)
        payload = archive_buffer.getvalue()
    else:
        payload = xml_bytes

    path = tmp_path / f"model{suffix}"
    path.write_bytes(payload)
    source = io.BytesIO(payload) if as_stream else path

    graph = import_cobra_model(source)

    assert graph.num_vertices == 441
    assert graph.num_edges == 555
    if as_stream:
        assert not source.closed
        assert source.getvalue() == payload
    else:
        assert path.read_bytes() == payload
