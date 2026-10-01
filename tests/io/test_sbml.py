"""Tests for the native SBML reader and graph conversion."""

import gzip
import io
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from corneto.backend import CvxpyBackend
from corneto.io import import_sbml_model, read_sbml
from corneto.methods.fba import MultiSampleFBA

CORE = "http://www.sbml.org/sbml/level3/version1/core"
FBC = "http://www.sbml.org/sbml/level3/version1/fbc/version2"

SBML = f'''<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="{CORE}" xmlns:fbc="{FBC}" level="3" version="1" fbc:required="true">
  <model id="raw_model_id" fbc:strict="true">
    <listOfCompartments>
      <compartment id="c" constant="true" />
      <compartment id="e" constant="true" />
    </listOfCompartments>
    <listOfSpecies>
      <species id="glc__D_e" compartment="e" boundaryCondition="true" constant="false" hasOnlySubstanceUnits="false" />
      <species id="glc__D_c" compartment="c" boundaryCondition="false" constant="false" hasOnlySubstanceUnits="false" />
    </listOfSpecies>
    <listOfParameters>
      <parameter id="lb_zero" value="0" constant="true" />
      <parameter id="ub_import" value="5" constant="true" />
      <parameter id="ub_biomass" value="1000" constant="true" />
    </listOfParameters>
    <fbc:listOfGeneProducts>
      <fbc:geneProduct fbc:id="geneA" fbc:label="gene-A" />
      <fbc:geneProduct fbc:id="geneB" fbc:label="gene-B" />
    </fbc:listOfGeneProducts>
    <listOfReactions>
      <reaction id="EX_glc__D_e" reversible="false" fbc:lowerFluxBound="lb_zero" fbc:upperFluxBound="ub_import">
        <listOfReactants>
          <speciesReference species="glc__D_e" stoichiometry="1" constant="true" />
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="glc__D_c" stoichiometry="2" constant="true" />
        </listOfProducts>
        <fbc:geneProductAssociation>
          <fbc:and>
            <fbc:geneProductRef fbc:geneProduct="geneA" />
            <fbc:geneProductRef fbc:geneProduct="geneB" />
          </fbc:and>
        </fbc:geneProductAssociation>
      </reaction>
      <reaction id="BIOMASS" reversible="false" fbc:lowerFluxBound="lb_zero" fbc:upperFluxBound="ub_biomass">
        <listOfReactants>
          <speciesReference species="glc__D_c" stoichiometry="3" constant="true" />
        </listOfReactants>
      </reaction>
    </listOfReactions>
    <fbc:listOfObjectives fbc:activeObjective="growth">
      <fbc:objective fbc:id="growth" fbc:type="maximize">
        <fbc:listOfFluxObjectives>
          <fbc:fluxObjective fbc:reaction="BIOMASS" fbc:coefficient="2" />
        </fbc:listOfFluxObjectives>
      </fbc:objective>
    </fbc:listOfObjectives>
  </model>
</sbml>
'''


def test_native_reader_and_converter_preserve_sbml_semantics():
    parsed = read_sbml(io.BytesIO(SBML.encode()))

    assert parsed.model_info["id"] == "raw_model_id"
    assert parsed.reactions["EX_glc__D_e"]["reactants"] == {"glc__D_e": 1.0}
    assert parsed.reactions["EX_glc__D_e"]["products"] == {"glc__D_c": 2.0}
    assert parsed.reactions["EX_glc__D_e"]["lower_bound"] == 0
    assert parsed.reactions["EX_glc__D_e"]["upper_bound"] == 5
    assert parsed.reactions["EX_glc__D_e"]["gpr"] == "(geneA and geneB)"
    assert parsed.objectives["growth"]["sense"] == "maximize"
    assert parsed.objectives["growth"]["coefficients"] == {"BIOMASS": 2.0}

    graph = import_sbml_model(io.BytesIO(SBML.encode()))
    assert graph.num_vertices == 1
    assert graph.vertices == ("glc__D_c",)
    assert graph.get_graph_attributes()["sbml_excluded_species"].keys() == {"glc__D_e"}
    assert graph.get_graph_attributes()["sbml_excluded_species"]["glc__D_e"]["boundary_condition"]
    assert graph.get_graph_attributes()["fba_objective"] == {"BIOMASS": -2.0}

    import_edge = next(iter(graph.get_edges_by_attr("id", "EX_glc__D_e")))
    edge = graph.get_attr_edge(import_edge)
    assert edge["stoichiometry"] == {"glc__D_e": -1.0, "glc__D_c": 2.0}
    assert edge["default_lb"] == 0
    assert edge["default_ub"] == 5
    assert edge["GPR"] == "(geneA and geneB)"


def test_native_sbml_objective_can_be_passed_explicitly_to_fba():
    graph = import_sbml_model(io.BytesIO(SBML.encode()))
    objective = graph.get_graph_attributes()["fba_objective"]
    backend = CvxpyBackend()
    problem = MultiSampleFBA(backend=backend).build(graph, objectives=objective)

    result = problem.solve(solver="SCIPY")
    biomass_edge = next(iter(graph.get_edges_by_attr("id", "BIOMASS")))
    assert result.status == "optimal"
    assert np.isclose(problem.expr.flow[biomass_edge].value, 10 / 3)


def test_minimization_objective_sign_is_retained():
    minimized = SBML.replace('fbc:type="maximize"', 'fbc:type="minimize"')
    graph = import_sbml_model(io.BytesIO(minimized.encode()))

    assert graph.get_graph_attributes()["fba_objective"] == {"BIOMASS": 2.0}


def test_annotations_are_preserved_when_requested():
    species = (
        '<species id="glc__D_c" compartment="c" boundaryCondition="false" constant="false" '
        'hasOnlySubstanceUnits="false" />'
    )
    annotated_species = (
        '<species id="glc__D_c" compartment="c" boundaryCondition="false" constant="false" '
        'hasOnlySubstanceUnits="false">'
        '<annotation xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:RDF><rdf:Description><rdf:li rdf:resource="https://identifiers.org/chebi/CHEBI:1234" />'
        "</rdf:Description></rdf:RDF></annotation></species>"
    )
    annotated = SBML.replace(species, annotated_species)

    parsed = read_sbml(io.BytesIO(annotated.encode()), annotations=True)
    graph = import_sbml_model(io.BytesIO(annotated.encode()), annotations=True)

    assert parsed.species["glc__D_c"]["annotations"] == ("https://identifiers.org/chebi/CHEBI:1234",)
    assert graph.get_attr_vertex("glc__D_c")["annotations"] == ("https://identifiers.org/chebi/CHEBI:1234",)


def test_paths_gzip_and_caller_owned_streams(tmp_path):
    plain_path = tmp_path / "model.xml"
    plain_path.write_text(SBML)
    with plain_path.open("rb") as stream:
        read_sbml(stream)
        assert not stream.closed

    gzip_path = tmp_path / "model.xml.gz"
    with gzip.open(gzip_path, "wb") as stream:
        stream.write(SBML.encode())
    assert read_sbml(gzip_path).model_info["id"] == "raw_model_id"
    assert import_sbml_model(plain_path).num_edges == 2


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (
            SBML.replace('species="glc__D_c" stoichiometry="2"', 'species="missing" stoichiometry="2"'),
            "unknown species",
        ),
        (SBML.replace('fbc:geneProduct="geneB"', 'fbc:geneProduct="missing_gene"'), "unknown gene product"),
        (SBML.replace(' fbc:upperFluxBound="ub_import"', "", 1), "missing FBC upperFluxBound"),
        (
            SBML.replace('fbc:lowerFluxBound="lb_zero"', 'fbc:lowerFluxBound="missing_bound"', 1),
            "unresolved lowerFluxBound",
        ),
        (SBML.replace('id="lb_zero" value="0"', 'id="lb_zero" value="6"'), "lower bound above upper bound"),
        (
            SBML.replace("    <fbc:listOfGeneProducts>", "    <listOfEvents />\n    <fbc:listOfGeneProducts>"),
            "listOfEvents is unsupported",
        ),
    ],
)
def test_invalid_references_and_unsupported_semantics_raise(document, message):
    with pytest.raises(ValueError, match=message):
        read_sbml(io.BytesIO(document.encode()))


def test_parser_module_loads_and_runs_without_site_packages(tmp_path):
    io_path = Path(__file__).parents[2] / "corneto" / "io"
    model_path = tmp_path / "model.xml"
    model_path.write_text(SBML)
    script = textwrap.dedent(
        """
        import importlib.util
        import sys
        import types

        package = types.ModuleType("corneto")
        package.__path__ = []
        io_package = types.ModuleType("corneto.io")
        io_package.__path__ = [sys.argv[1]]
        sys.modules["corneto"] = package
        sys.modules["corneto.io"] = io_package
        for name in ("_base", "_sbml"):
            module_name = f"corneto.io.{name}"
            spec = importlib.util.spec_from_file_location(module_name, f"{sys.argv[1]}/{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        parsed = sys.modules["corneto.io._sbml"].read_sbml(sys.argv[2])
        assert parsed.active_objective == "growth"
        """
    )
    subprocess.run(
        [sys.executable, "-S", "-c", script, str(io_path), str(model_path)],
        check=True,
        capture_output=True,
        text=True,
    )
