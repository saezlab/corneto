import os

import numpy as np
import pytest

from corneto.backend import PicosBackend
from corneto.data import Data
from corneto.io import import_miom_model
from corneto.methods.fba import MultiSampleFBA


@pytest.fixture
def metabolic_network():
    """Load a large test graph from the pickle file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "mitocore.miom")
    return import_miom_model(file_path)


def test_fba_without_data(metabolic_network, backend):
    """Test a standard FBA problem with no objective."""
    fba = MultiSampleFBA(backend=backend)
    P = fba.build(metabolic_network)
    result = P.solve()
    assert result.status == "optimal"


def test_single_sample_standard_fba(metabolic_network, backend):
    """Test the standard FBA method with a single sample."""
    fba = MultiSampleFBA(backend=backend)
    data = Data.from_cdict(
        {
            "sample1": {
                "EX_biomass_e": {
                    "role": "objective",
                },
            }
        }
    )
    P = fba.build(metabolic_network, data)
    P.solve()
    rid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_biomass_e")))
    assert np.isclose(P.expr.flow[rid].value, 100.8924, atol=1e-4)


def test_single_sample_objective_name_uses_reaction_ids(metabolic_network, backend):
    """Objective names should expose user-facing reaction IDs, not internal edge indices."""
    fba = MultiSampleFBA(backend=backend)
    data = Data.from_cdict(
        {
            "sample1": {
                "EX_biomass_e": {
                    "role": "objective",
                },
            }
        }
    )

    P = fba.build(metabolic_network, data)
    objective_names = {obj.name for obj in P.objectives if obj.name}

    assert "objective_sample1__EX_biomass_e" in objective_names


def test_two_samples_standard_fba_with_ko(metabolic_network, backend):
    """Test the standard FBA method with two samples."""
    fba = MultiSampleFBA(backend=backend)
    data = Data.from_cdict(
        {
            "sample1": {
                "EX_biomass_e": {
                    "role": "objective",
                },
            },
            "sample2": {
                "EX_biomass_e": {
                    "role": "objective",
                },
                "MDHm": {
                    "lower_bound": 0,
                    "upper_bound": 0,
                },
            },
        }
    )
    P = fba.build(metabolic_network, data)
    P.solve()
    rid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_biomass_e")))
    assert np.isclose(P.expr.flow[rid, 0].value, 100.89, atol=1e-2)
    assert np.isclose(P.expr.flow[rid, 1].value, 27.88, atol=1e-2)


def test_ko_bounds_enforced_in_indicator_variables(metabolic_network, backend):
    """Blocked reactions (lb=ub=0) must force indicator variables to zero."""
    fba = MultiSampleFBA(backend=backend)
    data = Data.from_cdict(
        {
            "sample1": {
                "EX_biomass_e": {
                    "role": "objective",
                },
            },
            "sample2": {
                "EX_biomass_e": {
                    "role": "objective",
                },
                "MDHm": {
                    "lower_bound": 0,
                    "upper_bound": 0,
                },
            },
        }
    )

    P = fba.build(metabolic_network, data)
    mdhm_rid = next(iter(metabolic_network.get_edges_by_attr("id", "MDHm")))

    # Try to force the KO reaction to be "active". This must remain impossible.
    P.add_objective(-P.expr.edge_has_flux[mdhm_rid, 1], name="maximize_mdhm_indicator")
    P.solve()

    assert np.isclose(P.expr.flow[mdhm_rid, 1].value, 0.0, atol=1e-9)
    assert np.isclose(P.expr.edge_has_flux[mdhm_rid, 1].value, 0.0, atol=1e-9)


def test_single_sample_sparse_fba(metabolic_network, backend):
    """Test the sparse FBA method with a single sample."""
    if isinstance(backend, PicosBackend):
        pytest.skip("Sparse FBA slow with GLPK and PicosBackend")

    fba = MultiSampleFBA(backend=backend, beta_reg=1)
    data = Data.from_cdict(
        {
            "sample1": {
                "EX_biomass_e": {
                    "role": "objective",
                    "lower_bound": 100.80,
                },
            }
        }
    )
    P = fba.build(metabolic_network, data)
    P.solve()
    rid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_biomass_e")))
    n_rxns = np.sum(np.round(P.expr.edge_has_flux.value))
    assert P.expr.flow[rid].value >= 100.79
    assert n_rxns <= 135
    assert n_rxns >= 124
