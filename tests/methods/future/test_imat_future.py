import os

import numpy as np
import pytest

from corneto._data import Data
from corneto.io import import_miom_model
from corneto.methods.future.imat import MultiSampleIMAT


@pytest.fixture
def metabolic_network():
    """Load the mitocore metabolic network."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "mitocore.miom")
    return import_miom_model(file_path)


def test_single_sample_imat_low_expression_ko(metabolic_network, backend):
    """Test iMAT with low expression data"""
    # Genes for MDHm (Malate Dehydrogenase) in MitoCore are 4141 and 4142
    # We'll treat them as lowly expressed.
    imat = MultiSampleIMAT(
        backend=backend,
        low_expression_threshold=-0.5,
        high_expression_threshold=0.5,
    )

    # Create data where MDHm-related genes are lowly expressed
    data = Data.from_cdict(
        {
            "sample1": {
                "EX_biomass_e": {"role": "objective"},
                "4141": {"value": -1.0, "mapping": "none"},
                "4142": {"value": -1.0, "mapping": "none"},
            }
        }
    )

    problem = imat.build(metabolic_network, data)
    problem.solve()

    # Check that the flux through MDHm is zero because its genes are lowly expressed
    rid = next(iter(metabolic_network.get_edges_by_attr("id", "MDHm")))
    mdhm_flux = problem.expr.flow[rid].value
    assert np.isclose(mdhm_flux, 0.0, atol=1e-4)

    # For comparison, the biomass flux should be reduced, similar to a hard knockout
    bid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_biomass_e")))
    biomass_flux = problem.expr.flow[bid].value
    assert biomass_flux < 10 and biomass_flux > 1


def test_single_sample_imat_high_expression(metabolic_network, backend):
    """Test iMAT with high expression data to encourage flux through a reaction."""
    # Genes for ACONTm (Aconitase) are 59, 60, 61. We'll mark them as highly expressed.
    imat = MultiSampleIMAT(
        backend=backend,
        low_expression_threshold=-0.5,
        high_expression_threshold=0.5,
        beta_reg=0.01,  # Add a small regularization to get a unique solution
    )

    # Create data where ACONTm-related genes are highly expressed
    data = Data.from_cdict(
        {
            "sample1": {
                "EX_biomass_e": {"role": "objective"},
                "59": {"value": 1.0, "mapping": "none"},
                "60": {"value": 1.0, "mapping": "none"},
                "61": {"value": 1.0, "mapping": "none"},
            }
        }
    )

    problem = imat.build(metabolic_network, data)
    problem.solve()

    # Check that the flux through ACONTm is active
    rid = next(iter(metabolic_network.get_edges_by_attr("id", "ACONTm")))
    acontm_flux = problem.expr.flow[rid].value
    assert abs(acontm_flux) > 1.0

    # Biomass should be optimal
    bid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_biomass_e")))
    biomass_flux = problem.expr.flow[bid].value
    assert np.isclose(biomass_flux, 100.89, atol=1e-2)
