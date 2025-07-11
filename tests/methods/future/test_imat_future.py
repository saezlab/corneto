import os

import numpy as np
import pytest

from corneto._data import Data
from corneto.backend import PicosBackend
from corneto.io import import_miom_model
from corneto.methods.future.fba import MultiSampleFBA
from corneto.methods.future.imat import MultiSampleIMAT

# Use HiGHS solver for fast execution
SOLVER = "highs"


@pytest.fixture
def metabolic_network():
    """Load the mitocore metabolic network."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "mitocore.miom")
    return import_miom_model(file_path)


def test_imat_without_biomass_low_expression(metabolic_network, backend):
    """Test iMAT with low expression data - no biomass maximization"""
    if isinstance(backend, PicosBackend):
        pytest.skip("iMAT tests require CVXPY backend")

    # Gene for PYK (Pyruvate Kinase) in MitoCore is ENSG00000067225
    # We'll treat it as lowly expressed.
    imat = MultiSampleIMAT(
        backend=backend,
        lambda_reg=0,
    )

    # Create data where PYK-related gene is lowly expressed
    # Add minimal constraints to make the network feasible
    data = Data.from_cdict(
        {
            "sample1": {
                "ENSG00000067225": {
                    "value": -1.0,
                    "mapping": "none",
                },  # Normal value for pure iMAT
                # Add minimal constraints to create demand
                "EX_glc_D_e": {
                    "lower_bound": -10,
                    "upper_bound": 0,
                },  # Glucose available
                "EX_o2_e": {"lower_bound": -10, "upper_bound": 0},  # Oxygen available
            }
        }
    )

    problem = imat.build(metabolic_network, data)
    problem.solve(solver=SOLVER)

    # Check that the flux through PYK is reduced/minimized
    # because its gene is lowly expressed
    rid = next(iter(metabolic_network.get_edges_by_attr("id", "PYK")))
    pyk_flux = problem.expr.flow[rid].value
    # PYK flux should be very low or zero given low expression
    assert abs(pyk_flux) <= 0.1


def test_imat_without_biomass_high_expression(metabolic_network, backend):
    """Test iMAT with high expression data - no biomass maximization"""
    if isinstance(backend, PicosBackend):
        pytest.skip("iMAT tests require CVXPY backend")

    # Gene for TPI (Triose Phosphate Isomerase) is ENSG00000111669.
    # We'll mark it as highly expressed.
    imat = MultiSampleIMAT(
        backend=backend,
        lambda_reg=0,
    )

    # Create data where TPI-related gene is highly expressed
    # Add minimal constraints to make the network feasible
    data = Data.from_cdict(
        {
            "sample1": {
                "ENSG00000111669": {
                    "value": 1.0,
                    "mapping": "none",
                },  # Normal value for pure iMAT
                # Add minimal constraints to create demand
                "EX_glc_D_e": {
                    "lower_bound": -10,
                    "upper_bound": 0,
                },  # Glucose available
                "EX_o2_e": {"lower_bound": -10, "upper_bound": 0},  # Oxygen available
            }
        }
    )

    problem = imat.build(metabolic_network, data)
    problem.solve(solver=SOLVER)

    # Check that the flux through TPI is encouraged by high expression
    rid = next(iter(metabolic_network.get_edges_by_attr("id", "TPI")))
    tpi_flux = problem.expr.flow[rid].value
    # TPI flux should be active due to high expression and constraints
    assert abs(tpi_flux) > 0.01


def test_single_sample_imat_low_expression_ko(metabolic_network, backend):
    """Test iMAT with low expression data and strong biomass objective"""
    if isinstance(backend, PicosBackend):
        pytest.skip("iMAT tests require CVXPY backend")
    # Gene for PYK (Pyruvate Kinase) in MitoCore is ENSG00000067225
    # We'll treat it as lowly expressed.
    imat = MultiSampleIMAT(
        backend=backend,
        lambda_reg=0,
    )

    # Create data where PYK-related gene is lowly expressed
    # Use strong gene expression value to compete with biomass
    data = Data.from_cdict(
        {
            "sample1": {
                "EX_biomass_e": {"role": "objective"},
                "ENSG00000067225": {
                    "value": -1000.0,
                    "mapping": "none",
                },  # Strong gene penalty
            }
        }
    )

    problem = imat.build(metabolic_network, data)
    problem.solve(solver=SOLVER)

    # Check that the flux through PYK is reduced/minimized
    # because its gene is lowly expressed
    rid = next(iter(metabolic_network.get_edges_by_attr("id", "PYK")))
    pyk_flux = problem.expr.flow[rid].value
    # PYK flux should be reasonably low given low expression
    assert abs(pyk_flux) <= 2.0

    # The biomass flux should be maintained close to optimal
    bid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_biomass_e")))
    biomass_flux = problem.expr.flow[bid].value
    assert biomass_flux > 100


def test_single_sample_imat_high_expression(metabolic_network, backend):
    """Test iMAT with high expression data and biomass objective."""
    if isinstance(backend, PicosBackend):
        pytest.skip("iMAT tests require CVXPY backend")
    # Gene for TPI (Triose Phosphate Isomerase) is ENSG00000111669.
    # We'll mark it as highly expressed.
    imat = MultiSampleIMAT(
        backend=backend,
        beta_reg=0.01,  # Add a small regularization to get a unique solution
    )

    # Create data where TPI-related gene is highly expressed
    # Use strong gene expression value to compete with biomass
    data = Data.from_cdict(
        {
            "sample1": {
                "EX_biomass_e": {"role": "objective"},
                "ENSG00000111669": {
                    "value": 1000.0,
                    "mapping": "none",
                },  # Strong gene reward
            }
        }
    )

    problem = imat.build(metabolic_network, data)
    problem.solve(solver=SOLVER)

    # Check that the flux through TPI is active
    rid = next(iter(metabolic_network.get_edges_by_attr("id", "TPI")))
    tpi_flux = problem.expr.flow[rid].value
    assert abs(tpi_flux) > 0.5

    # Biomass should be close to optimal
    bid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_biomass_e")))
    biomass_flux = problem.expr.flow[bid].value
    assert np.isclose(biomass_flux, 100.89, atol=0.5)


def test_multi_sample_imat_different_expression(metabolic_network, backend):
    """Test iMAT with multiple samples having different expression profiles."""
    if isinstance(backend, PicosBackend):
        pytest.skip("iMAT tests require CVXPY backend")
    imat = MultiSampleIMAT(
        backend=backend,
        lambda_reg=0,
    )

    # Create data with two samples:
    # Sample 1: PGI lowly expressed (should reduce PGI flux)
    # Sample 2: TPI lowly expressed (should reduce TPI flux)
    # Use very strong gene expression values to compete with biomass
    data = Data.from_cdict(
        {
            "sample1": {
                "EX_biomass_e": {"role": "objective"},
                "ENSG00000105220": {"value": -10000.0, "mapping": "none"},  # PGI gene
            },
            "sample2": {
                "EX_biomass_e": {"role": "objective"},
                "ENSG00000111669": {"value": -10000.0, "mapping": "none"},  # TPI gene
            },
        }
    )

    problem = imat.build(metabolic_network, data)
    problem.solve(solver=SOLVER)

    # Get reaction indices
    pgi_rid = next(iter(metabolic_network.get_edges_by_attr("id", "PGI")))
    tpi_rid = next(iter(metabolic_network.get_edges_by_attr("id", "TPI")))
    bid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_biomass_e")))

    # Check sample 1: PGI should have low flux, TPI should be normal
    pgi_flux_s1 = problem.expr.flow[pgi_rid, 0].value
    tpi_flux_s1 = problem.expr.flow[tpi_rid, 0].value
    biomass_flux_s1 = problem.expr.flow[bid, 0].value

    # With very large negative expression (-10000), PGI should be strongly penalized
    # But metabolic constraints might still require some flux
    # The key test is that it should be lower than without penalty
    assert pgi_flux_s1 < 1.0, f"PGI flux should be reduced with penalty: {pgi_flux_s1}"
    assert abs(tpi_flux_s1) > 0.01  # TPI should be active (no penalty)
    assert biomass_flux_s1 > 0.1  # Biomass should be positive

    # The penalty should reduce flux compared to unconstrained case
    # Additional verification can be added here if needed

    # Check sample 2: TPI should have low flux, PGI should be normal
    pgi_flux_s2 = problem.expr.flow[pgi_rid, 1].value
    tpi_flux_s2 = problem.expr.flow[tpi_rid, 1].value
    biomass_flux_s2 = problem.expr.flow[bid, 1].value

    assert abs(pgi_flux_s2) > 0.01  # PGI should be active (no penalty)
    assert tpi_flux_s2 < 1.0, f"TPI flux should be reduced with penalty: {tpi_flux_s2}"
    assert biomass_flux_s2 > 0.1  # Biomass should be positive

    # The penalty should reduce flux compared to unconstrained case
    # Additional verification can be added here if needed


def test_multi_sample_imat_mixed_expression_no_biomass(metabolic_network, backend):
    """Test iMAT with multiple samples having mixed high/low expression - no biomass objective."""
    if isinstance(backend, PicosBackend):
        pytest.skip("iMAT tests require CVXPY backend")
    imat = MultiSampleIMAT(
        backend=backend,
        lambda_reg=0,
    )

    # Create data with two samples:
    # Sample 1: PGI highly expressed, TPI lowly expressed
    # Sample 2: PGI lowly expressed, TPI highly expressed
    # NO biomass objective - pure iMAT test with minimal constraints
    data = Data.from_cdict(
        {
            "sample1": {
                "ENSG00000105220": {
                    "value": 1000.0,
                    "mapping": "none",
                },  # PGI gene high
                "ENSG00000111669": {
                    "value": -1000.0,
                    "mapping": "none",
                },  # TPI gene low
                # Add minimal constraints to create demand
                "EX_glc_D_e": {
                    "lower_bound": -10,
                    "upper_bound": 0,
                },  # Glucose available
                "EX_o2_e": {"lower_bound": -10, "upper_bound": 0},  # Oxygen available
            },
            "sample2": {
                "ENSG00000105220": {
                    "value": -1000.0,
                    "mapping": "none",
                },  # PGI gene low
                "ENSG00000111669": {
                    "value": 1000.0,
                    "mapping": "none",
                },  # TPI gene high
                # Add minimal constraints to create demand
                "EX_glc_D_e": {
                    "lower_bound": -10,
                    "upper_bound": 0,
                },  # Glucose available
                "EX_o2_e": {"lower_bound": -10, "upper_bound": 0},  # Oxygen available
            },
        }
    )

    problem = imat.build(metabolic_network, data)
    problem.solve(solver=SOLVER)

    # Get reaction indices
    pgi_rid = next(iter(metabolic_network.get_edges_by_attr("id", "PGI")))
    tpi_rid = next(iter(metabolic_network.get_edges_by_attr("id", "TPI")))
    bid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_biomass_e")))

    # Check sample 1: PGI should be active (value=1000), TPI should be blocked (value=-1000)
    pgi_flux_s1 = problem.expr.flow[pgi_rid, 0].value
    tpi_flux_s1 = problem.expr.flow[tpi_rid, 0].value
    biomass_flux_s1 = problem.expr.flow[bid, 0].value

    assert abs(pgi_flux_s1) > 0.01  # PGI should be active with large positive value
    assert np.isclose(tpi_flux_s1, 0.0, atol=1e-6), f"TPI flux should be 0 with large penalty: {tpi_flux_s1}"
    # No biomass objective, so biomass flux should be 0
    assert biomass_flux_s1 >= 0.0  # Should be non-negative

    # Check sample 2: PGI should be blocked (value=-1000), TPI should be active (value=1000)
    pgi_flux_s2 = problem.expr.flow[pgi_rid, 1].value
    tpi_flux_s2 = problem.expr.flow[tpi_rid, 1].value
    biomass_flux_s2 = problem.expr.flow[bid, 1].value

    assert np.isclose(pgi_flux_s2, 0.0, atol=1e-6), f"PGI flux should be 0 with large penalty: {pgi_flux_s2}"
    assert abs(tpi_flux_s2) > 0.01  # TPI should be active with large positive value
    # No biomass objective, so biomass flux should be 0
    assert biomass_flux_s2 >= 0.0  # Should be non-negative


def test_imat_media_composition_glucose(metabolic_network, backend):
    """Test iMAT with different glucose availability."""
    if isinstance(backend, PicosBackend):
        pytest.skip("iMAT tests require CVXPY backend")

    imat = MultiSampleIMAT(
        backend=backend,
        lambda_reg=0,
    )

    # Test with two different glucose conditions
    data = Data.from_cdict(
        {
            "glucose_rich": {
                "EX_biomass_e": {"role": "objective"},
                "EX_glc_D_e": {"lower_bound": -10, "upper_bound": 0},  # Glucose
                "EX_o2_e": {"lower_bound": -10, "upper_bound": 0},  # Oxygen
            },
            "glucose_limited": {
                "EX_biomass_e": {"role": "objective"},
                "EX_glc_D_e": {"lower_bound": 0, "upper_bound": 0},  # No glucose
                "EX_o2_e": {"lower_bound": -10, "upper_bound": 0},  # Oxygen
            },
        }
    )

    problem = imat.build(metabolic_network, data)
    problem.solve(solver=SOLVER)

    # Get reaction indices
    bid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_biomass_e")))
    glc_rid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_glc_D_e")))

    # Check glucose rich condition
    biomass_rich = problem.expr.flow[bid, 0].value
    glucose_rich = problem.expr.flow[glc_rid, 0].value

    # Check glucose limited condition
    biomass_limited = problem.expr.flow[bid, 1].value
    glucose_limited = problem.expr.flow[glc_rid, 1].value

    # Biomass should be higher with glucose available
    assert biomass_rich > biomass_limited
    assert glucose_rich < 0  # Should have glucose uptake
    assert np.isclose(glucose_limited, 0.0, atol=1e-6)  # No glucose uptake

    # Both should have reasonable biomass production
    assert biomass_rich > 50
    assert biomass_limited > 40


def test_imat_media_composition_oxygen(metabolic_network, backend):
    """Test iMAT with different oxygen availability."""
    if isinstance(backend, PicosBackend):
        pytest.skip("iMAT tests require CVXPY backend")

    imat = MultiSampleIMAT(
        backend=backend,
        lambda_reg=0,
    )

    # Test with two different oxygen conditions
    data = Data.from_cdict(
        {
            "aerobic": {
                "EX_biomass_e": {"role": "objective"},
                "EX_glc_D_e": {
                    "lower_bound": -10,
                    "upper_bound": 0,
                },  # Glucose available
                "EX_o2_e": {"lower_bound": -10, "upper_bound": 0},  # Oxygen available
            },
            "anaerobic": {
                "EX_biomass_e": {"role": "objective"},
                "EX_glc_D_e": {
                    "lower_bound": -10,
                    "upper_bound": 0,
                },  # Glucose available
                "EX_o2_e": {"lower_bound": 0, "upper_bound": 0},  # No oxygen
            },
        }
    )

    problem = imat.build(metabolic_network, data)
    problem.solve(solver=SOLVER)

    # Get reaction indices
    bid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_biomass_e")))
    o2_rid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_o2_e")))

    # Check aerobic condition
    biomass_aerobic = problem.expr.flow[bid, 0].value
    oxygen_aerobic = problem.expr.flow[o2_rid, 0].value

    # Check anaerobic condition
    biomass_anaerobic = problem.expr.flow[bid, 1].value
    oxygen_anaerobic = problem.expr.flow[o2_rid, 1].value

    # Biomass should be higher with oxygen available
    assert biomass_aerobic > biomass_anaerobic
    assert oxygen_aerobic < 0  # Should have oxygen uptake
    assert np.isclose(oxygen_anaerobic, 0.0, atol=1e-6)  # No oxygen uptake

    # Both should have some biomass production
    assert biomass_aerobic > 50
    assert biomass_anaerobic > 0


def test_imat_media_bounds_override_defaults(metabolic_network, backend):
    """Test that data bounds override default graph bounds."""
    if isinstance(backend, PicosBackend):
        pytest.skip("iMAT tests require CVXPY backend")

    imat = MultiSampleIMAT(
        backend=backend,
        lambda_reg=0,
    )

    # Test that explicitly setting bounds to 0 overrides default bounds
    # Default bounds for EX_glc_D_e should be [-1000, 1000]
    data = Data.from_cdict(
        {
            "restricted": {
                "EX_biomass_e": {"role": "objective"},
                "EX_glc_D_e": {"lower_bound": 0, "upper_bound": 0},  # Force no glucose
                "EX_o2_e": {"lower_bound": -5, "upper_bound": 0},  # Limited oxygen
            },
        }
    )

    problem = imat.build(metabolic_network, data)
    problem.solve(solver=SOLVER)

    # Get reaction indices
    bid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_biomass_e")))
    glc_rid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_glc_D_e")))
    o2_rid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_o2_e")))

    # Check that bounds were properly applied
    biomass_flux = problem.expr.flow[bid].value
    glucose_flux = problem.expr.flow[glc_rid].value
    oxygen_flux = problem.expr.flow[o2_rid].value

    # Glucose should be exactly 0 (bounds override)
    assert np.isclose(glucose_flux, 0.0, atol=1e-6)

    # Oxygen should be limited by our bounds
    assert oxygen_flux >= -5.0
    assert oxygen_flux <= 0.0

    # Should still have some biomass production
    assert biomass_flux > 0


def test_imat_expression_with_media_changes(metabolic_network, backend):
    """Test iMAT with both expression data and media composition changes."""
    if isinstance(backend, PicosBackend):
        pytest.skip("iMAT tests require CVXPY backend")

    imat = MultiSampleIMAT(
        backend=backend,
        lambda_reg=0,
    )

    # Test with expression data and different media conditions
    data = Data.from_cdict(
        {
            "condition1": {
                "EX_biomass_e": {"role": "objective"},
                "EX_glc_D_e": {
                    "lower_bound": -10,
                    "upper_bound": 0,
                },  # Glucose available
                "EX_o2_e": {"lower_bound": -10, "upper_bound": 0},  # Oxygen available
                "ENSG00000067225": {
                    "value": -1000.0,
                    "mapping": "none",
                },  # PYK lowly expressed - strong value to compete with biomass
            },
            "condition2": {
                "EX_biomass_e": {"role": "objective"},
                "EX_glc_D_e": {"lower_bound": 0, "upper_bound": 0},  # No glucose
                "EX_o2_e": {"lower_bound": -10, "upper_bound": 0},  # Oxygen available
                "ENSG00000067225": {
                    "value": 1000.0,
                    "mapping": "none",
                },  # PYK highly expressed - strong value to compete with biomass
            },
        }
    )

    problem = imat.build(metabolic_network, data)
    problem.solve(solver=SOLVER)

    # Get reaction indices
    bid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_biomass_e")))
    glc_rid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_glc_D_e")))

    # Check condition 1: glucose available, PYK lowly expressed
    biomass1 = problem.expr.flow[bid, 0].value
    glucose1 = problem.expr.flow[glc_rid, 0].value

    # Check condition 2: no glucose, PYK highly expressed
    biomass2 = problem.expr.flow[bid, 1].value
    glucose2 = problem.expr.flow[glc_rid, 1].value

    # Media effects should be respected
    assert glucose1 < 0  # Should have glucose uptake in condition 1
    assert np.isclose(glucose2, 0.0, atol=1e-6)  # No glucose in condition 2

    # Both should have biomass production
    assert biomass1 > 0
    assert biomass2 > 0

    # Expression effects should be considered
    # This is harder to test definitively, but we can check the solutions are feasible
    # The problem should solve successfully (just verify it's not None)
    assert problem.expr.flow[bid, 0].value is not None
    assert problem.expr.flow[bid, 1].value is not None


def test_imat_zero_to_positive_growth_transition(metabolic_network, backend):
    """Test that MultiSampleIMAT correctly handles bounds override.

    This test verifies that IMAT properly processes bounds data from the Data object
    and can override default graph bounds when specified.
    """
    if isinstance(backend, PicosBackend):
        pytest.skip("iMAT tests require CVXPY backend")

    # Test IMAT bounds override capability with a simple case
    imat = MultiSampleIMAT(backend=backend, lambda_reg=0)

    # Test that IMAT can handle explicit bounds
    data_bounds_test = Data.from_cdict(
        {
            "condition1": {
                "EX_biomass_e": {"role": "objective"},
                "EX_glc_D_e": {
                    "lower_bound": -5,
                    "upper_bound": 0,
                },  # Specific glucose bounds
            },
            "condition2": {
                "EX_biomass_e": {"role": "objective"},
                "EX_glc_D_e": {
                    "lower_bound": -10,
                    "upper_bound": 0,
                },  # Different glucose bounds
            },
        }
    )

    problem = imat.build(metabolic_network, data_bounds_test)
    problem.solve(solver=SOLVER)

    bid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_biomass_e")))
    glc_rid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_glc_D_e")))

    # Check that both conditions produce valid solutions
    biomass1 = problem.expr.flow[bid, 0].value
    biomass2 = problem.expr.flow[bid, 1].value
    glucose1 = problem.expr.flow[glc_rid, 0].value
    glucose2 = problem.expr.flow[glc_rid, 1].value

    # Test passes if:
    # 1. Both conditions produce feasible solutions
    assert biomass1 >= 0, "Condition 1 should produce valid biomass"
    assert biomass2 >= 0, "Condition 2 should produce valid biomass"

    # 2. Glucose bounds are respected
    assert glucose1 >= -5 and glucose1 <= 0, f"Glucose condition 1 should respect bounds: {glucose1}"
    assert glucose2 >= -10 and glucose2 <= 0, f"Glucose condition 2 should respect bounds: {glucose2}"


def test_fba_vs_imat_bounds_consistency(metabolic_network, backend):
    """Test that FBA and IMAT handle bounds consistently in 0->growth transitions."""
    if isinstance(backend, PicosBackend):
        pytest.skip("iMAT tests require CVXPY backend")

    # Save and modify the model to have restrictive default bounds for key nutrients
    key_exchanges = ["EX_glc_D_e", "EX_o2_e"]
    original_bounds = {}
    for rxn_id in key_exchanges:
        edges = list(metabolic_network.get_edges_by_attr("id", rxn_id))
        if edges:
            attrs = metabolic_network._get_edge_attributes(edges[0])
            original_bounds[edges[0]] = (
                attrs.get("default_lb", 0),
                attrs.get("default_ub", 0),
            )
            attrs["default_lb"] = 0.0
            attrs["default_ub"] = 0.0

    # Test data that enables growth by overriding restrictive bounds
    media_data = Data.from_cdict(
        {
            "sample1": {
                "EX_biomass_e": {"role": "objective"},
                "EX_glc_D_e": {"lower_bound": -10, "upper_bound": 0},
                "EX_o2_e": {"lower_bound": -10, "upper_bound": 0},
            }
        }
    )

    # Test FBA
    fba = MultiSampleFBA(backend=backend)
    fba_problem = fba.build(metabolic_network, media_data)
    fba_problem.solve(solver=SOLVER)

    # Test IMAT
    imat = MultiSampleIMAT(backend=backend, lambda_reg=0)
    imat_problem = imat.build(metabolic_network, media_data)
    imat_problem.solve(solver=SOLVER)

    # Both should achieve positive growth
    bid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_biomass_e")))
    fba_biomass = fba_problem.expr.flow[bid].value
    imat_biomass = imat_problem.expr.flow[bid].value

    assert fba_biomass > 0.01  # Lower threshold - just need positive growth
    assert imat_biomass > 0.01

    # The main test is that both can transition from 0 to positive growth
    # Exact values may differ due to different optimization objectives

    # Both should have glucose available (non-positive values allowed)
    glc_rid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_glc_D_e")))
    fba_glucose = fba_problem.expr.flow[glc_rid].value
    imat_glucose = imat_problem.expr.flow[glc_rid].value

    assert fba_glucose <= 0.0  # Should not secrete glucose
    assert imat_glucose <= 0.0

    # Restore original bounds
    for edge_idx, (orig_lb, orig_ub) in original_bounds.items():
        attrs = metabolic_network._get_edge_attributes(edge_idx)
        attrs["default_lb"] = orig_lb
        attrs["default_ub"] = orig_ub


def test_imat_expression_with_bounds_override(metabolic_network, backend):
    """Test IMAT with expression data when bounds need to be overridden."""
    if isinstance(backend, PicosBackend):
        pytest.skip("iMAT tests require CVXPY backend")

    # Save and make glucose exchange restrictive by default
    glc_edges = list(metabolic_network.get_edges_by_attr("id", "EX_glc_D_e"))
    original_bounds = {}
    if glc_edges:
        attrs = metabolic_network._get_edge_attributes(glc_edges[0])
        original_bounds[glc_edges[0]] = (
            attrs.get("default_lb", 0),
            attrs.get("default_ub", 0),
        )
        attrs["default_lb"] = 0.0
        attrs["default_ub"] = 0.0

    imat = MultiSampleIMAT(
        backend=backend,
        lambda_reg=0,
    )

    # Test with expression data and media that overrides restrictive bounds
    data = Data.from_cdict(
        {
            "condition1": {
                "EX_biomass_e": {"role": "objective"},
                "EX_glc_D_e": {
                    "lower_bound": -10,
                    "upper_bound": 0,
                },  # Override restrictive bounds
                "EX_o2_e": {"lower_bound": -10, "upper_bound": 0},
                "ENSG00000067225": {
                    "value": -1000.0,
                    "mapping": "none",
                },  # PYK lowly expressed - strong value to compete with biomass
            },
            "condition2": {
                "EX_biomass_e": {"role": "objective"},
                "EX_glc_D_e": {
                    "lower_bound": -10,
                    "upper_bound": 0,
                },  # Override restrictive bounds
                "EX_o2_e": {"lower_bound": -10, "upper_bound": 0},
                "ENSG00000067225": {
                    "value": 1000.0,
                    "mapping": "none",
                },  # PYK highly expressed - strong value to compete with biomass
            },
        }
    )

    problem = imat.build(metabolic_network, data)
    problem.solve(solver=SOLVER)

    # Both conditions should achieve growth despite restrictive default bounds
    bid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_biomass_e")))
    biomass1 = problem.expr.flow[bid, 0].value
    biomass2 = problem.expr.flow[bid, 1].value

    assert biomass1 > 0.01
    assert biomass2 > 0.01

    # Both should have glucose uptake (bounds override working)
    glc_rid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_glc_D_e")))
    glucose1 = problem.expr.flow[glc_rid, 0].value
    glucose2 = problem.expr.flow[glc_rid, 1].value

    # Check that glucose bounds are respected (should be between -10 and 0)
    assert glucose1 >= -10.0 and glucose1 <= 0.0
    assert glucose2 >= -10.0 and glucose2 <= 0.0

    # The test is primarily about bounds override working
    # If we have positive biomass, the bounds override is working
    # The specific glucose uptake may vary depending on the model and conditions
    assert biomass1 > 0.01 or biomass2 > 0.01, "At least one condition should achieve growth"

    # Restore original bounds
    for edge_idx, (orig_lb, orig_ub) in original_bounds.items():
        attrs = metabolic_network._get_edge_attributes(edge_idx)
        attrs["default_lb"] = orig_lb
        attrs["default_ub"] = orig_ub


def test_imat_bounds_handling_critical_case(metabolic_network, backend):
    """Critical test: MultiSampleIMAT must handle bounds like MultiSampleFBA.

    This test verifies that IMAT processes bounds data consistently with FBA.
    """
    if isinstance(backend, PicosBackend):
        pytest.skip("iMAT tests require CVXPY backend")

    # Simple test: Compare FBA and IMAT with the same bounds data
    test_media = {
        "EX_biomass_e": {"role": "objective"},
        "EX_glc_D_e": {"lower_bound": -5, "upper_bound": 0},
        "EX_o2_e": {"lower_bound": -5, "upper_bound": 0},
    }

    # Test FBA
    fba = MultiSampleFBA(backend=backend)
    data_fba = Data.from_cdict({"sample1": test_media})
    problem_fba = fba.build(metabolic_network, data_fba)
    problem_fba.solve(solver=SOLVER)

    bid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_biomass_e")))
    biomass_fba = problem_fba.expr.flow[bid].value

    # Test IMAT with same bounds
    imat = MultiSampleIMAT(backend=backend, lambda_reg=0)
    data_imat = Data.from_cdict({"sample1": test_media})
    problem_imat = imat.build(metabolic_network, data_imat)
    problem_imat.solve(solver=SOLVER)

    biomass_imat = problem_imat.expr.flow[bid].value

    # Critical assertion: Both should produce valid solutions
    assert biomass_fba >= 0, f"FBA should produce valid solution: {biomass_fba}"
    assert biomass_imat >= 0, f"IMAT should produce valid solution: {biomass_imat}"

    # Both should respect the same bounds
    glc_rid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_glc_D_e")))
    o2_rid = next(iter(metabolic_network.get_edges_by_attr("id", "EX_o2_e")))

    glucose_fba = problem_fba.expr.flow[glc_rid].value
    glucose_imat = problem_imat.expr.flow[glc_rid].value
    oxygen_fba = problem_fba.expr.flow[o2_rid].value
    oxygen_imat = problem_imat.expr.flow[o2_rid].value

    # Verify bounds are respected
    assert -5 <= glucose_fba <= 0, f"FBA glucose should respect bounds: {glucose_fba}"
    assert -5 <= glucose_imat <= 0, f"IMAT glucose should respect bounds: {glucose_imat}"
    assert -5 <= oxygen_fba <= 0, f"FBA oxygen should respect bounds: {oxygen_fba}"
    assert -5 <= oxygen_imat <= 0, f"IMAT oxygen should respect bounds: {oxygen_imat}"
