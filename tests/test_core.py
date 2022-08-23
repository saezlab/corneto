import pytest
import pathlib


def test_add_reaction_to_empty_renet_inplace():
    from corneto.core import DenseReNet

    rn = DenseReNet.empty()
    rn.add_reaction("r1", {"A": -1, "B": 1}, value=1.0, inplace=True)
    assert rn.reactions == ["r1"]
    assert set(rn.species) == {"A", "B"}
    assert rn.stoichiometry[rn.get_species_id("A"), :] == -1
    assert rn.stoichiometry[rn.get_species_id("B"), :] == 1
