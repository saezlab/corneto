import pytest
import pathlib


def test_add_reaction_to_empty_renet_inplace():
    from corneto._core import DenseReNet

    rn = DenseReNet.empty()
    rn.add_reaction("r1", {"A": -1, "B": 1}, value=1.0, inplace=True)
    assert rn.reactions == ["r1"]
    assert set(rn.species) == {"A", "B"}
    assert rn.stoichiometry[rn.get_species_id("A"), :] == -1
    assert rn.stoichiometry[rn.get_species_id("B"), :] == 1


def test_add_reaction_to_empty_renet():
    from corneto._core import DenseReNet

    rn = DenseReNet.empty()
    rn2 = rn.add_reaction("r1", {"A": -1, "B": 1}, value=1.0, inplace=False)
    assert rn2.reactions == ["r1"]
    assert set(rn2.species) == {"A", "B"}
    assert rn2.stoichiometry[rn2.get_species_id("A"), :] == -1
    assert rn2.stoichiometry[rn2.get_species_id("B"), :] == 1
    assert rn != rn2
    assert len(rn.reactions) == 0
