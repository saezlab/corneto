import pytest
import pathlib

def test_read_sif():
    from corneto._io import _read_sif

    file = pathlib.Path(__file__).parent.joinpath("sif", "PKN-LiverDREAM.sif")
    tpl = _read_sif(file)
    if len(tpl) != 58:
        pytest.fail(
            f"Detected incorrent number of SIF tuples in file ({len(tpl)}. Expected 58)."
        )
    reactants = set(l for l, _, _ in tpl)
    if len(reactants) != 37:
        pytest.fail(
            f"Detected incorrect number of reactants in SIF file ({len(reactants)}). Expected 37."
        )
    products = set(r for _, _, r in tpl)
    if len(products) != 36:
        pytest.fail(
            f"Detected incorrect number of products in SIF file ({len(reactants)}). Expected 36."
        )
    return True

