import pathlib


def test_read_sif():
    from corneto._io import _read_sif

    file = pathlib.Path(__file__).parent.joinpath("sif", "PKN-LiverDREAM.sif")
    tpl = _read_sif(file)

    # Check number of SIF tuples
    assert len(tpl) == 58, f"Expected 58 SIF tuples, got {len(tpl)}"

    # Check number of reactants
    reactants = set(l for l, _, _ in tpl)
    assert len(reactants) == 37, f"Expected 37 unique reactants, got {len(reactants)}"

    # Check number of products
    products = set(r for _, _, r in tpl)
    assert len(products) == 36, f"Expected 36 unique products, got {len(products)}"

    # Optional: Check for specific expected entries (example)
    # assert ("NodeX", "activation", "NodeY") in tpl, "Expected specific reaction not found in SIF data"


def test_load_compressed_gem():
    from corneto._io import _load_compressed_gem

    file = pathlib.Path(__file__).parent.joinpath("gem", "mitocore.xz")
    S, R, M = _load_compressed_gem(file)

    # Check matrix dimensions
    assert S.shape == (441, 555), f"Expected stoichiometric matrix shape (441, 555), got {S.shape}"
    assert R.shape == (555,), f"Expected reaction vector shape (555,), got {R.shape}"
    assert M.shape == (441,), f"Expected metabolite vector shape (441,), got {M.shape}"

