from corneto.methods import runVanillaCarnival


def test_vanilla_carnival():
    pkn = [
        ("I1", 1, "N1"),
        ("N1", 1, "M1"),
        ("N1", 1, "M2"),
        ("I2", -1, "N2"),
        ("N2", -1, "M2"),
        ("N2", -1, "M1"),
    ]
    measurements = {"M1": 1, "M2": 1}
    perturbations = {"I1": 1, "I2": 1}
    p, Gf = runVanillaCarnival(perturbations, measurements, pkn)
    V = list(Gf.vertices)
    act = p.get_symbol("species_activated_c0").value
    inh = p.get_symbol("species_inhibited_c0").value
    val = act - inh
    assert val[V.index("M1")] == 1
    assert val[V.index("M2")] == 1
    assert val[V.index("N1")] >= 0
    assert val[V.index("N2")] <= 0
    assert (abs(val[V.index("N1")]) + abs(val[V.index("N2")])) == 1
