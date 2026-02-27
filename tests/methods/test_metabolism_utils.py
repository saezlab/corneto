import pytest

from corneto.methods.metabolism._utils import (
    GENE_PATTERN_PRESETS,
    evaluate_gpr,
    evaluate_gpr_expression,
    evaluate_gpr_rules,
    get_genes_from_gpr,
    resolve_pattern,
)


@pytest.mark.parametrize(
    "expression,symbol_values,expected",
    [
        ("", {"A": 1.0}, 0),
        ("A", {}, 0),
        ("A", {"A": 2.5}, 2.5),
        ("A and B", {"A": 3.0, "B": 5.0}, 3.0),
        ("A or B", {"A": 3.0, "B": 5.0}, 5.0),
        ("A and (B or C)", {"A": 5.0, "B": 2.0, "C": 4.0}, 4.0),
        ("(A and B) or C", {"A": 5.0, "B": 2.0, "C": 4.0}, 4.0),
    ],
)
def test_evaluate_gpr_typical_cases(expression, symbol_values, expected):
    assert evaluate_gpr(expression, symbol_values) == expected


def test_evaluate_gpr_with_missing_genes_uses_default_value():
    value = evaluate_gpr("A and B", {"A": 2.0}, default_value=-1.0)
    assert value == -1.0


def test_evaluate_gpr_expression_applies_rule_list():
    expressions = ["A", "A and B", "A or B", "A and (B or C)"]
    symbol_values = {"A": 10.0, "B": 3.0, "C": 7.0}

    values = evaluate_gpr_expression(expressions, symbol_values)

    assert values == [10.0, 3.0, 10.0, 7.0]


def test_evaluate_gpr_rules_multiple_samples_sequential():
    expressions = ["A and B", "A or B"]
    samples = [
        {"A": 2.0, "B": 5.0},
        {"A": -1.0, "B": -3.0},
    ]

    values = evaluate_gpr_rules(expressions, samples, n_processes=0)

    assert values == [[2.0, 5.0], [-1.0, -3.0]]


def test_evaluate_gpr_rules_multiple_samples_parallel():
    expressions = ["A and B", "A or B"]
    samples = [
        {"A": 2.0, "B": 5.0},
        {"A": -1.0, "B": -3.0},
    ]

    values = evaluate_gpr_rules(expressions, samples, n_processes=2)

    assert values == [[2.0, 5.0], [-1.0, -3.0]]


@pytest.mark.parametrize(
    "expression,symbol_values,expected",
    [
        ("A and B", {"A": -2.0, "B": -5.0}, -2.0),
        ("A or B", {"A": -2.0, "B": -5.0}, -5.0),
        ("A and B", {"A": -2.0, "B": 5.0}, -2.0),
        ("A or B", {"A": -2.0, "B": 5.0}, 5.0),
    ],
)
def test_evaluate_gpr_preserves_signed_operator_semantics(expression, symbol_values, expected):
    assert evaluate_gpr(expression, symbol_values) == expected


def test_get_genes_from_gpr_extracts_symbols_only():
    genes = get_genes_from_gpr("(geneA and geneB) or gene_C")
    assert genes == {"geneA", "geneB", "gene_C"}


def test_evaluate_gpr_invalid_syntax_returns_default_value():
    assert evaluate_gpr("A and (B or", {"A": 1.0, "B": 2.0}, default_value=-9.0) == -9.0


def test_get_genes_from_gpr_with_pattern_mode_ecoli_bnumber():
    genes = get_genes_from_gpr(
        "(b0002 and b0114) or geneX",
        regex=None,
        pattern_mode="ecoli_bnumber",
    )
    assert genes == {"b0002", "b0114"}


def test_resolve_pattern_rejects_unknown_mode():
    with pytest.raises(ValueError):
        resolve_pattern(pattern=None, pattern_mode="unknown_mode")


def test_gene_pattern_presets_include_default_mode():
    assert "symbol_simple" in GENE_PATTERN_PRESETS
