"""Tests for the shared stable topological-sort utility."""

import pytest

from corneto.graph._topology import topological_sort


def test_topological_sort_preserves_order_isolates_and_parallel_edges():
    order = topological_sort(
        ("A", "B", "C", "isolate"),
        {"A": ("B", "C", "B")},
    )

    assert order == ["A", "isolate", "C", "B"]


def test_topological_sort_rejects_cycles_and_unknown_successors():
    with pytest.raises(ValueError, match="cycle"):
        topological_sort(("A", "B", "isolate"), {"A": ("B",), "B": ("A",)})

    with pytest.raises(ValueError, match="unknown vertex"):
        topological_sort(("A",), {"A": ("missing",)})
