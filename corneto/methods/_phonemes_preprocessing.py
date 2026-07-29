"""Dependency-free preprocessing helpers for PHONEMeS."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np


def _finite_threshold(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number, got {value!r}.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if positive and result <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value!r}.")
    return result


def _scale_scores(scores: np.ndarray) -> np.ndarray:
    values = scores[:, None] if scores.ndim == 1 else scores
    negative_scale = np.max(np.where(values < 0, -values, 0), axis=0, keepdims=True)
    positive_scale = np.max(np.where(values > 0, values, 0), axis=0, keepdims=True)
    negative_scale = np.where(negative_scale > 0, negative_scale, 1)
    positive_scale = np.where(positive_scale > 0, positive_scale, 1)
    scaled = np.where(values < 0, values / negative_scale, values / positive_scale)
    return scaled[:, 0] if scores.ndim == 1 else scaled


def _score_array(
    pvalues: Any,
    *,
    fold_changes: Any,
    pvalue_threshold: float,
    fold_change_threshold: float | None,
    direction: str,
    scale: bool,
) -> np.ndarray:
    pvalue_array = np.asarray(pvalues, dtype=float)
    if pvalue_array.ndim not in {1, 2}:
        raise ValueError("pvalues must be one-dimensional (sites) or two-dimensional (sites by conditions).")
    if pvalue_array.size == 0:
        raise ValueError("pvalues must not be empty.")
    if not np.all(np.isfinite(pvalue_array)):
        raise ValueError("pvalues must contain only finite values.")
    if np.any((pvalue_array < 0) | (pvalue_array > 1)):
        raise ValueError("pvalues must be between 0 and 1.")

    supported = pvalue_array < pvalue_threshold
    if fold_change_threshold is not None:
        if fold_changes is None:
            raise ValueError("fold_changes is required when fold_change_threshold is provided.")
        fold_change_array = np.asarray(fold_changes, dtype=float)
        if fold_change_array.shape != pvalue_array.shape:
            raise ValueError(
                "fold_changes must have the same shape as pvalues: "
                f"expected {pvalue_array.shape}, got {fold_change_array.shape}."
            )
        if not np.all(np.isfinite(fold_change_array)):
            raise ValueError("fold_changes must contain only finite values.")
        if direction == "both":
            supported &= np.abs(fold_change_array) >= fold_change_threshold
        elif direction == "up":
            supported &= fold_change_array >= fold_change_threshold
        else:
            supported &= fold_change_array <= -fold_change_threshold

    safe_pvalues = np.maximum(pvalue_array, np.finfo(float).tiny)
    magnitude = np.abs(np.log2(safe_pvalues / pvalue_threshold))
    scores = np.where(supported, -magnitude, magnitude)
    return _scale_scores(scores) if scale else scores


def _mapping_kind(values: Mapping, *, name: str) -> str:
    if not values:
        raise ValueError(f"{name} must not be empty.")
    is_nested = [isinstance(value, Mapping) for value in values.values()]
    if all(is_nested):
        return "nested"
    if any(is_nested):
        raise TypeError(f"{name} must be either a flat site mapping or a uniformly nested condition-to-site mapping.")
    return "flat"


def _matching_mapping(reference: Mapping, value: Any, *, name: str) -> Mapping | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must use the same mapping structure as pvalues.")
    if tuple(value) != tuple(reference):
        if set(value) != set(reference):
            raise ValueError(f"{name} keys must match pvalues keys.")
        return {key: value[key] for key in reference}
    return value


def _score_mapping(
    pvalues: Mapping,
    *,
    fold_changes: Any,
    pvalue_threshold: float,
    fold_change_threshold: float | None,
    direction: str,
    scale: bool,
) -> dict:
    kind = _mapping_kind(pvalues, name="pvalues")
    fold_mapping = _matching_mapping(pvalues, fold_changes, name="fold_changes")
    if kind == "nested":
        return {
            condition: _score_mapping(
                condition_pvalues,
                fold_changes=None if fold_mapping is None else fold_mapping[condition],
                pvalue_threshold=pvalue_threshold,
                fold_change_threshold=fold_change_threshold,
                direction=direction,
                scale=scale,
            )
            for condition, condition_pvalues in pvalues.items()
        }

    identifiers = tuple(pvalues)
    score_values = _score_array(
        [pvalues[identifier] for identifier in identifiers],
        fold_changes=(None if fold_mapping is None else [fold_mapping[identifier] for identifier in identifiers]),
        pvalue_threshold=pvalue_threshold,
        fold_change_threshold=fold_change_threshold,
        direction=direction,
        scale=scale,
    )
    return dict(zip(identifiers, score_values.tolist()))


def _optional_pandas():
    try:
        import pandas as pd
    except ImportError:
        return None
    return pd


def normalize_phonemes_score_mapping(scores, *, many: bool):
    """Return the mapping form accepted by PHONEMeS build methods."""
    if isinstance(scores, Mapping):
        return scores
    pd = _optional_pandas()
    expected = "a condition-by-site mapping or pandas DataFrame" if many else "a site mapping or pandas Series"
    if pd is None:
        raise TypeError(f"phosphosite_scores must be {expected}.")
    if many and isinstance(scores, pd.DataFrame):
        if not scores.index.is_unique or not scores.columns.is_unique:
            raise ValueError("phosphosite_scores DataFrame must have unique phosphosite and condition labels.")
        return scores.to_dict()
    if not many and isinstance(scores, pd.Series):
        if not scores.index.is_unique:
            raise ValueError("phosphosite_scores Series must have unique phosphosite labels.")
        return scores.to_dict()
    raise TypeError(f"phosphosite_scores must be {expected}.")


def compute_phonemes_scores(
    pvalues,
    *,
    fold_changes=None,
    pvalue_threshold: float = 0.05,
    fold_change_threshold: float | None = None,
    direction: str = "both",
    scale: bool = True,
):
    """Compute PHONEMeS node scores from p-values and fold changes.

    Flat mappings, pandas Series, and one-dimensional arrays represent one
    condition. Nested mappings, pandas DataFrames, and two-dimensional arrays
    represent sites by conditions. The returned object preserves the input
    container type for NumPy and pandas inputs; mappings return dictionaries.

    Negative scores reward measurements satisfying the selected evidence
    criteria, while positive scores penalize unsupported measurements.

    Args:
        pvalues: P-values indexed or ordered by phosphosite.
        fold_changes: Optional fold changes with the same structure and labels.
        pvalue_threshold: Significance threshold in ``(0, 1]``.
        fold_change_threshold: Optional nonnegative absolute fold-change threshold.
        direction: Fold-change direction: ``"both"``, ``"up"``, or ``"down"``.
        scale: Scale positive and negative scores separately per condition.

    Returns:
        Scores with the same labels and container type as ``pvalues``.
    """
    threshold = _finite_threshold(
        pvalue_threshold,
        name="pvalue_threshold",
        positive=True,
    )
    if threshold > 1:
        raise ValueError("pvalue_threshold must be less than or equal to 1.")
    if fold_change_threshold is not None:
        fold_change_threshold = _finite_threshold(
            fold_change_threshold,
            name="fold_change_threshold",
        )
        if fold_change_threshold < 0:
            raise ValueError("fold_change_threshold must be nonnegative.")
    if direction not in {"both", "up", "down"}:
        raise ValueError("direction must be 'both', 'up', or 'down'.")
    if not isinstance(scale, bool):
        raise TypeError(f"scale must be a boolean, got {scale!r}.")

    if isinstance(pvalues, Mapping):
        return _score_mapping(
            pvalues,
            fold_changes=fold_changes,
            pvalue_threshold=threshold,
            fold_change_threshold=fold_change_threshold,
            direction=direction,
            scale=scale,
        )

    pd = _optional_pandas()
    if pd is not None and isinstance(pvalues, (pd.Series, pd.DataFrame)):
        if not pvalues.index.is_unique:
            raise ValueError("pvalues index must contain unique phosphosite identifiers.")
        if isinstance(pvalues, pd.DataFrame) and not pvalues.columns.is_unique:
            raise ValueError("pvalues columns must contain unique condition names.")

        aligned_fold_changes = None
        if fold_changes is not None:
            if type(fold_changes) is not type(pvalues):
                raise TypeError("fold_changes must have the same pandas container type as pvalues.")
            if isinstance(pvalues, pd.Series):
                if set(fold_changes.index) != set(pvalues.index):
                    raise ValueError("fold_changes index must match pvalues index.")
                aligned_fold_changes = fold_changes.reindex(pvalues.index).to_numpy()
            else:
                if set(fold_changes.index) != set(pvalues.index) or set(fold_changes.columns) != set(pvalues.columns):
                    raise ValueError("fold_changes index and columns must match pvalues.")
                aligned_fold_changes = fold_changes.reindex(
                    index=pvalues.index,
                    columns=pvalues.columns,
                ).to_numpy()

        scores = _score_array(
            pvalues.to_numpy(),
            fold_changes=aligned_fold_changes,
            pvalue_threshold=threshold,
            fold_change_threshold=fold_change_threshold,
            direction=direction,
            scale=scale,
        )
        if isinstance(pvalues, pd.Series):
            return pd.Series(scores, index=pvalues.index, name=pvalues.name)
        return pd.DataFrame(scores, index=pvalues.index, columns=pvalues.columns)

    return _score_array(
        pvalues,
        fold_changes=fold_changes,
        pvalue_threshold=threshold,
        fold_change_threshold=fold_change_threshold,
        direction=direction,
        scale=scale,
    )
