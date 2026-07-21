"""Deprecated compatibility namespace for signaling methods."""

import warnings

warnings.warn(
    "corneto.methods.signal is deprecated since CORNETO 1.0.0rc1; use "
    "corneto.methods.signaling instead. The compatibility package will be "
    "removed in CORNETO 2.0.",
    FutureWarning,
    stacklevel=2,
)
