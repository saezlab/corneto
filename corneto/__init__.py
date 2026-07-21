"""Public CORNETO API."""

import sys
import warnings

from corneto import _plotting as pl
from corneto._constants import Direction, VarType
from corneto._logging import disable_logging, enable_logging, set_verbosity
from corneto._util import info, suppress_output
from corneto.backend import DEFAULT_BACKEND, DEFAULT_SOLVER, available_backends
from corneto.backend import DEFAULT_BACKEND as opt
from corneto.data import Data, Feature, GraphData, Sample
from corneto.graph import Attr, Attributes, EdgeType, Graph


def get_version():
    import os
    import re

    here = os.path.abspath(os.path.dirname(__file__))
    pyproject_path = os.path.join(here, "..", "pyproject.toml")

    with open(pyproject_path, "r") as f:
        content = f.read()

    # Regex to find the version number
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Version not found in pyproject.toml.")


_DEPRECATED_BACKEND_ALIASES = {"K", "ops"}


def __getattr__(name):
    if name in _DEPRECATED_BACKEND_ALIASES:
        warnings.warn(
            f"corneto.{name} is deprecated since CORNETO 1.0.0rc1; use "
            "corneto.opt instead. The alias will be removed in CORNETO 2.0.",
            FutureWarning,
            stacklevel=2,
        )
        return opt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _DEPRECATED_BACKEND_ALIASES)


__all__ = [
    "DEFAULT_BACKEND",
    "DEFAULT_SOLVER",
    "Attr",
    "Attributes",
    "Data",
    "Direction",
    "EdgeType",
    "Feature",
    "Graph",
    "GraphData",
    "K",
    "Sample",
    "VarType",
    "available_backends",
    "disable_logging",
    "enable_logging",
    "info",
    "ops",
    "opt",
    "pl",
    "set_verbosity",
    "suppress_output",
]

try:
    # Python 3.8 and newer
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    # Python < 3.8
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("corneto")
except PackageNotFoundError:
    # Source checkout mode (not installed): keep import working.
    try:
        __version__ = get_version()
    except Exception:
        __version__ = "0+unknown"

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["pl"]})

del sys
