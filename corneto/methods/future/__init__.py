"""Deprecated compatibility namespace for network inference methods.

Import methods from :mod:`corneto.methods` instead. This namespace remains
available throughout the CORNETO 1.x release line for migration compatibility.

"""

import warnings

warnings.warn(
    "corneto.methods.future is deprecated since CORNETO 1.0.0rc1; "
    "import methods from corneto.methods instead. The compatibility namespace "
    "will be removed in CORNETO 2.0.",
    FutureWarning,
    stacklevel=2,
)

from corneto.methods.future.carnival import CarnivalFlow, CarnivalILP  # noqa: E402
from corneto.methods.future.fba import MultiSampleFBA  # noqa: E402
from corneto.methods.future.imat import MultiSampleIMAT  # noqa: E402
from corneto.methods.future.pcst import PrizeCollectingSteinerTree  # noqa: E402
from corneto.methods.future.steiner import SteinerTreeFlow  # noqa: E402

__all__ = [
    "CarnivalFlow",
    "CarnivalILP",
    "MultiSampleFBA",
    "MultiSampleIMAT",
    "PrizeCollectingSteinerTree",
    "SteinerTreeFlow",
]
