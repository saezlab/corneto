import sys

from corneto import _plotting as pl
from corneto._constants import *
from corneto._graph import Attr, Attributes, EdgeType, Graph
from corneto._util import info
from corneto.backend import DEFAULT_BACKEND, DEFAULT_SOLVER, available_backends
from corneto.backend import DEFAULT_BACKEND as K
from corneto.backend import DEFAULT_BACKEND as ops
from corneto.backend._base import HammingLoss as hamming_loss
from corneto.backend._base import Indicator, NonZeroIndicator

# from corneto._core import GReNet as Graph
from corneto.methods import (
    create_flow_graph,
    default_sign_loss,
    signaling,
    signflow_constraints,
)
from corneto.utils import Attr, Attributes

__all__ = [
    "Attr",
    "EdgeType",
    "Attributes",
    "Graph",
    "info",
    "DEFAULT_BACKEND",
    "available_backends",
    "K",
    "ops",
]


import_sif = Graph.from_sif

__version__ = "1.0.0.dev0"


sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["pl"]})

del sys
