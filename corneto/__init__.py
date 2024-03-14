import sys

from corneto._constants import *

from corneto import _ml as ml
from corneto import _plotting as pl
from corneto._graph import Graph, Attr, Attributes, EdgeType
from corneto._util import info
from corneto.utils import Attr, Attributes

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

__all__ = [
    "Attr",
    "EdgeType",
    "Attributes",
    "Graph",
    "info",
    "DEFAULT_BACKEND",
    "available_backends",
    "K",
    "ops"
]


import_sif = Graph.from_sif

__version__ = "1.0.0-alpha.0"


sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["ml", "pl"]})

del sys
