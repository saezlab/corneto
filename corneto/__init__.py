from corneto._constants import *

# from corneto._core import ReNet
from corneto.backend import available_backends, DEFAULT_BACKEND, DEFAULT_SOLVER

# from corneto._core import GReNet as Graph
from corneto.methods import (
    create_flow_graph,
    signflow,
    signflow_constraints,
    default_sign_loss,
)
from corneto.backend._base import HammingLoss as hamming_loss

# from corneto._core import ReNet
from corneto._core import Graph
from corneto._util import info
from corneto.backend import DEFAULT_BACKEND as K

import_sif = Graph.from_sif


def legacy_plot(g: Graph, **kwargs):
    from corneto._legacy import GReNet

    g0 = GReNet.from_ngraph(g)
    g0.plot(**kwargs)


__version__ = "0.9.1-alpha.6"
