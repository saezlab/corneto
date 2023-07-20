import logging
import sys
import os
import numpy as np
from importlib.util import find_spec
from functools import wraps

LOGGER = logging.getLogger("__corneto__")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.INFO)
_stream_handler.setFormatter(
    logging.Formatter(
        fmt="(CORNETO) %(asctime)-18s - %(levelname)-8s: %(message)s",
        datefmt="%b %d %I:%M:%S %p",
    )
)
LOGGER.addHandler(_stream_handler)


USE_NUMBA = find_spec("numba") and not os.environ.get("CORNETO_IGNORE_NUMBA", False)

try_sparse = lambda x: x

if find_spec("scipy"):
    try:
        from scipy import sparse

        try_sparse = lambda x: sparse.csr_matrix(x)
        LOGGER.debug(f"Using scipy csr sparse matrices by default")
    except Exception as e:
        LOGGER.debug(f"Scipy not installed, using numpy dense matrices instead: {e}")
