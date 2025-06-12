import logging
import os
import sys
from importlib.util import find_spec

import numpy as np

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

# Experimental
USE_NUMBA = find_spec("numba") and not os.environ.get("CORNETO_IGNORE_NUMBA", False)

try:
    from scipy import sparse  # type: ignore

    sparsify = sparse.csr_matrix
    LOGGER.debug("Using scipy csr sparse matrices by default")
except ImportError:

    def sparsify(x):
        LOGGER.warning("Scipy not installed, using numpy dense matrices instead.")
        return x

    LOGGER.debug("Scipy not installed, using numpy dense matrices instead.")


def _numpy_array(arg1, shape=None, dtype=None):
    if isinstance(arg1, np.ndarray):
        return arg1

    # Handle (data, (row_ind, col_ind)) case
    if isinstance(arg1, tuple) and len(arg1) == 2 and isinstance(arg1[1], tuple) and len(arg1[1]) == 2:
        data, (row_ind, col_ind) = arg1
        if shape is None:
            shape = (max(row_ind) + 1, max(col_ind) + 1)
        mat = np.zeros(shape, dtype=dtype)
        for d, r, c in zip(data, row_ind, col_ind):
            mat[r, c] = d
        return mat

    # Handle (data, indices, indptr) case
    elif isinstance(arg1, tuple) and len(arg1) == 3:
        data, indices, indptr = arg1
        if shape is None:
            shape = (len(indptr) - 1, max(indices) + 1)
        mat = np.zeros(shape, dtype=dtype if dtype is not None else float)
        for i in range(len(indptr) - 1):
            for j in range(indptr[i], indptr[i + 1]):
                mat[i, indices[j]] = data[j]
        return mat

    else:
        raise ValueError("Invalid input format")


def _get_matrix_builder():
    try:
        from scipy import sparse  # type: ignore

        return sparse.csr_array
    except ImportError:
        return _numpy_array
