import logging
import sys
import importlib
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
