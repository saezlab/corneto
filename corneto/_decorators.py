import os
from functools import wraps
from importlib.util import find_spec
from typing import Callable
from corneto._settings import LOGGER

USE_NUMBA = find_spec("numba") and not os.environ.get("CORNETO_IGNORE_NUMBA", False)


def _jit(*_args, **_kwargs):
    def _dummy_jit(func):
        @wraps(func)
        def _wrapped_func(*args, **kwargs):
            return func(*args, **kwargs)

        return _wrapped_func

    return _dummy_jit


jit: Callable = _jit

if USE_NUMBA:
    try:
        from numba import jit as _numba_jit

        jit = _numba_jit
    except ImportError as e:
        LOGGER.warn(getattr(e, "message", repr(e)))


def _proxy(func):
    @wraps(func)
    def _wrapper_func(self, *args, **kwargs):
        if len(args) > 0:
            # Function is providing 'other' expression
            if hasattr(args[0], "_expr"):
                args = list(args)
                args[0] = args[0]._expr
        # Call the function in the original expression (PICOS or CVXPY) if available
        if hasattr(self._expr, func.__name__):
            return self._create(getattr(self._expr, func.__name__)(*args, **kwargs))
        return self._create(func(self, *args, **kwargs))

    return _wrapper_func
