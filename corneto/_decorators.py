from functools import wraps
from typing import Callable

from corneto._settings import LOGGER, USE_NUMBA


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
    except Exception as e:
        LOGGER.warning(getattr(e, "message", repr(e)))
        jit = _jit


def _delegate(func):
    @wraps(func)
    def _wrapper_func(self, *args, **kwargs):
        symbols = set()
        if len(args) > 0:
            # Function is providing 'other' expression
            if hasattr(args[0], "_expr"):
                args = list(args)
                symbols.update(args[0]._proxy_symbols)
                if getattr(args[0], "is_symbol", lambda: False)():
                    symbols.add(args[0])
                # Extract the original backend symbol
                args[0] = args[0]._expr
                # Attach the list of original symbols to the backend expression
                setattr(args[0], "_proxy_symbols", symbols)
        if hasattr(self._expr, func.__name__):
            # Check if its callable
            f = getattr(self._expr, func.__name__)
            if callable(f):
                # Call the function in the original expression (PICOS/CVXPY/.. backend)
                # if available. E.g., if function is __add__, checks if the backend
                # expression has that function and uses it instead, this returns a
                # new backend expression which is wrapped back to CORNETO expr.
                return self._create(f(*args, **kwargs), symbols)
        return self._create(func(self, *args, **kwargs), symbols)

    return _wrapper_func
