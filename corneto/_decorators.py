import os
from functools import wraps
from importlib.util import find_spec
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
        LOGGER.warn(getattr(e, "message", repr(e)))
        jit = _jit


def _proxy(func):
    @wraps(func)
    def _wrapper_func(self, *args, **kwargs):
        symbols = set()
        # if hasattr(self, '_proxy_symbols'):
        #    symbols.update(self._proxy_symbols)
        # if getattr(self, 'is_symbol', lambda: False)():
        #    symbols.add(self)
        if len(args) > 0:
            # Function is providing 'other' expression
            if hasattr(args[0], "_expr"):
                args = list(args)
                symbols.update(args[0]._proxy_symbols)
                if getattr(args[0], "is_symbol", lambda: False)():
                    symbols.add(args[0])
                args[0] = args[0]._expr  # symbol is lost
        if hasattr(self._expr, func.__name__):
            # Call the function in the original expression (PICOS or CVXPY) if available
            # e.g., if function is __add__, checks if the PICOS/CVXPY expression has that function
            # and uses it instead, this returns a new PICOS/CVXPY expression which is wrapped in a CtProxyExpression
            return self._create(
                getattr(self._expr, func.__name__)(*args, **kwargs), symbols
            )
        # This is not a native function, run it and create a new CtProxyExpression
        return self._create(func(self, *args, **kwargs), symbols)

    return _wrapper_func
