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
    """
    A decorator that wraps a function to provide extended functionality
    when applied within a class. This decorator modifies the behavior
    of the function `func` to handle expression objects and delegate
    calls to their underlying representations, while maintaining a set of
    symbols associated with the expression objects.

    The primary use of this decorator is to allow mathematical and
    operational transformations on proxy objects (like expressions in a
    symbolic or algebraic framework) that abstract underlying complex
    behaviors (like algebraic expressions handled by a computational backend
    such as PICOS or CVXPY).

    Parameters:
    func (Callable): The function to be wrapped. This function should be
                     a method of a class that handles expressions. It is
                     expected to operate on instances of the class and
                     potentially other similar objects.

    Returns:
    Callable: A wrapper function `_wrapper_func` that takes the same arguments as `func`.
              This function intercepts calls to `func`, updates and manages symbols,
              and delegates operations to the underlying computational backend if possible.

    Decorators:
    @wraps(func): This decorator is used to preserve the name, docstring, and other
                  attributes of the original function `func`.

    Usage:
    To use this decorator, apply it to methods in a class that represents expressions,
    where such methods need to interact with the underlying computational or symbolic
    representation of those expressions. The decorator handles conversion and delegation
    logic, facilitating the interaction with more complex backends transparently.

    Example:
    ```python
    class Expression:
        def _create(self, expr, symbols):
            # Implementation details...
            pass

        @_delegate
        def __add__(self, other):
            # Additional functionality can be inserted here.
            pass
    ```
    """
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
            # Call the function in the original expression (PICOS/CVXPY/.. backend)
            # if available. E.g., if function is __add__, checks if the backend
            # expression has that function and uses it instead, this returns a
            # new backend expression which is wrapped back to CORNETO expr.
            return self._create(
                getattr(self._expr, func.__name__)(*args, **kwargs), symbols
            )
        return self._create(func(self, *args, **kwargs), symbols)

    return _wrapper_func
