from functools import wraps
from typing import Any, Callable, Optional, Set, Union

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


def _delegate(
    func: Optional[Callable] = None, *, override: bool = False
) -> Union[Callable, Callable[[Callable], Callable]]:
    """A decorator for delegating method calls to a backend expression if available.

    The decorator checks if `self._expr` has a callable attribute with
    the same name as the decorated function. If so, and if `override` is False,
    the backend method is called instead.

    It also processes the first positional argument (if any) by:
      - Checking if it has an `_expr` attribute.
      - Updating a set of proxy symbols from `_proxy_symbols`.
      - Replacing the argument with its underlying expression.

    Parameters:
        func (Callable, optional): The function to decorate.
        override (bool): If True, always call the decorated function.
                         If False (default) and the backend provides a
                         callable attribute, delegate to it.

    Returns:
        The wrapped function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def _wrapper_func(self, *args, **kwargs):
            # Initialize a set to collect proxy symbols.
            symbols: Set[Any] = set()
            # Work with mutable list for potential argument modifications.
            args_list = list(args)

            # Process the first argument if it has an '_expr' attribute.
            if args_list and hasattr(args_list[0], "_expr"):
                other = args_list[0]
                # Merge any existing proxy symbols.
                symbols.update(getattr(other, "_proxy_symbols", set()))
                if getattr(other, "is_symbol", lambda: False)():
                    symbols.add(other)
                # Replace with the underlying expression.
                args_list[0] = other._expr
                # Propagate the proxy symbols to the underlying expression.
                setattr(args_list[0], "_proxy_symbols", symbols)

            # Attempt to delegate to the backend if available and not overridden.
            backend_method = getattr(self._expr, func.__name__, None)
            if not override and callable(backend_method):
                result = backend_method(*args_list, **kwargs)
            else:
                result = func(self, *args_list, **kwargs)
            return self._create(result, symbols)

        return _wrapper_func

    # Allow decorator usage with or without parameters.
    if func is None:
        return decorator
    else:
        return decorator(func)
