import importlib
from functools import wraps

from corneto._settings import LOGGER


class OptionalModule:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None
        try:
            self.module = importlib.import_module(module_name)
        except ImportError:
            LOGGER.debug(f"Optional module {module_name} not found.")

    def __getattr__(self, item):
        if self.module:
            return getattr(self.module, item)
        else:
            return self._create_dummy(item)

    def _create_dummy(self, name):
        def _dummy(*args, **kwargs):
            def _decorator(func):
                @wraps(func)
                def _wrapped_func(*_args, **_kwargs):
                    return func(*_args, **_kwargs)

                return _wrapped_func

            return _decorator

        return _dummy
