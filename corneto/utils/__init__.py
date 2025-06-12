import importlib
import re
from functools import wraps
from pathlib import Path

from corneto.utils._attr import Attr, Attributes
from corneto.utils._citations import (
    format_authors,
    get_bibtex_from_keys,
    parse_bibtex,
    render_references_html,
    show_bibtex,
    show_references,
)
from corneto.utils._solvers import check_gurobi


class OptionalModule:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None
        try:
            self.module = importlib.import_module(module_name)
        except ImportError:
            pass

    def __getattr__(self, item):
        if self.module:
            return getattr(self.module, item)
        else:
            return self._create_dummy(item)

    def _create_dummy(self, name):
        def dummy(*args, **kwargs):
            return None

        return dummy


class OptionalNumba(OptionalModule):
    def __init__(self):
        super().__init__("numba")

    def _create_dummy(self, name):
        if name == "jit":

            def _jit(*_args, **_kwargs):
                def _dummy_jit(func):
                    @wraps(func)
                    def _wrapped_func(*args, **kwargs):
                        return func(*args, **kwargs)

                    return _wrapped_func

                return _dummy_jit

            return _jit

        if name in ["uint16", "uint32", "uint64", "int16", "int32", "int64"]:
            return int
        if name in ["float32", "float64", "complex64", "complex128"]:
            return float
        if name == "prange":
            return range

        return super()._create_dummy(name)


# Create an instance for numba
numba = OptionalModule("numba")


def get_library_version(lib_name):
    pyproject_path = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    try:
        with pyproject_path.open("r", encoding="utf-8") as f:
            pyproject_content = f.read()

        # Normalize the library name to lower case to handle case-insensitive search
        lib_name = lib_name.lower()
        # Regex pattern to extract library details
        # This pattern looks for the library name, then captures the version and checks if it is optional
        pattern = re.compile(
            rf"{lib_name}\s*=\s*\{{?\s*version\s*=\s*\"([^\"]+)\"[^}}]*optional\s*=\s*(true|false)",
            re.IGNORECASE,
        )

        # Search for the pattern in the provided text
        match = pattern.search(pyproject_content)
        if match:
            # Check if the library is marked as optional
            if match.group(2).lower() == "true":
                return match.group(1)
            else:
                return None
        else:
            raise ValueError(f"Library {lib_name} not found in pyproject.toml")

    except FileNotFoundError:
        raise RuntimeError("pyproject.toml not found. Ensure your project structure is correct.")


def import_optional_module(module_name):
    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError as e:
        expected_version = get_library_version(module_name)
        if expected_version:
            error_msg = f"{module_name} version {expected_version} is required. Install it with 'pip install {module_name}' or 'pip install corneto[{module_name}]'"
        else:
            error_msg = f"{module_name} is not installed and no version specification found in pyproject.toml. It may be an optional dependency not configured."
        raise ImportError(error_msg) from e
