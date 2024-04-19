from corneto.utils._attr import Attr, Attributes
import re
from pathlib import Path
import importlib


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
        raise RuntimeError(
            "pyproject.toml not found. Ensure your project structure is correct."
        )


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
