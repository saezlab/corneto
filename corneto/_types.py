from typing import TYPE_CHECKING, Any, FrozenSet, Tuple, TypeVar, Union

from corneto._settings import LOGGER

# ------------------------------------------------------------------------------
# Basic Type Definitions
# ------------------------------------------------------------------------------

Edge = Tuple[FrozenSet[Any], FrozenSet[Any]]
StrOrInt = Union[str, int]
TupleSIF = Tuple[str, int, str]


def _import_optional(module_name: str, attribute: str) -> Any:
    """Attempt to import an attribute from a given module.

    Logs whether the import was successful or not. If the import fails,
    returns `Any` to serve as a fallback type.

    Args:
        module_name (str): The name of the module.
        attribute (str): The name of the attribute to import.

    Returns:
        The imported attribute, or `Any` if the import fails.
    """
    try:
        # Import the module and get the specified attribute.
        module = __import__(module_name, fromlist=[attribute])
        attr = getattr(module, attribute)
        LOGGER.debug(f"Successfully loaded '{attribute}' from '{module_name}'.")
        return attr
    except (ImportError, AttributeError) as e:
        # Log the failure and fallback to Any.
        LOGGER.warning(f"Failed to import '{attribute}' from '{module_name}': {e}. Defaulting to Any.")
        return Any


# ------------------------------------------------------------------------------
# Optional Dependency Imports for Type Checking
# ------------------------------------------------------------------------------

if TYPE_CHECKING:
    # During static type checking, attempt to load optional dependencies
    # to provide more precise type hints.
    _Model = _import_optional("cobra.core.model", "Model")
    _NxGraph = _import_optional("networkx", "Graph")
    _NxDiGraph = _import_optional("networkx", "DiGraph")
else:
    # At runtime, these types are not needed for functionality,
    # so they default to Any to avoid unnecessary imports and potential errors.
    _Model = Any
    _NxGraph = Any
    _NxDiGraph = Any


# ------------------------------------------------------------------------------
# Type Variables
# ------------------------------------------------------------------------------

# These TypeVars use the optionally loaded types as bounds. When the optional
# packages are available, they allow for stricter type checking.
CobraModel = TypeVar("CobraModel", bound=_Model)
NxGraph = TypeVar("NxGraph", bound=_NxGraph)
NxDiGraph = TypeVar("NxDiGraph", bound=_NxDiGraph)
