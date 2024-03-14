from typing import TYPE_CHECKING, FrozenSet, Tuple, Union, Any, TypeVar
from corneto._settings import LOGGER
import importlib

Edge = Tuple[FrozenSet[Any], FrozenSet[Any]]
StrOrInt = Union[str, int]
TupleSIF = Tuple[str, int, str]


def _safe_load(module_name, class_name):
    try:
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        LOGGER.debug(f"f{class_name} from {module_name} successfully loaded")
        return class_
    except (ImportError, AttributeError):
        return Any


_Model = Any
_NxGraph = Any
_NxDiGraph = Any

if TYPE_CHECKING:
    _Model = _safe_load("cobra.core.model", "Model")
    _NxGraph = _safe_load("networkx", "Graph")
    _NxDiGraph = _safe_load("networkx", "DiGraph")


CobraModel = TypeVar("CobraModel", bound=_Model)
NxGraph = TypeVar("NxGraph", bound=_NxGraph)
NxDiGraph = TypeVar("NxDiGraph", bound=_NxDiGraph)
