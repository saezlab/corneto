from corneto.backend._base import Backend, VarType
from corneto.backend._cvxpy_backend import CvxpyBackend
from corneto.backend._picos_backend import PicosBackend
import corneto._settings as s

supported_backends = [CvxpyBackend(), PicosBackend()]


def available_backends():
    return [b for b in supported_backends if b.is_available()]


DEFAULT_BACKEND = available_backends()[0] if len(available_backends()) > 0 else None

if not DEFAULT_BACKEND:
    s.LOGGER.warn(
        "None of the supported backends found. Please install CVXPY or PICOS to create and solve optimization problems."
    )
