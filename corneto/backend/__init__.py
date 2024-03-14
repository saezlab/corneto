# from importlib.util import find_spec
from typing import Optional
from corneto.backend._base import Backend, VarType, Indicator, NonZeroIndicator
from corneto.backend._cvxpy_backend import CvxpyBackend
from corneto.backend._picos_backend import PicosBackend
import corneto._settings as s

supported_backends = [CvxpyBackend(), PicosBackend()]

__all__ = ["Backend", "VarType", "CvxpyBackend", "PicosBackend", "s"]


def available_backends():
    return [b for b in supported_backends if b.is_available()]


_cvxpy_mip_solvers = ["GUROBI", "CPLEX", "MOSEK", "SCIP", "CBC", "GLPK_MI"]
_picos_mip_solvers = ["gurobi", "cplex", "mosek", "scip", "glpk"]


DEFAULT_BACKEND: Optional[Backend] = (
    available_backends()[0] if len(available_backends()) > 0 else None
)
DEFAULT_SOLVER = None

if not DEFAULT_BACKEND:
    s.LOGGER.warn(
        "None of the supported backends found. You can install the default backend with `pip install cvxpy`."
    )
else:
    if isinstance(DEFAULT_BACKEND, CvxpyBackend):
        from cvxpy import installed_solvers

        available = [name.lower() for name in installed_solvers()]
        for solver in _cvxpy_mip_solvers:
            if solver.lower() in available:
                DEFAULT_SOLVER = solver
                break
    else:
        import picos as pc

        available = [name.lower() for name in pc.available_solvers()]
        for solver in _picos_mip_solvers:
            if solver.lower() in available:
                DEFAULT_SOLVER = solver
                break
    if not DEFAULT_SOLVER:
        s.LOGGER.warn(f"MIP solver not found for {DEFAULT_BACKEND}")
