import corneto._settings as s
from corneto.backend._base import (
    Backend,
    NoBackend,
    VarType,
)
from corneto.backend._cvxpy_backend import CvxpyBackend
from corneto.backend._picos_backend import PicosBackend

supported_backends = [CvxpyBackend(), PicosBackend()]

__all__ = ["Backend", "CvxpyBackend", "PicosBackend", "VarType", "s"]


def available_backends():
    return [b for b in supported_backends if b.is_available()]


_cvxpy_mip_solvers = ["GUROBI", "CPLEX", "SCIP", "SCIPY", "CBC", "GLPK_MI"]
_picos_mip_solvers = ["gurobi", "cplex", "scip", "glpk"]


(DEFAULT_BACKEND,) = available_backends()[:1] or (NoBackend(),)
DEFAULT_SOLVER = None

if not DEFAULT_BACKEND:
    s.LOGGER.warn(DEFAULT_BACKEND._error)

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
