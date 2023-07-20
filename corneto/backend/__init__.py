from importlib.util import find_spec
from corneto.backend._base import Backend, VarType
from corneto.backend._cvxpy_backend import CvxpyBackend
from corneto.backend._picos_backend import PicosBackend
import corneto._settings as s

supported_backends = [CvxpyBackend(), PicosBackend()]


def available_backends():
    return [b for b in supported_backends if b.is_available()]

# TODO: fix the way we check for solvers
_cvxpy_solvers = ["gurobi", "cplex", "mosek", "scip", "cbc", "glpk_mi", "scipy", "copt"]
_picos_solvers = ["gurobi", "cplex", "mosek", "scip", "glpk", "scipy", "copt"]


DEFAULT_BACKEND = available_backends()[0] if len(available_backends()) > 0 else None
DEFAULT_SOLVER = None

if not DEFAULT_BACKEND:
    s.LOGGER.warn(
        "None of the supported backends found. Please install CVXPY or PICOS to create and solve optimization problems."
    )
else:
    if isinstance(DEFAULT_BACKEND, CvxpyBackend):
        from cvxpy import installed_solvers

        available = [name.lower() for name in installed_solvers()]
        for solver in _cvxpy_solvers:
            if solver in available:
                DEFAULT_SOLVER = solver
                break
    else:
        import picos as pc

        available = [name.lower() for name in pc.available_solvers()]
        for solver in _picos_solvers:
            if solver in available:
                DEFAULT_BACKEND = solver
                break
    if not DEFAULT_SOLVER:
        s.LOGGER.warn(f"MIP solver not found for {DEFAULT_BACKEND}")
