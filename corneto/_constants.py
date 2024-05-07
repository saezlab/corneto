from enum import Enum

VAR_FLOW = "_flow"
DEFAULT_LB = -10
DEFAULT_UB = 10

EXPR_NAME_FLOW_NZI = "with_flow"
EXPR_NAME_FLOW = "flow"
EXPR_NAME_FLOW_IPOS = "positive_flow"
EXPR_NAME_FLOW_INEG = "negative_flow"

VAR_DAG = "_dag_layer"


class Solver(str, Enum):
    GUROBI = "gurobi"
    COIN_OR_CBC = "cbc"
    CPLEX = "cplex"
    GLPK_MI = "glpk_mi"
    SCIP = "scip"
    CVXOPT = "cvxopt"
    MOSEK = "mosek"
    SCIPY = "scipy"

    def to_cvxpy(self) -> str:
        return self.value.upper()


class Backends(str, Enum):
    CVXPY = ("cvxpy",)
    PICOS = ("picos",)


class VarType(str, Enum):
    INTEGER = "integer"
    CONTINUOUS = "continuous"
    BINARY = "binary"


class Direction(str, Enum):
    MAX = "max"
    MIN = "min"


GLOBAL_SOLVER_PARAMS = ["max_seconds", "verbosity"]
