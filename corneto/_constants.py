from enum import Enum

VAR_FLOW = "_flow"


class Solver(str, Enum):
    GUROBI = "gurobi"
    COIN_OR_CBC = "cbc"
    CPLEX = ("cplex",)
    GLPK_MI = ("glpk_mi",)
    SCIP = ("scip",)
    CVXOPT = ("cvxopt",)
    MOSEK = "mosek"

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


class IdType(str, Enum):
    REACTION = "reaction"
    SPECIES = "species"


class GraphIdType(str, Enum):
    EDGE = "edge"
    NODE = "node"


class SpeciesType(str, Enum):
    REACTANT = "reactant"
    PRODUCT = "product"
    BOTH = "both"


GLOBAL_SOLVER_PARAMS = ["max_seconds", "verbosity"]
