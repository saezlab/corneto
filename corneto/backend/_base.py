import abc
import numbers
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from corneto._constants import *
from corneto._decorators import _delegate
from corneto._graph import BaseGraph
from corneto._settings import LOGGER, _get_matrix_builder
from corneto.utils import Attributes


def _eq_shape(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        if len(a.shape) == 1 and len(b.shape) == 2:
            return a.shape[0] == b.shape[0] and b.shape[1] == 1
        if len(a.shape) == 2 and len(b.shape) == 1:
            return a.shape[0] == b.shape[0] and a.shape[1] == 1
    return a.shape == b.shape


def _identical_columns(array):
    # Get the first column as a reference column
    ref_column = array[:, 0]

    # Compare all columns to the reference column
    # np.all will check if all elements in the result are True along axis 0 (down the rows)
    are_columns_identical = np.all(array == ref_column[:, np.newaxis], axis=0)

    # np.all on the result checks if all columns are identical to the reference column
    return np.all(are_columns_identical)


class CExpression(abc.ABC):
    # Arithmetic operator overloading with Numpy
    # See: https://www.cvxpy.org/_modules/cvxpy/expressions/expression.html#Expression
    __array_priority__ = 100

    def __init__(self, expr: Any, symbols: Optional[Set["CSymbol"]] = None) -> None:
        super().__init__()
        self._expr = expr
        self._proxy_symbols: Set["CSymbol"] = set()
        self._name = ""
        if symbols:
            self._proxy_symbols.update(symbols)

    def is_symbol(self) -> bool:
        return False

    def _create(self, expr: Any, atoms: Iterable) -> "CExpression":
        symbols = {s for s in atoms if isinstance(s, CSymbol)}
        if isinstance(self, CSymbol):
            symbols.add(self)
        if isinstance(self, CExpression):
            symbols.update(self._proxy_symbols)
        if isinstance(expr, CSymbol):
            symbols.add(expr)
        if isinstance(expr, CExpression):
            symbols.update(expr._proxy_symbols)
        # Ask to create a CVXPY/PICOS/.. expression
        return self._create_proxy_expr(expr, symbols)

    @property
    def name(self):
        return self._name

    @abc.abstractmethod
    def _create_proxy_expr(
        self, expr: Any, symbols: Optional[Set["CSymbol"]] = None
    ) -> "CExpression":
        pass

    @property
    @abc.abstractmethod
    def value(self) -> np.ndarray:
        pass

    @property
    def e(self) -> Any:
        return self._expr

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._expr.shape

    def __hash__(self) -> int:
        return self._expr.__hash__()

    def apply(self, fun, *args, **kwargs) -> "CExpression":
        return self._create(fun(self._expr, *args, **kwargs), {})

    @property
    def T(self):
        return self._create(self._expr.T, {})

    @abc.abstractmethod
    def _elementwise_mul(self, other: Any) -> Any:
        pass

    @_delegate
    def multiply(self, other: Any) -> "CExpression":
        return self._elementwise_mul(other)

    @abc.abstractmethod
    def _norm(self, p: int = 2) -> Any:
        pass

    @_delegate
    def norm(self, p: int = 2) -> "CExpression":
        return self._norm(p=p)

    @abc.abstractmethod
    def _sum(self, axis: Optional[int] = None) -> Any:
        pass

    @_delegate
    def sum(self, axis: Optional[int] = None) -> "CExpression":
        return self._sum(axis=axis)

    @abc.abstractmethod
    def _max(self, axis: Optional[int] = None) -> Any:
        pass

    @_delegate
    def max(self, axis: Optional[int] = None) -> "CExpression":
        return self._max(axis=axis)

    @_delegate
    def __getitem__(self, item) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __abs__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __pow__(self, power: float) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __rpow__(self, base: float) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __add__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __radd__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __sub__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __rsub__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __mul__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __matmul__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __truediv__(self, other: "CExpression") -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __div__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __rdiv__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __rtruediv__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __rmul__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __rmatmul__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __neg__(self) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __eq__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __le__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __lt__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __ge__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    @_delegate
    def __gt__(self, other: Any) -> "CExpression":  # type: ignore
        pass

    def __str__(self) -> str:
        return self._expr.__str__()

    def __repr__(self) -> str:
        return self._expr.__repr__()

    # TODO: add functions along axis: https://www.cvxpy.org/tutorial/functions/index.html


class CSymbol(CExpression):
    def __init__(
        self,
        expr: Any,
        name: str,
        lb: Optional[Union[float, np.ndarray]] = None,
        ub: Optional[Union[float, np.ndarray]] = None,
        vartype: VarType = VarType.CONTINUOUS,
    ) -> None:
        super().__init__(expr)
        lb_r: Optional[np.ndarray] = None
        ub_r: Optional[np.ndarray] = None
        self._provided_lb = lb
        self._provided_ub = ub

        if lb is None:
            if vartype == VarType.CONTINUOUS:
                lb_r = np.full(expr.shape, -np.inf)
                self._lb_ignore = True
            elif vartype == VarType.INTEGER:
                lb_r = np.full(expr.shape, np.iinfo(int).min)
                self._lb_ignore = True
            elif vartype == VarType.BINARY:
                lb_r = np.zeros(expr.shape)
        if ub is None:
            if vartype == VarType.CONTINUOUS:
                ub_r = np.full(expr.shape, np.inf)
                self._ub_ignore = True
            elif vartype == VarType.INTEGER:
                ub_r = np.full(expr.shape, np.iinfo(int).max)
                self._ub_ignore = True
            elif vartype == VarType.BINARY:
                ub_r = np.ones(expr.shape)
        if isinstance(lb, np.ndarray):
            # TODO: change the way we handle this
            if not _eq_shape(lb, expr):
                raise ValueError(
                    f"Shape of lb is {lb.shape}, whereas symbol has a shape of {expr.shape}"
                )
            lb_r = lb
        elif isinstance(lb, numbers.Number):
            lb_r = np.full(expr.shape, lb)
        else:
            if lb is not None:
                raise ValueError(
                    f"lb has an invalid type ({type(lb)}). It must be a number or numpy array"
                )
        if isinstance(ub, np.ndarray):
            if not _eq_shape(ub, expr):
                raise ValueError(
                    f"Shape of ub is {ub.shape}, whereas symbol has a shape of {expr.shape}"
                )
            ub_r = ub
        elif isinstance(ub, numbers.Number):
            ub_r = np.full(expr.shape, ub)
        else:
            if ub is not None:
                raise ValueError(
                    f"ub has an invalid type ({type(ub)}). It must be a number or numpy array"
                )
        self._lb = lb_r
        self._ub = ub_r
        self._name = name
        self._vartype = vartype

    def is_symbol(self) -> bool:
        return True

    @property
    def lb(self) -> np.ndarray:
        return self._lb

    @property
    def ub(self) -> np.ndarray:
        return self._ub

    @property
    def name(self) -> str:
        return self._name


class ProblemDef:
    def __init__(
        self,
        backend: Optional["Backend"] = None,
        constraints: Optional[List[CExpression]] = None,
        objectives: Optional[List[CExpression]] = None,
        expressions: Optional[Dict[str, CExpression]] = None,
        weights: Optional[List[float]] = None,
        direction: Direction = Direction.MIN,
        graph: Optional[BaseGraph] = None,  # TODO: decouple this from backend
    ) -> None:
        if objectives is None:
            objectives = []
        if weights is None:
            weights = [1.0] * len(objectives)
        else:
            if len(weights) != len(objectives):
                raise ValueError(
                    f"The number of weights ({len(weights)}) should match the number of objectives ({len(objectives)})"
                )
        self._backend = backend
        self._constraints = constraints if constraints else []
        self._objectives = objectives if objectives else []
        self._weights = weights if weights else []
        self._direction = direction
        # Registered expressions with names
        self._expressions = dict()
        if expressions is not None:
            self._expressions.update(expressions)
        # Create a new class of problems on top of a graph
        # where edges/nodes have associated optimization variables
        # TODO: check which use cases are using _graph
        self._graph = graph

    @property
    def symbols(self) -> Dict[str, CSymbol]:
        # show deprecated warning:
        # warnings.warn("Use ProblemDef.expressions instead.", DeprecationWarning)
        return {
            s.name: s for s in Backend.get_symbols(self._constraints + self._objectives)
        }

    @property
    def expressions(self) -> Attributes:
        attr = Attributes()
        sym = self.symbols
        attr.update(self.symbols)
        attr.update({k: v for k, v in self._expressions.items() if k not in sym})
        return attr

    @property
    def expr(self) -> Attributes:
        return self.expressions

    def get_symbol(self, name) -> CSymbol:
        return self.symbols[name]

    def get_symbols(self, *args) -> List[CSymbol]:
        return [self.get_symbol(n) for n in args]

    @property
    def constraints(self) -> List[CExpression]:
        return self._constraints

    @property
    def objectives(self) -> List[CExpression]:
        return self._objectives

    @property
    def weights(self) -> List[float]:
        return self._weights

    @property
    def direction(self) -> Direction:
        return self._direction

    def copy(self) -> "ProblemDef":
        # TODO: Fix copy! add tests
        return ProblemDef(
            self._backend,
            self._constraints,
            self._objectives,
            self._expressions,
            self._weights,
            self._direction,
            self._graph,
        )

    def _add(self, other: Any, inplace: bool = False):
        if isinstance(other, ProblemDef):
            return self.merge(other, inplace=inplace)
        elif isinstance(other, CSymbol):
            LOGGER.warn(f"Ignoring request to add symbol {other} (not required)")
            return self
        elif isinstance(other, CExpression) and not isinstance(other, CSymbol):
            # TODO: Check if the expression is a constraint!
            return self.add_constraints([other], inplace=inplace)
        elif isinstance(other, Iterable):
            o = self
            if not inplace:
                o = self.copy()
            for e in other:
                if isinstance(e, CSymbol):
                    # o.add_symbols([e], inplace=True)
                    pass
                elif isinstance(e, CExpression) and not isinstance(e, CSymbol):
                    o.add_constraints([e], inplace=True)
                else:
                    raise ValueError(f"Unsupported type {type(e)}")
            return o
        else:
            raise ValueError(f"Cannot add {type(other)} to ProblemDef")

    def __add__(self, other: Any) -> "ProblemDef":
        return self._add(other, inplace=False)

    def __iadd__(self, other: Any) -> "ProblemDef":
        return self._add(other, inplace=True)

    def solve(
        self,
        solver: Optional[Union[str, Solver]] = None,
        max_seconds: int = None,
        warm_start: bool = False,
        verbosity: int = 0,
        **options,
    ) -> Any:
        if self._backend is None:
            raise ValueError("No backend assigned.")
        return self._backend.solve(
            self,
            solver=solver,
            max_seconds=max_seconds,
            warm_start=warm_start,
            verbosity=verbosity,
            **options,
        )

    def merge(self, other: "ProblemDef", inplace=False) -> "ProblemDef":
        # TODO: If the other is empty (or instance of grammar?) build the problem before merging
        if isinstance(other, ProblemDef) and hasattr(
            other, "_build_problem"
        ):  # TODO Change by isinstance
            f = getattr(other, "_build_problem")
            other = f(self)
        b = self._backend if not None else other._backend
        if not b:
            raise ValueError("Problems have no backend associated.")
        if self._backend and other._backend and (self._backend != other._backend):
            raise ValueError(
                "The two problems have different instantiations of the backend."
            )
        if inplace:
            self.add_constraints(other._constraints, inplace=True)
            self.add_objectives(other._objectives, other._weights, inplace=True)
            self.add_expressions(other._expressions, inplace=True)
            return self
        c = self._constraints + other._constraints
        e = self._expressions.copy()
        e.update(other._expressions)
        w = self._weights + other._weights
        o = self._objectives + other._objectives
        # TODO: generalize for any subclass of ProblemDef
        return self.__class__(b, c, o, e, w)

    def register(
        self, name: str, expr: CExpression, inplace: bool = True
    ) -> "ProblemDef":
        if name in self._expressions:
            raise ValueError(f"Expression with name {name} already exists")
        if name in self.symbols:
            raise ValueError(f"Symbol with name {name} already exists")
        return self.add_expressions({name: expr}, inplace=inplace)

    def add_constraints(
        self,
        constraints: Union[CExpression, List[CExpression]],
        inplace: bool = True,
    ) -> "ProblemDef":
        # TODO: Auto-load symbols from expressions
        if not isinstance(constraints, list):
            constraints = [constraints]
        for c in constraints:
            if isinstance(c, CSymbol):
                raise ValueError(f"The variable {c.name} was added as a constraint")
        if inplace:
            self._constraints.extend(constraints)
            return self
        # TODO: generalize for subclasses of ProblemDef?
        return ProblemDef(
            self._backend,
            self._constraints + constraints,
            self._objectives,
            self._expressions,
            self._weights,
        )

    def add_objectives(
        self,
        objectives: Union[CExpression, List[CExpression]],
        weights: Union[float, List[float]] = 1.0,
        inplace: bool = True,
    ) -> "ProblemDef":
        if not isinstance(objectives, list):
            objectives = [objectives]
        if not isinstance(weights, list):
            weights = [weights] * len(objectives)
        if len(weights) != len(objectives):
            raise ValueError("Number of weights must match number of objectives")
        if inplace:
            self._objectives.extend(objectives)
            self._weights.extend(weights)
            return self
        return ProblemDef(
            self._backend,
            self._constraints,
            self._objectives + objectives,
            self._expressions,
            self._weights + weights,
        )

    def add_expressions(
        self,
        expressions: Dict[str, CExpression],
        inplace: bool = True,
    ) -> "ProblemDef":
        if inplace:
            self._expressions.update(expressions)
            return self
        e = dict()
        e.update(self._expressions)
        e.update(expressions)
        return ProblemDef(
            self._backend,
            self._constraints,
            self._objectives,
            e,
            self._weights,
        )


class ProblemBuilder(ProblemDef):
    def __init__(self) -> None:
        super().__init__(None, None, None, None, None, Direction.MIN)

    def _build_problem(self, other: ProblemDef) -> ProblemDef:
        raise NotImplementedError()

    def merge(self, other: ProblemDef, inplace=False) -> ProblemDef:
        return other.merge(self._build_problem(other), inplace)


class Backend(abc.ABC):
    def __init__(
        self,
        default_solver: Optional[str] = None,
        sparse_class: Callable = _get_matrix_builder(),
    ) -> None:
        self._default_solver = default_solver
        self._sparse = sparse_class

    def is_available(self) -> bool:
        try:
            self._load()
            return True
        except Exception as e:
            LOGGER.debug(str(e))
            return False

    def version(self) -> str:
        return self._load().__version__

    @staticmethod
    def get_symbols(expressions: Iterable[CExpression]) -> Set[CSymbol]:
        symbols: Set[CSymbol] = set()
        for e in expressions:
            s: Set[CSymbol] = getattr(e, "_proxy_symbols", {})  # type: ignore
            if isinstance(e, CSymbol):
                s.add(e)
            symbols.update(s)
        return symbols

    @abc.abstractmethod
    def _load(self) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def available_solvers(self) -> List[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def Variable(
        self,
        name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        lb: Optional[Union[float, np.ndarray]] = None,
        ub: Optional[Union[float, np.ndarray]] = None,
        vartype: VarType = VarType.CONTINUOUS,
    ) -> CSymbol:
        raise NotImplementedError()

    def Problem(
        self,
        constraints: Optional[Union[CExpression, List[CExpression]]] = None,
        objectives: Optional[Union[CExpression, List[CExpression]]] = None,
        expressions: Optional[Dict[str, CExpression]] = None,
        weights: Optional[Union[float, List[float]]] = None,
        direction: Direction = Direction.MIN,
    ) -> ProblemDef:
        if isinstance(constraints, CExpression):
            constraints = [constraints]
        if constraints is None:
            constraints = []
        if isinstance(objectives, CExpression):
            objectives = [objectives]
        if isinstance(weights, float):
            weights = [weights]
        elif isinstance(weights, numbers.Number):
            weights = [float(weights)]
        return ProblemDef(
            self, constraints, objectives, expressions, weights, direction
        )

    @abc.abstractmethod
    def build(self, p: ProblemDef) -> Any:
        raise NotImplementedError()

    def solve(
        self,
        p: ProblemDef,
        solver: Optional[Union[str, Solver]] = None,
        max_seconds: Optional[int] = None,
        warm_start: bool = False,
        verbosity: int = 0,
        **options,
    ):
        if solver is None:
            if self._default_solver is None:
                from corneto.backend import DEFAULT_SOLVER

                solver = DEFAULT_SOLVER
            else:
                solver = self._default_solver

        o: Optional[CExpression]
        if p.objectives is not None and len(p.objectives) > 1:
            if len(p.weights) != len(p.objectives):
                raise ValueError("Number of weights must match number of objectives")
            # auto-convert to a weighted sum
            # type: ignore
            o = sum(
                p.weights[i] * p.objectives[i] if p.weights[i] != 0.0 else 0.0  # type: ignore
                for i in range(len(p.objectives))
            )
        else:
            o = (
                p.weights[0] * p.objectives[0]
                if p.objectives and p.weights[0] != 0
                else None
            )
        return self._solve(
            p,
            objective=o,
            solver=solver,
            max_seconds=max_seconds,
            warm_start=warm_start,
            verbosity=verbosity,
            **options,
        )

    @abc.abstractmethod
    def _solve(
        self,
        p: ProblemDef,
        objective: Optional[CExpression] = None,
        solver: Optional[Union[str, Solver]] = None,
        max_seconds: Optional[int] = None,
        warm_start: bool = False,
        verbosity: int = 0,
        **options,
    ):
        raise NotImplementedError()

    def Constant(self) -> CSymbol:
        raise NotImplementedError()

    def Parameter(self) -> CSymbol:
        raise NotImplementedError()

    def Flow(
        self,
        g: BaseGraph,
        lb: Optional[Union[float, List, np.ndarray]] = 0,
        ub: Optional[Union[float, List, np.ndarray]] = DEFAULT_UB,
        n_flows: int = 1,
        values: bool = False,
        shared_bounds: bool = False,
        varname: str = VAR_FLOW,
        create_nonzero_indicators: bool = False,
        alias_flow: str = EXPR_NAME_FLOW,
        alias_flow_ipos: str = EXPR_NAME_FLOW_IPOS,
        alias_flow_ineg: str = EXPR_NAME_FLOW_INEG,
        alias_nonzero_flow: str = EXPR_NAME_FLOW_NZI,
        indicator_tolerance: float = 1e-4,
    ) -> ProblemDef:
        shape: Tuple = (g.num_edges,)
        if isinstance(lb, list):
            lb = np.array(lb)
        if isinstance(ub, list):
            ub = np.array(ub)
        if n_flows > 1:
            shape = (g.num_edges, n_flows)
            # If lb/ub are vectors, duplicate for each flow
            if isinstance(lb, (int, float)):
                lb = np.ones(shape) * lb
            if isinstance(ub, (int, float)):
                ub = np.ones(shape) * ub
            if isinstance(lb, np.ndarray) and len(lb.shape) == 1:
                lb = np.tile(lb, (n_flows, 1)).T
            if isinstance(ub, np.ndarray) and len(ub.shape) == 1:
                ub = np.tile(ub, (n_flows, 1)).T
        F = self.Variable(varname, shape, lb, ub)
        A = self._sparse(g.get_vertex_incidence_matrix_as_lists(values=values))
        P = self.Problem(A @ F == 0)
        if shared_bounds and n_flows > 1:
            # check num dims of lb
            if len(shape) > 1 and shape[1] > 1 and not _identical_columns(lb):
                raise ValueError(
                    "shared_bounds=True cannot be used when lower bounds are not identical across flows"
                )
            if len(shape) > 1 and shape[1] > 1 and not _identical_columns(ub):
                raise ValueError(
                    "shared_bounds=True cannot be used when upper bounds are not identical across flows"
                )
            S = F.sum(axis=1)
            P += S <= ub[:, 0]
            P += S >= lb[:, 0]
        if create_nonzero_indicators:
            P += NonZeroIndicator(tolerance=indicator_tolerance)
            Ip = P.get_symbol(varname + "_ipos")
            In = P.get_symbol(varname + "_ineg")
            P.register(alias_flow_ipos, Ip)
            P.register(alias_flow_ineg, In)
            P.register(alias_nonzero_flow, Ip + In)
        P.register(alias_flow, F)
        return P

    def Acyclic(
        self,
        g: BaseGraph,
        P: ProblemDef,
        indicator_positive_var_name: str = None,
        indicator_negative_var_name: str = None,
        acyclic_var_name: str = VAR_DAG,
        max_parents: Optional[Union[int, Dict[Any, int]]] = None,
        vertex_lb_dist: Optional[np.ndarray] = None,
    ) -> ProblemDef:
        """Create acyclicity constraint.

        This function creates acyclicity constraints. The acyclic constraints
        ensure that the selected edges are acyclic, i.e. there are no cycles in the graph on the given property.
        Acyclicity can be applied e.g. over the flow constrains or over the signal property.

        Parameters:
        ----------
        g : BaseGraph
            The graph that defines the problem.
        P : ProblemDef
            The problem definition.
        indicator_positive_var_name : str
            The name of the indicator variable, i.e. which edge are selected. By default EXPR_NAME_FLOW_IPOS.
        indicator_negative_var_name : str, optional
            The name of the indicator variable for negative flows, by default None.
            In case a negative flow appears, the source and target nodes of the edge are reversed.
            For example, A->B with positive flow implies order(B) > order(A),
            with negative flow it implies order(A) > order(B).
        acyclic_var_name : str, optional
            The name of the acyclic variable, by default VAR_DAG.
        max_parents : Optional[Union[int, Dict[Any, int]]], optional
            The maximum number of parents per node. If an integer is provided, the maximum number
            of parents is the same for all nodes. If a dictionary is provided, the maximum number
            of parents can be different for each node. By default None.
        vertex_lb_dist : Optional[np.ndarray], optional
            The lower bound distribution of the vertices. By default None.

        Returns:
        -------
        ProblemDef
            The problem definition with acyclic constraints.

        Raises:
        ------
        NotImplementedError
            If hyperedges are used.
        """
        for s, t in g.E:
            if len(s) > 1 or len(t) > 1:
                raise NotImplementedError("Hyperedges not supported")
        if isinstance(max_parents, int):
            max_parents = {v: max_parents for v in g.vertices}
        Ip = In = None
        if (
            indicator_positive_var_name is not None
            and indicator_negative_var_name is not None
        ):
            Ip = P.expressions[indicator_positive_var_name]
            In = P.expressions[indicator_negative_var_name]
            I = Ip + In
        elif indicator_positive_var_name is not None:
            Ip = P.expressions[indicator_positive_var_name]
            I = Ip
        elif indicator_negative_var_name is not None:
            In = P.expressions[indicator_negative_var_name]
            I = In
        else:
            raise ValueError("At least one indicator variable name is required")

        # Limit the number of parents per node, if requested
        if max_parents is not None:
            # Get indexes of edges vi->vj for all vi
            for v, max in max_parents.items():
                edges_idx = [i for i, _ in g.in_edges(v)]
                if len(edges_idx) > 0:
                    # Sum selected parent edges
                    P += np.ones((len(edges_idx),)) @ I[edges_idx] <= max
        # detect the number of DAG layers to add
        if len(I.shape) == 1:
            n_order = 1
        else:
            n_order = I.shape[1]

        # Create a DAG layer num for each vertex
        L = self.Variable(
            acyclic_var_name, (g.num_vertices, n_order), 0, g.num_vertices - 1
        )
        vix = {v: i for i, v in enumerate(g.vertices)}
        for i_order in range(n_order):
            if n_order == 1:
                Ip_i_order = Ip
                In_i_order = In
            else:
                if Ip is not None:
                    Ip_i_order = Ip[:, i_order]
                if In is not None:
                    In_i_order = In[:, i_order]

            # These constraints are not compatible with hyperedges
            if Ip is not None:
                # Get edges s->t that can have a positive flow
                # check if Ip has ub field
                if hasattr(Ip, "ub"):
                    e_pos = [(i, g.get_edge(i)) for i in np.flatnonzero(Ip.ub > 0)]
                    e_ix = np.array(
                        [i for i, (s, t) in e_pos if len(s) > 0 and len(t) > 0]
                    )
                else:
                    e_ix = np.array(
                        [i for i, (s, t) in enumerate(g.E) if len(s) > 0 and len(t) > 0]
                    )
                edges = [g.get_edge(i) for i in e_ix]
                # Get the index of the source / target vertices of the edge
                s_idx = np.array([vix[list(s)[0]] for (s, _) in edges])
                t_idx = np.array([vix[list(t)[0]] for (_, t) in edges])
                # The layer position in a DAG of the target vertex of the edge
                # has to be greater than the source vertex, otherwise Ip (pos flow) has to be 0
                if len(e_ix) > 0:
                    P += L[t_idx, i_order] - L[s_idx, i_order] >= Ip_i_order[e_ix] + (
                        1 - g.num_vertices
                    ) * (1 - Ip_i_order[e_ix])
                    P += L[t_idx, i_order] - L[s_idx, i_order] <= g.num_vertices - 1
            if In is not None:
                # NOTE: Negative flows eq. to reversed directed edge
                # Get edges s->t that can have a positive flow
                if hasattr(In, "lb"):
                    e_neg = [(i, g.get_edge(i)) for i in np.flatnonzero(In.lb < 0)]
                    e_ix = np.array(
                        [i for i, (s, t) in e_neg if len(s) > 0 and len(t) > 0]
                    )
                else:
                    e_ix = np.array(
                        [i for i, (s, t) in enumerate(g.E) if len(s) > 0 and len(t) > 0]
                    )
                edges = [g.get_edge(i) for i in e_ix]
                # Get the index of the source / target vertices of the edge
                s_idx = np.array([vix[list(s)[0]] for (s, _) in edges])
                t_idx = np.array([vix[list(t)[0]] for (_, t) in edges])
                if len(e_ix) > 0:
                    P += L[s_idx, i_order] - L[t_idx, i_order] >= In_i_order[e_ix] + (
                        1 - g.num_vertices
                    ) * (1 - In_i_order[e_ix])
                    P += L[s_idx, i_order] - L[t_idx, i_order] <= g.num_vertices - 1
            # TODO: Raise error if hypergraph
        return P

    def AcyclicFlow(
        self,
        g: BaseGraph,
        lb: Union[float, np.ndarray] = 0,
        ub: Union[float, np.ndarray] = DEFAULT_UB,
        values: bool = False,
        max_parents: Optional[Union[int, Dict[Any, int]]] = None,
        vertex_lb_dist: Optional[np.ndarray] = None,
        varname: str = VAR_FLOW,
        alias_flow: str = EXPR_NAME_FLOW,
        alias_flow_ipos: str = EXPR_NAME_FLOW_IPOS,
        alias_flow_ineg: str = EXPR_NAME_FLOW_INEG,
        alias_nonzero_flow: str = EXPR_NAME_FLOW_NZI,
        indicator_tolerance: float = 1e-4,
    ) -> ProblemDef:
        if not varname:
            varname = VAR_FLOW
        if not isinstance(lb, np.ndarray):
            lb = np.array([lb] * g.num_edges)
        if not isinstance(ub, np.ndarray):
            ub = np.array([ub] * g.num_edges)
        for s, t in g.E:
            if len(s) > 1 or len(t) > 1:
                raise NotImplementedError("Hyperedges not supported")
        if isinstance(max_parents, int):
            max_parents = {v: max_parents for v in g.vertices}
        P = self.Flow(
            g,
            lb,
            ub,
            values=values,
            varname=varname,
            alias_flow=alias_flow,
            alias_flow_ipos=alias_flow_ipos,
            alias_flow_ineg=alias_flow_ineg,
            alias_nonzero_flow=alias_nonzero_flow,
            indicator_tolerance=indicator_tolerance,
            create_nonzero_indicators=True,
        )
        # TODO: recover easily the created indicators!
        # TODO: Optionally provide a lower bound of pos for each vertex
        Ip = P.get_symbol(varname + "_ipos") if any(ub > 0) else None
        In = P.get_symbol(varname + "_ineg") if any(lb < 0) else None
        if Ip is not None and In is not None:
            I = Ip + In
        elif Ip is not None:
            I = Ip
        elif In is not None:
            I = In
        else:
            raise ValueError()
        # Limit the number of parents per node, if requested
        if max_parents is not None:
            # Get indexes of edges vi->vj for all vi
            for v, max in max_parents.items():
                edges_idx = [i for i, _ in g.in_edges(v)]
                if len(edges_idx) > 0:
                    # Sum selected parent edges
                    P += np.ones((len(edges_idx),)) @ I[edges_idx] <= max
        # Create a DAG layer num for each vertex
        L = self.Variable("_dag_layer_pos", (g.num_vertices,), 0, g.num_vertices - 1)
        vix = {v: i for i, v in enumerate(g.vertices)}
        # These constraints are not compatible with hyperedges
        if np.any(ub > 0):
            # Get edges s->t that can have a positive flow
            e_pos = [(i, g.get_edge(i)) for i in np.flatnonzero(ub > 0)]
            e_ix = np.array([i for i, (s, t) in e_pos if len(s) > 0 and len(t) > 0])
            edges = [g.get_edge(i) for i in e_ix]
            # Get the index of the source / target vertices of the edge
            s_idx = np.array([vix[list(s)[0]] for (s, _) in edges])
            t_idx = np.array([vix[list(t)[0]] for (_, t) in edges])
            # The layer position in a DAG of the target vertex of the edge
            # has to be greater than the source vertex, otherwise Ip (pos flow) has to be 0
            if len(e_ix) > 0:
                P += L[t_idx] - L[s_idx] >= Ip[e_ix] + (1 - g.num_vertices) * (
                    1 - Ip[e_ix]
                )
                P += L[t_idx] - L[s_idx] <= g.num_vertices - 1
        if np.any(lb < 0):
            # NOTE: Negative flows eq. to reversed directed edge
            # Get edges s->t that can have a positive flow
            e_neg = [(i, g.get_edge(i)) for i in np.flatnonzero(lb < 0)]
            # Check if vertex not empty! print(e_neg)
            e_ix = np.array([i for i, (s, t) in e_neg if len(s) > 0 and len(t) > 0])
            edges = [g.get_edge(i) for i in e_ix]
            # Get the index of the source / target vertices of the edge
            s_idx = np.array([vix[list(s)[0]] for (s, _) in edges])
            t_idx = np.array([vix[list(t)[0]] for (_, t) in edges])
            if len(e_ix) > 0:
                P += L[s_idx] - L[t_idx] >= In[e_ix] + (1 - g.num_vertices) * (
                    1 - In[e_ix]
                )
                P += L[s_idx] - L[t_idx] <= g.num_vertices - 1
        # TODO: Raise error if hypergraph
        return P

    def Indicator(
        self,
        V: CSymbol,
        indexes: Optional[Union[Tuple, List, np.ndarray]] = None,
        suffix: str = "_i",
        name: Optional[str] = None,
    ) -> ProblemDef:
        # If I = 0 => V = 0, if I = 1, LB <= V <= UB (including 0)
        c = []
        S = V
        ub = V.ub
        lb = V.lb
        if indexes:
            S = V[indexes]
            ub = V.ub[indexes]
            lb = V.lb[indexes]
        if name is None:
            name = V.name + suffix
        # TODO: Add option to create shared indicators for n_flows > 1, so if
        # I_i = 0 => V_1i, ..., V_ni = 0, if I_i = 1, LB <= V_1i, ..., V_ni <= UB
        I = self.Variable(name, S.shape, 0, 1, vartype=VarType.BINARY)
        blocked = np.isclose(ub, 0) & np.isclose(lb, 0)
        if np.sum(blocked) > 0:
            # indexing compatible with CVXPY and PICOS
            idx = np.where(blocked)[0]
            c += [I[idx] == 0, S[idx] == 0]
        # Add constraint: lb * I <= V <= ub * I
        if V._provided_lb is None or V._provided_ub is None:
            raise ValueError(
                f"The continuous variable {V.name} is unbounded, indicators cannot be created."
            )
        c += [S >= I.multiply(lb), S <= I.multiply(ub)]
        return self.Problem(c)

    def NonZeroIndicator(
        self,
        V: CSymbol,
        indexes: Optional[Union[Tuple, List, np.ndarray]] = None,
        suffix_pos: str = "_ipos",
        suffix_neg: str = "_ineg",
        tolerance: float = 1e-3,
    ) -> ProblemDef:
        # NOTE: This can add integer feasibility issues to the problem
        c = []
        S = V
        lb = V.lb
        ub = V.ub
        if V._provided_lb is None or V._provided_ub is None:
            raise ValueError(
                f"The continuous variable {V.name} is unbounded, indicators cannot be created."
            )
        if indexes is not None and len(indexes) > 0:
            S = V[indexes]
            lb = V.lb[indexes]
            ub = V.ub[indexes]
        I_pos = self.Variable(
            V.name + suffix_pos, S.shape, 0, 1, vartype=VarType.BINARY
        )
        I_neg = self.Variable(
            V.name + suffix_neg, S.shape, 0, 1, vartype=VarType.BINARY
        )
        I = I_pos + I_neg
        c += [I <= 1]  # mutually exclusive
        if np.sum(ub <= 0) > 0:
            c += [I_pos[np.where(ub <= 0)[0]] == 0]
        if np.sum(lb >= 0) > 0:
            c += [I_neg[np.where(lb >= 0)[0]] == 0]
        # I_neg and I_pos mutually exclusive (I_pos + I_neg <= 1)
        # If I_pos = 1 and I_neg = 0: V >= tol AND V <= ub
        # If I_pos = 0 and I_neg = 1: V >= lb AND V <= -tol
        # If I_pos = 0 and I_neg = 0: V >= 0 AND V <= 0
        c += [
            S >= I_neg.multiply(lb) + I_pos * tolerance,
            S <= I_pos.multiply(ub) - I_neg * tolerance,
        ]
        P = self.Problem(c)
        P.register(EXPR_NAME_FLOW_NZI, I)
        return self.Problem(c)

    # TODO: Remove function
    def Indicators(
        self,
        V: CSymbol,
        tolerance=1e-3,
        positive=True,
        negative=True,
        suffix_pos="_ipos",
        suffix_neg="_ineg",
    ) -> ProblemDef:
        # Get upper/lower bounds for flow variables
        constraints = []
        if not (positive or negative):
            raise ValueError("At least one of positive or negative must be True.")
        if positive:
            I_pos = self.Variable(
                V.name + suffix_pos, V.shape, 0, 1, vartype=VarType.BINARY
            )
            if np.sum(V.ub <= 0) > 0:
                constraints.append(I_pos[np.where(V.ub <= 0)[0]] == 0)
        if negative:
            I_neg = self.Variable(
                V.name + suffix_neg, V.shape, 0, 1, vartype=VarType.BINARY
            )
            if np.sum(V.lb >= 0) > 0:
                constraints.append(I_neg[np.where(V.lb >= 0)[0]] == 0)
        if positive and negative:
            constraints.append(I_pos + I_neg <= 1)

        # lower bound constraints: F >= F_lb * I_neg + eps * I_pos
        # TODO: Use better constraints to avoid precision errors
        # by multiplying tolerance * indicator
        I_LBN = I_neg.multiply(V.lb) if negative else V.lb
        I_LBP = tolerance * I_pos if positive else 0
        LB = I_LBN + I_LBP
        constraints.append(V >= LB)
        # upper bound constraints: F <= F_ub * I_pos - eps * I_neg
        I_UBN = I_pos.multiply(V.ub) if positive else V.ub
        I_UBP = tolerance * I_neg if negative else 0
        UB = I_UBN - I_UBP
        constraints.append(V <= UB)
        return self.Problem(constraints)

    def Xor(self, x: CExpression, y: CExpression, varname="_xor"):
        # TODO: Generalize for matrices Xor(X,Y)
        if isinstance(x, CSymbol) and x._vartype != VarType.BINARY:
            raise ValueError(f"Variable x has type {x._vartype} instead of BINARY")
        if isinstance(y, CSymbol) and y._vartype != VarType.BINARY:
            raise ValueError(f"Variable x has type {y._vartype} instead of BINARY")
        if x.shape != y.shape:
            raise ValueError(f"Shape of x ({x.shape}) is different from y ({y.shape})")
        # Create a new binary variable to compute xor(x,y)
        xor = self.Variable(varname, x.shape, 0, 1, vartype=VarType.BINARY)
        return self.Problem(
            [xor >= x - y, xor >= y - x, xor <= x + y, xor <= 2 - x - y]
        )

    def linear_or(self, x: CSymbol, axis: Optional[int] = None, varname="_linear_or"):
        # Check if the variable is binary, otherwise throw an error
        if x._vartype != VarType.BINARY:
            raise ValueError(f"Variable x has type {x._vartype} instead of BINARY")
        Z = x.sum(axis=axis)
        Z_norm = Z / x.shape[axis]  # between 0-1
        # Create a new binary variable to compute linearized or
        Or = self.Variable(varname, Z.shape, 0, 1, vartype=VarType.BINARY)
        return self.Problem([Or >= Z_norm, Or <= Z])

    def linear_and(self, x: CSymbol, axis: Optional[int] = None, varname="_linear_and"):
        # Check if the variable is binary, otherwise throw an error
        if x._vartype != VarType.BINARY:
            raise ValueError(f"Variable x has type {x._vartype} instead of BINARY")
        Z = x.sum(axis=axis)
        N = x.shape[axis]
        Z_norm = Z / N
        And = self.Variable(varname, Z.shape, 0, 1, vartype=VarType.BINARY)
        return self.Problem([And <= Z_norm, And >= Z - N + 1])


class NoBackend(Backend):
    def __init__(self) -> None:
        self._error = "No backend available. Please install on of the supported backends, e.g. `pip install cvxpy`."

    def _load(self):
        return None

    def available_solvers(self):
        return []

    def Variable(
        self,
        *args,
        **kwargs,
    ) -> CSymbol:
        raise NotImplementedError(self._error)

    def _solve(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError(self._error)

    def Constant(self) -> CSymbol:
        raise NotImplementedError(self._error)

    def Parameter(self) -> CSymbol:
        raise NotImplementedError(self._error)

    def build(self, p: ProblemDef) -> Any:
        raise NotImplementedError(self._error)


def _find_continuous_var(p):
    # Search for continous vars
    cvars = [k for k, v in p.symbols.items() if v._vartype == VarType.CONTINUOUS]
    if len(cvars) == 0:
        raise ValueError("No available continuous vars for creating indicator vars")
    if len(cvars) == 1:
        LOGGER.debug(f"No variable provided, creating indicators for {cvars[0]}")
        return cvars[0]
    else:
        raise ValueError(
            f"There are {len(cvars)} continous vars, but no var_name is provided."
        )


class Indicator(ProblemBuilder):
    def __init__(
        self,
        name: Optional[str] = None,
        var_name: Optional[str] = None,
        indexes: Optional[Union[Tuple, List, np.ndarray]] = None,
        suffix: str = "_i",
    ) -> None:
        super().__init__()
        self.var_name = var_name
        self._suffix = suffix
        self._indexes = indexes
        self._name = name

    def _build_problem(self, other: ProblemDef):
        if other._backend is None:
            raise ValueError("Cannot combine problems without a main backend")
        if self.var_name is None:
            self.var_name = _find_continuous_var(other)
        return other._backend.Indicator(
            other.get_symbol(self.var_name),
            suffix=self._suffix,
            indexes=self._indexes,
            name=self._name,
        )


class NonZeroIndicator(ProblemBuilder):
    def __init__(
        self,
        var_name: Optional[str] = None,
        indexes: Optional[Union[Tuple, List, np.ndarray]] = None,
        tolerance=1e-4,
        suffix_pos: str = "_ipos",
        suffix_neg: str = "_ineg",
    ) -> None:
        super().__init__()
        self.var_name = var_name
        self._tolerance = tolerance
        self._suffix_pos = suffix_pos
        self._suffix_neg = suffix_neg
        self._indexes = indexes

    def _build_problem(self, other: ProblemDef):
        if other._backend is None:
            raise ValueError("Cannot combine empty grammars")
        if self.var_name is None:
            self.var_name = _find_continuous_var(other)
        return other._backend.NonZeroIndicator(
            other.get_symbol(self.var_name),
            suffix_pos=self._suffix_pos,
            suffix_neg=self._suffix_neg,
            tolerance=self._tolerance,
            indexes=self._indexes,
        )


class Indicators(ProblemBuilder):
    def __init__(
        self,
        var_name: Optional[str] = None,
        tolerance: float = 1e-3,
        positive: bool = True,
        negative: bool = False,
    ) -> None:
        # Probably there is some missing component which is not a problemdef
        super().__init__()
        self.var_name = var_name
        self._tol = tolerance
        self._pos = positive
        self._neg = negative

    def _build_problem(self, other: ProblemDef):
        if other._backend is None:
            raise ValueError("Cannot combine empty grammars")
        if self.var_name is None:
            self.var_name = _find_continuous_var(other)
        return other._backend.Indicators(
            other.get_symbol(self.var_name),
            tolerance=self._tol,
            positive=self._pos,
            negative=self._neg,
        )


class HammingLoss(ProblemBuilder):
    def __init__(
        self,
        reference: np.ndarray,
        y: Union[str, CExpression],
        penalty: float = 1.0,
    ) -> None:
        super().__init__()
        self.ref = reference
        self.y = y
        self.penalty = penalty

    def _build_problem(self, other: ProblemDef) -> ProblemDef:
        x = abs(self.ref)
        y = other.get_symbol(self.y) if isinstance(self.y, str) else self.y
        idx_one = np.where(x == 1)[0]
        idx_zero = np.where(x == 0)[0]
        P = ProblemDef()
        diff_zeros = y[idx_zero] - x[idx_zero]
        diff_ones = x[idx_one] - y[idx_one]
        hamming_dist = (
            np.ones(diff_zeros.shape) @ diff_zeros
            + np.ones(diff_ones.shape) @ diff_ones
        )
        P.add_objectives(hamming_dist, weights=self.penalty, inplace=True)  # type: ignore
        return P
