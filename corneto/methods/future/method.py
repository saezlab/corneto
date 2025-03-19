from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Dict

from corneto import DEFAULT_BACKEND
from corneto._constants import DEFAULT_LB, DEFAULT_UB
from corneto._graph import BaseGraph
from corneto.backend._base import Backend, ProblemDef
from corneto.data import Data


class Method(ABC):
    """Abstract base class for optimization methods that do not require a flow formulation.

    This class provides common functionalities such as data preprocessing, problem construction,
    and structured sparsity regularization.

    Args:
        lambda_reg: Regularization strength. Defaults to 0.0.
        reg_varname: Name of the variable to regularize. Required if lambda_reg > 0.
        reg_varname_suffix: Suffix for the regularization variable name. Defaults to "_OR".
        backend: Optimization backend to use. Defaults to DEFAULT_BACKEND.
    """

    def __init__(
        self,
        lambda_reg: float = 0.0,
        reg_varname: Optional[str] = None,
        reg_varname_suffix: str = "_OR",
        disable_structured_sparsity: bool = False,
        backend: Optional[Backend] = None,
    ):
        if backend is None:
            backend = DEFAULT_BACKEND
        self._backend = backend
        self.lambda_reg_param = backend.Parameter(
            name="lambda_reg_param", value=lambda_reg
        )
        self._reg_varname = reg_varname
        self._reg_varname_suffix = reg_varname_suffix
        self.problem = None
        self.processed_data = None
        self.processed_graph = None
        self.disable_structured_sparsity = disable_structured_sparsity

    @abstractmethod
    def preprocess(self, graph: BaseGraph, data: Data) -> Tuple[BaseGraph, Data]:
        """Preprocess the input graph and dataset before optimization.

        This method should perform any necessary transformations or validations.

        Args:
            graph: The input network graph.
            data: The dataset containing experimental measurements.

        Returns:
            A tuple containing:
              - The preprocessed graph.
              - The preprocessed dataset.
        """
        pass

    @abstractmethod
    def create_problem(self, graph: BaseGraph, data: Data):
        """Create the optimization problem.

        This method should define variables, constraints, and objectives according to the method's
        formulation.

        Args:
            graph: The preprocessed network graph.
            data: The preprocessed dataset.

        Returns:
            The complete optimization problem.
        """
        pass

    def build(self, graph: BaseGraph, data: Data) -> ProblemDef:
        """Build the complete optimization problem.

        The process involves:
          1. Preprocessing the inputs.
          2. Creating the optimization problem.
          3. Adding structured sparsity regularization if specified.

        Args:
            graph: The input network graph.
            data: The experimental dataset.

        Returns:
            The constructed optimization problem ready to be solved.

        Raises:
            ValueError: If lambda_reg > 0 but no regularization variable name is provided.
        """
        self.processed_graph, self.processed_data = self.preprocess(graph, data)
        self.problem = self.create_problem(self.processed_graph, self.processed_data)

        # Add structured sparsity regularization if needed.
        if not self.disable_structured_sparsity:
            if self._reg_varname is not None:
                reg_var = self.problem.expr[self._reg_varname]
                newvar_name = self._reg_varname + self._reg_varname_suffix
                ax = 0 if len(reg_var.shape) == 1 else 1
                #if len(reg_var.shape) == 1 or (len(reg_var.shape) == 2 and reg_var.shape[1] == 1):
                    # If reg_var is a 1D array, no need to compute or
                #    reg_var = reg_var[:, None]
                self.problem += self._backend.linear_or(
                    reg_var, axis=ax, varname=newvar_name
                )
                self.problem.add_objectives(
                    self.problem.expr[newvar_name].sum(),
                    weights=self.lambda_reg_param,
                )
            else:
                raise ValueError(
                    "Parameter lambda_reg > 0 but no regularization variable name provided"
                )
        return self.problem

    @property
    def backend(self):
        """Return the optimization backend being used."""
        return self._backend


class FlowMethod(Method):
    """Abstract base class for flow-based optimization methods.

    Extends Method with flow-specific functionalities such as setting flow bounds and creating a
    flow-based problem formulation.

    Args:
        flow_lower_bound: Lower bound for flow variables. Defaults to DEFAULT_LB.
        flow_upper_bound: Upper bound for flow variables. Defaults to DEFAULT_UB.
        num_flows: Number of parallel flows to use. Defaults to 1.
        shared_flow_bounds: Whether to share flow bounds across parallel flows. Defaults to False.
        lambda_reg: Regularization strength. Defaults to 0.0.
        reg_varname: Name of the variable to regularize. Required if lambda_reg > 0.
        reg_varname_suffix: Suffix for the regularization variable name. Defaults to "_OR".
        backend: Optimization backend to use. Defaults to DEFAULT_BACKEND.
    """

    def __init__(
        self,
        flow_lower_bound: float = 0,
        flow_upper_bound: float = 1000,
        num_flows: int = 1,
        shared_flow_bounds: bool = False,
        use_flow_coefficients: bool = False,
        lambda_reg: float = 0.0,
        reg_varname: Optional[str] = None,
        reg_varname_suffix: str = "_OR",
        disable_structured_sparsity: bool = False,
        backend: Optional[Backend] = None,
    ):
        super().__init__(
            lambda_reg=lambda_reg,
            reg_varname=reg_varname,
            reg_varname_suffix=reg_varname_suffix,
            disable_structured_sparsity=disable_structured_sparsity,
            backend=backend,
        )
        self._flow_lb = flow_lower_bound
        self._flow_ub = flow_upper_bound
        self._num_flows = num_flows
        self._shared_flow_bounds = shared_flow_bounds
        self._use_flow_coefficients = use_flow_coefficients

    def get_flow_bounds(self, graph: BaseGraph, data: Data) -> Dict[str, Data]:
        """Get the flow bounds and parameters for creating a flow problem.

        This method can be overridden by subclasses to provide custom flow bounds
        based on the graph or data.

        Args:
            graph: The preprocessed network graph.
            data: The preprocessed dataset.

        Returns:
            A dictionary containing flow configuration parameters:
                - 'lb': Lower bounds for flows (float, array, or None)
                - 'ub': Upper bounds for flows (float, array, or None)
                - 'n_flows': Number of flows (int)
                - 'shared_bounds': Whether bounds are shared across flows (bool)
        """
        return {
            "lb": self._flow_lb,
            "ub": self._flow_ub,
            "n_flows": self._num_flows,
            "shared_bounds": self._shared_flow_bounds,
        }

    @abstractmethod
    def create_flow_based_problem(self, flow_problem, graph: BaseGraph, data: Data):
        """Create the optimization problem with flow-based constraints.

        This method should set up the optimization problem by defining variables, constraints, and
        objectives based on the flow formulation.

        Args:
            flow_problem: The base flow problem object provided by the backend.
            graph: The preprocessed network graph.
            data: The preprocessed dataset.

        Returns:
            The complete flow-based optimization problem.
        """
        pass

    def create_problem(self, graph: BaseGraph, data: Data):
        """Create the optimization problem using a flow-based formulation.

        Internally, this method gets flow bounds, creates a base flow problem using the backend,
        and then extends it by invoking create_flow_based_problem.

        Args:
            graph: The preprocessed network graph.
            data: The preprocessed dataset.

        Returns:
            The complete optimization problem with flow-based constraints.
        """
        flow_params = self.get_flow_bounds(graph, data)
        flow_problem = self.backend.Flow(
            graph,
            lb=flow_params["lb"],
            ub=flow_params["ub"],
            n_flows=flow_params["n_flows"],
            values=self._use_flow_coefficients,
            shared_bounds=flow_params["shared_bounds"],
        )
        return self.create_flow_based_problem(flow_problem, graph, data)
