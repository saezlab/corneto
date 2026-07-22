from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from corneto import DEFAULT_BACKEND
from corneto.backend._base import Backend, ProblemDef
from corneto.data import Data
from corneto.graph import BaseGraph


class Method(ABC):
    """Base class for optimization methods without a flow formulation.

    This class provides data preprocessing, problem construction, and structured
    sparsity regularization.

    Args:
        lambda_reg: Regularization strength. Defaults to 0.0.
        reg_varname: Name of the variable to regularize. Required if lambda_reg > 0.
        reg_varname_suffix: Suffix for the regularization variable name.
            Defaults to "_OR".
        backend: Optimization backend to use. Defaults to DEFAULT_BACKEND.
    """

    __show_info_on_import__ = False

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
        self.lambda_reg_param = backend.Parameter(name="lambda_reg_param", value=lambda_reg)
        self._reg_varname = reg_varname
        self._reg_varname_suffix = reg_varname_suffix
        self.problem = None
        self.processed_data = None
        self.processed_graph = None
        self.disable_structured_sparsity = disable_structured_sparsity

    def __repr__(self) -> str:
        """Returns a string representation of the method.

        Includes the method name, description, parameters, and formatted
        citations (if any).

        Returns:
            A formatted string representation of the method.
        """
        import inspect

        from corneto.utils._citations import format_references_plaintext

        name = self.name()
        description = self.description()
        citation_keys = self.references()

        repr_str = f"{name or self.__class__.__name__}"

        if description:
            repr_str += f": {description})"

        # Add parameters information
        repr_str += "\n\nParameters:"
        # Get all instance attributes that don't start with _ (except _backend)
        params = {}
        for attr_name, attr_value in self.__dict__.items():
            if attr_name == "_backend":
                params["backend"] = attr_value.__class__.__name__
            elif not attr_name.startswith("_") or attr_name in [
                "_reg_varname",
                "_reg_varname_suffix",
            ]:
                # Skip initially unset problem and processed input attributes.
                if attr_name in ["problem", "processed_data", "processed_graph"] and attr_value is None:
                    continue
                # Skip complex objects that aren't useful for a summary
                if isinstance(attr_value, (list, dict, tuple)) and len(str(attr_value)) > 100:
                    continue
                # Format the parameter name (remove leading underscore if present)
                param_name = attr_name[1:] if attr_name.startswith("_") else attr_name
                params[param_name] = attr_value

        # Get signature to determine default values
        signature = None
        try:
            signature = inspect.signature(self.__class__.__init__)
        except (ValueError, TypeError):
            pass

        if params:
            for name, value in sorted(params.items()):
                # Check if this is a default value
                is_default = False
                if signature and name in signature.parameters:
                    param = signature.parameters[name]
                    if param.default is not param.empty:
                        if param.default == value:
                            is_default = True

                # Format the parameter value
                if isinstance(value, (int, float, bool, str)) or value is None:
                    value_str = str(value)
                elif hasattr(value, "__class__"):
                    value_str = f"<{value.__class__.__name__}>"
                else:
                    value_str = str(value)

                # Add default indicator if applicable
                if is_default:
                    repr_str += f"\n  {name} = {value_str} (default)"
                else:
                    repr_str += f"\n  {name} = {value_str}"
        else:
            repr_str += "\n  No parameters"

        if citation_keys:
            repr_str += "\n\nReferences:"
            repr_str += format_references_plaintext(citation_keys)

        return repr_str

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
    def create_problem(self, graph: BaseGraph, data: Data) -> ProblemDef:
        """Create the optimization problem.

        This method should define variables, constraints, and objectives
        according to the method's formulation.

        Args:
            graph: The preprocessed network graph.
            data: The preprocessed dataset.

        Returns:
            The complete optimization problem.
        """
        pass

    def build_from_data(self, graph: BaseGraph, data: Optional[Data] = None) -> ProblemDef:
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
            ValueError: If lambda_reg > 0 but no regularization variable name
                is provided.
        """
        if data is None:
            data = Data.empty()
        self.processed_graph, self.processed_data = self.preprocess(graph, data)
        self.problem = self.create_problem(self.processed_graph, self.processed_data)

        # Add structured sparsity regularization if needed.
        if not self.disable_structured_sparsity:
            if self._reg_varname is not None:
                reg_var = self.problem.expr[self._reg_varname]
                newvar_name = self._reg_varname + self._reg_varname_suffix
                ax = 0 if len(reg_var.shape) == 1 else 1
                # A 1D vector can be summed directly without a linear OR.
                if len(reg_var.shape) == 1 or reg_var.shape[1] == 1 or reg_var.shape[0] == 1:
                    self.problem.add_objective(
                        reg_var.sum(),
                        weight=self.lambda_reg_param,
                        name=f"regularization_{self._reg_varname}",
                    )
                else:
                    # Structured sparsity regularization
                    self.problem += self._backend.linear_or(reg_var, axis=ax, varname=newvar_name)
                    self.problem.add_objective(
                        self.problem.expr[newvar_name].sum(),
                        weight=self.lambda_reg_param,
                        name=f"regularization_{newvar_name}",
                    )
            else:
                raise ValueError("Parameter lambda_reg > 0 but no regularization variable name provided")
        return self.problem

    def build(self, graph: BaseGraph, data: Optional[Data] = None) -> ProblemDef:
        """Build from a :class:`~corneto.data.Data` object.

        This compatibility entry point is retained for custom ``Method``
        subclasses. Public CORNETO methods provide method-specific ``build``
        signatures and expose this generic path as :meth:`build_from_data`.
        """
        return self.build_from_data(graph, data)

    @staticmethod
    def name() -> str:
        """Returns the name of the method.

        Returns:
            The name of the optimization method.
        """
        return ""

    def description(self) -> str:
        """Returns a description of the method.

        Returns:
            A string describing the optimization method.
        """
        return ""

    @staticmethod
    def references() -> list:
        """Returns citation keys for this method.

        Returns:
            A list of citation keys that can be used to lookup BibTeX entries.
        """
        return []

    @classmethod
    def show_references(cls):
        """Display formatted citations in a Jupyter notebook."""
        from corneto.utils._citations import show_references

        return show_references(cls.references())

    @classmethod
    def show_bibtex(cls):
        """Display raw BibTeX entries in a formatted block for easy copying."""
        from corneto.utils._citations import show_bibtex

        return show_bibtex(cls.references())

    @property
    def backend(self):
        """Return the optimization backend being used."""
        return self._backend


class FlowMethod(Method):
    """Abstract base class for flow-based optimization methods.

    Extends Method with flow bounds and flow-based problem construction.

    Args:
        flow_lower_bound: Lower bound for flow variables. Defaults to DEFAULT_LB.
        flow_upper_bound: Upper bound for flow variables. Defaults to DEFAULT_UB.
        num_flows: Number of parallel flows to use. Defaults to 1.
        shared_flow_bounds: Whether to share bounds across parallel flows.
            Defaults to False.
        lambda_reg: Regularization strength. Defaults to 0.0.
        reg_varname: Name of the variable to regularize. Required if lambda_reg > 0.
        reg_varname_suffix: Suffix for the regularization variable name.
            Defaults to "_OR".
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

        This method should define variables, constraints, and objectives based
        on the flow formulation.

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

        Internally, this gets flow bounds, creates a base flow problem using the
        backend, and extends it by invoking create_flow_based_problem.

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
