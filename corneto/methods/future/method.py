from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np

from corneto import DEFAULT_BACKEND
from corneto._data import Data
from corneto._graph import BaseGraph
from corneto.backend._base import Backend, ProblemDef


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

        Includes the method name, description, parameters, and formatted citations (if any).

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
                # For problem, processed_data, etc. that could be None initially, skip them
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
                # If we only have a 1D vector, we dont need linear or, we compute the sum:
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

        show_references(cls.references())

    @classmethod
    def show_bibtex(cls):
        """Display raw BibTeX entries in a formatted block for easy copying."""
        from corneto.utils._citations import show_bibtex

        show_bibtex(cls.references())

    @classmethod
    def about(cls):
        """Display information about the method in a Jupyter notebook."""
        cls.show_references()

    @property
    def backend(self):
        """Return the optimization backend being used."""
        return self._backend

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


class GeneralizedMultiSampleMethod(Method):
    """A method that extends single-sample methods to handle multiple samples.

    This class takes a base optimization method designed for a single sample and generalizes it
    to work with multiple samples by creating separate problems for each sample and combining them
    into a larger optimization problem.

    Args:
        base_method: The base optimization method to generalize.
        edge_selection_varname: Variable name (without suffix) that indicates if an edge is selected
            or not in the base method problem. Required for proper structured sparsity regularization
            across samples.
        lambda_reg: Regularization strength. Defaults to 0.0.
        reg_varname: Name of the variable to regularize. Required if lambda_reg > 0.
        reg_varname_suffix: Suffix for the regularization variable name. Defaults to "_OR".
        disable_structured_sparsity: Whether to disable structured sparsity regularization. Defaults to False.
        backend: Optimization backend to use. Defaults to DEFAULT_BACKEND.
    """

    def __init__(
        self,
        base_method: Method,
        edge_selection_varname: Optional[str] = None,
        lambda_reg: float = 0.0,
        reg_varname: Optional[str] = None,
        reg_varname_suffix: str = "_OR",
        disable_structured_sparsity: bool = False,
    ):
        # If structured sparsity is enabled and lambda_reg > 0, we'll use the edge_selection_varname
        # to create a combined regularization variable spanning all samples
        if not disable_structured_sparsity and lambda_reg > 0:
            if edge_selection_varname is None:
                raise ValueError(
                    "edge_selection_varname is required when lambda_reg > 0 and structured sparsity is enabled"
                )
            reg_varname = edge_selection_varname

        super().__init__(
            lambda_reg=lambda_reg,
            reg_varname=reg_varname,
            reg_varname_suffix=reg_varname_suffix,
            disable_structured_sparsity=disable_structured_sparsity,
            backend=base_method._backend,
        )
        self.base_method = base_method
        self.edge_selection_varname = edge_selection_varname
        self._preprocessed_graphs = None
        self._selected_edges = None

    def preprocess(self, graph: BaseGraph, data: Data) -> Tuple[BaseGraph, Data]:
        return graph, data

    def create_problem(self, graph: BaseGraph, data: Data):
        """Create a combined optimization problem from multiple samples.

        This method:
        1. Iterates through all samples in the dataset
        2. Creates a single-sample dataset for each sample
        3. Builds a problem for each sample using the base method
        4. Adds a unique suffix to all variables in this problem
        5. Combines all problems into one large problem
        6. If structured sparsity is enabled, creates an hstack of edge selection variables

        Args:
            graph: The preprocessed network graph.
            data: The preprocessed dataset with multiple samples.

        Returns:
            The combined optimization problem.
        """
        # If there's only one sample, just use the base method directly
        if len(data.samples) <= 1:
            return self.base_method.build(graph, data)

        combined_problem = None
        edge_selection_vars = []

        G = graph.copy()
        for i in range(G.num_edges):
            attr = G.get_attr_edge(i)
            attr["id"] = i

        # Process each sample individually
        processed_graphs = []
        processed_data = []
        list_selected_edges = []
        self.processed_graph = processed_graphs
        for i, (sample_id, sample) in enumerate(data.samples.items()):
            # Create a Data object with only the current sample
            sample_data = Data()
            # sample_data.data[sample_id] = sample
            sample_data.add_sample(sample_id, sample)

            # Build the problem for this sample using the base method
            sample_problem = self.base_method.build(G, sample_data)
            m_graph = self.base_method.processed_graph
            m_data = self.base_method.processed_data
            # Store the processed graph and data for this sample
            processed_graphs.append(m_graph)
            processed_data.append(m_data)
            edge_ids = m_graph.get_attr_from_edges("id")
            # Store the selected edges for this sample
            list_selected_edges.append(edge_ids)

            # If we're using structured sparsity, collect the edge selection variable for this sample
            if not self.disable_structured_sparsity and self.lambda_reg_param.value > 0 and self.edge_selection_varname:
                if self.edge_selection_varname in sample_problem.expr:
                    edge_selection_vars.append(sample_problem.expr[self.edge_selection_varname])
                else:
                    raise ValueError(
                        f"Edge selection variable '{self.edge_selection_varname}' not found in the problem created by the base method"
                    )

            # Add a unique suffix to all variables in this problem
            sample_problem.add_suffix(f"_{i}", inplace=True)

            # Add this problem to the combined problem
            if combined_problem is None:
                combined_problem = sample_problem
            else:
                # Combine the two problems
                combined_problem += sample_problem

        # If we have collected edge selection variables and structured sparsity is enabled,
        # create a combined variable for regularization
        if edge_selection_vars and not self.disable_structured_sparsity and self.lambda_reg_param.value > 0:
            num_samples = len(data.samples)
            final_result = self._backend.Constant(np.zeros((graph.num_edges, num_samples)))
            for i, selected_edges in enumerate(list_selected_edges):
                # The length of selected_edges matches the number of edges in preprocessed graph i
                # It can contain None if the edge does not appear in the original graph, those we
                # need to ignore.
                vec = edge_selection_vars[i]
                if len(selected_edges) != vec.shape[0]:
                    raise ValueError(f"Mismatch between selected edges and edge selection variable size for sample {i}")
                # We need to map the values in `vec`, which indicates if an edge in the preprocessed graph
                # is selected or not, to the edges in the original graph, using the selected_edges list that
                # contains the indexes of the edges in the original graph. The variable selected_edges contains
                # None for edges that are not selected in the original graph, those we ignore.
                # --- Step 1: Filter out None entries ---
                # Create a boolean mask that is True for valid indices (not None)
                valid_mask = np.array([edge is not None for edge in selected_edges])

                # Extract only the valid entries from vec (reshape as a column vector)
                valid_vec = vec[np.flatnonzero(valid_mask)]  # .reshape(-1, 1)

                # Extract the valid edge indices (they are already 0-indexed)
                valid_edges = np.array([edge for edge in selected_edges if edge is not None])

                # --- Step 2: Build the indicator matrix E ---
                # m is the number of edges in the original graph
                m = graph.num_edges  # Total number of rows in the target vector

                # Create a column vector of row indices for the original graph: shape (m, 1)
                rows = np.arange(m).reshape(m, 1)

                # Build E by comparing rows with valid_edges.
                # valid_edges is reshaped to a row vector (shape: (1, n_valid)) for proper broadcasting,
                # resulting in an indicator matrix E of shape (m, n_valid).
                E = (rows == valid_edges.reshape(1, -1)).astype(int)

                # --- Step 3: Matrix multiplication to scatter the values ---
                # Here the "@" operator multiplies the indicator matrix E with valid_vec.
                # Y is then a vector of shape (m, 1) with the values from vec mapped
                # to their corresponding positions in the original graph.
                Y = E @ valid_vec
                one_hot = np.eye(num_samples)[i].reshape(1, num_samples)
                # Now compute the outer product using the "@" operator.
                # Y is (m,1) and one_hot is (1, num_samples) so their product is (m, num_samples).
                contribution = Y @ one_hot

                # Algebraically add the contribution to the final_result matrix.
                final_result = final_result + contribution

            # Horizontally stack edge selection variables from all samples
            # NOTE: We need to make sure that the indexes of the variables
            # match the edge locations in the graph
            combined_problem.register(self._reg_varname, final_result)

        return combined_problem
