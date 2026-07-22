"""Modern CARNIVAL optimization implementations."""

from typing import Any, Iterable, Optional, Set, Tuple

import numpy as np

from corneto._constants import VarType
from corneto._settings import sparsify
from corneto._util import unique_iter
from corneto.backend._base import Backend
from corneto.data import Data
from corneto.graph import BaseGraph

# from corneto.methods import expand_graph_for_flows
from corneto.methods._base import FlowMethod, Method
from corneto.methods._input_utils import (
    DEFAULT_CONDITION,
    data_from_features,
    legacy_data,
    validate_condition_maps,
    validate_vertices,
)
from corneto.methods.signaling._utils import (
    get_incidence_matrices_of_edges,
    get_interactions,
)


def create_flow_graph(G: BaseGraph, inputs: Iterable[Any], outputs: Iterable[Any]) -> BaseGraph:
    """Add flow edges to perturbed and measured nodes in graph ``G``."""
    G1 = G.copy()
    for v in unique_iter(outputs):
        G1.add_edge(v, ())
    for v in unique_iter(inputs):
        G1.add_edge((), v)
    return G1


def prune_graph(
    G: BaseGraph,
    data: Data,
    property_key: str = "type",
    input_key: str = "input",
    output_key: str = "output",
) -> Tuple[BaseGraph, Data]:
    """Prune the given BaseGraph according to specified dataset.

    Steps:
    1. For each condition in dataset:
       - Find relevant vertices (graph vertices ∩ condition inputs/outputs)
       - Prune subgraph using relevant vertices
       - Collect remaining input/output keys
    2. Collect pruned input/output vertices across conditions
    3. Prune original graph using all collected vertices

    Args:
        G: Graph-like object with:
            - V attribute (list/set of vertices)
            - prune(inputs, outputs) method returning a subgraph
        data: Data object containing input/output measurements
        property_key: Feature metadata key defining the measurement type.
        input_key: Value of ``property_key`` identifying input measurements.
        output_key: Value of ``property_key`` identifying output measurements.

    Returns:
        Tuple[BaseGraph, Data]: A tuple containing:
            - The pruned graph using all relevant vertices
            - The pruned dataset with pruned vertices
    """
    graph_vertices: Set[Any] = set(G.V)
    reachable_inputs = set()
    reachable_outputs = set()

    for sample in data.samples.values():
        sample_inputs = sample.query.select(lambda f: f.data[property_key] == input_key).pluck()

        sample_outputs = sample.query.select(lambda f: f.data[property_key] == output_key).pluck()

        # Intersect with the current graph's vertices
        inputs_in_graph = graph_vertices & sample_inputs
        outputs_in_graph = graph_vertices & sample_outputs

        # Prune the graph based on relevant inputs and outputs
        sub_graph = G.prune(list(inputs_in_graph), list(outputs_in_graph))
        subgraph_vertices = set(sub_graph.V)
        reachable_inputs.update(inputs_in_graph & subgraph_vertices)
        reachable_outputs.update(outputs_in_graph & subgraph_vertices)

    # Prune the original graph with all collected inputs/outputs
    pruned_graph = G.prune(list(reachable_inputs), list(reachable_outputs))
    pruned_data = data.query.filter_features(lambda f: f.id in pruned_graph.V).collect()
    # pruned_data = data.subset(feature_predicate=lambda f: f.id in pruned_graph.V)
    return pruned_graph, pruned_data


def create_signed_error_expression(P, values, index_of_vertices=None, condition_index=None, vertex_variable=None):
    # If variable not provided, assumes we have the expected variables in the problem
    if vertex_variable is None:
        if "vertex_value" not in P.expr:
            raise ValueError("vertex_variable must be provided if not in the problem")
        vertex_variable = P.expr.vertex_value
    if index_of_vertices is None:
        index_of_vertices = range(vertex_variable.shape[0])
    if len(index_of_vertices) != len(values):
        raise ValueError("index_of_vertices must be the same length as values")
    if len(vertex_variable.shape) > 2:
        raise ValueError("vertex_variable must be 1D or 2D")
    if len(vertex_variable.shape) == 2:
        if condition_index is None:
            raise ValueError("condition_index must be provided if there are more than one sample")
        return (1 - vertex_variable[index_of_vertices, condition_index].multiply(np.sign(values))).multiply(abs(values))
    else:
        if condition_index is not None and condition_index > 0:
            raise ValueError("condition_index must be None or 0 if there is only one single sample")
        return (1 - vertex_variable[index_of_vertices].multiply(np.sign(values))).multiply(abs(values))


def _carnival_data(graph, perturbations, transcription_factors) -> Data:
    conditions = validate_condition_maps(
        perturbations=perturbations,
        transcription_factors=transcription_factors,
    )
    features_by_condition = {}
    for condition in conditions["perturbations"]:
        inputs = validate_vertices(
            graph,
            conditions["perturbations"][condition],
            argument="perturbations",
            condition=condition,
        )
        outputs = validate_vertices(
            graph,
            conditions["transcription_factors"][condition],
            argument="transcription_factors",
            condition=condition,
        )
        overlap = inputs.keys() & outputs.keys()
        if overlap:
            raise ValueError(
                f"Vertices cannot be both perturbations and transcription_factors "
                f"in condition {condition!r}: {sorted(overlap, key=repr)!r}."
            )
        features_by_condition[condition] = [
            {"id": identifier, "value": value, "mapping": "vertex", "role": "input"}
            for identifier, value in inputs.items()
        ] + [
            {"id": identifier, "value": value, "mapping": "vertex", "role": "output"}
            for identifier, value in outputs.items()
        ]
    return data_from_features(features_by_condition)


class _CarnivalUserInputs:
    """User-facing input conversion shared by CARNIVAL formulations."""

    def build(
        self,
        pkn: BaseGraph,
        data: Optional[Data] = None,
        *,
        perturbations=None,
        transcription_factors=None,
    ):
        old_data = legacy_data(data, method=self.__class__.__name__)
        if old_data is not None:
            if perturbations is not None or transcription_factors is not None:
                raise TypeError("Do not combine a Data object with explicit scientific inputs.")
            return self.build_from_data(pkn, old_data)
        if perturbations is None or transcription_factors is None:
            raise TypeError("build() requires perturbations= and transcription_factors=.")
        return self.build_many(
            pkn,
            perturbations={DEFAULT_CONDITION: perturbations},
            transcription_factors={DEFAULT_CONDITION: transcription_factors},
        )

    def build_many(self, pkn: BaseGraph, *, perturbations, transcription_factors):
        data = _carnival_data(pkn, perturbations, transcription_factors)
        return self.build_from_data(pkn, data)


class CarnivalFlow(_CarnivalUserInputs, FlowMethod):
    """Flow-base, multi-sample CARNIVAL method for intracellular signaling.

    Implements multi-sample intracellular network inference using
    an extension of the CARNIVAL method to model signal propagation.

    Args:
        exclusive_signal_paths: Whether proteins cannot be simultaneously
            activated/inhibited through different paths. Default: True
        lambda_reg: Regularization for edge signals across samples.
            Higher values give sparser solutions. Default: 0.0
        max_flow: Upper limit on the flow. It relates to the maximum number
            of branches in the signaling tree. Minimum value is 1. Reducing
            this number decreases the size of the solutions. Decrease it
            to reduce the solution space size and increase optimization speed.
            Default: 1000.
        enable_bfs_heuristic: Use BFS heuristic to strengthen acyclicity
            constraints. Default: True
        backend: Optimization backend to use. Default: None

    """

    def __init__(
        self,
        lambda_reg=0.0,
        exclusive_signal_paths=True,
        vertex_lb_dist=None,
        max_flow=1000,
        enable_bfs_heuristic=True,
        indirect_rule_penalty=0,
        depth_penalty=0,
        data_type_key="role",
        data_input_key="input",
        data_output_key="output",
        backend: Optional[Backend] = None,
    ):
        super().__init__(
            backend=backend,
            lambda_reg=lambda_reg,
            reg_varname="edge_has_signal",
            flow_upper_bound=max_flow,
        )
        self.exclusive_signal_paths = exclusive_signal_paths
        self.data_type_key = data_type_key
        self.data_input_key = data_input_key
        self.data_output_key = data_output_key
        self.vertex_lb_dist = vertex_lb_dist
        self.use_heuristic_bfs = enable_bfs_heuristic
        self.indirect_rule_penalty = indirect_rule_penalty

    def preprocess(self, graph: BaseGraph, data: Data) -> Tuple[BaseGraph, Data]:
        """Preprocess the input graph and dataset before optimization.

        This method performs two main preprocessing steps:
        1. Prunes the graph based on the input conditions to remove irrelevant vertices
        2. Expands the graph to accommodate flow-based constraints

        Args:
            graph (BaseGraph): The input network graph to be processed
            data (Data): Experimental input and output measurements.

        Returns:
            Tuple[BaseGraph, Dataset]: A tuple containing:
                - The preprocessed graph with expanded flow capabilities
                - The preprocessed dataset with standardized format
        """
        pruned_graph, pruned_data = prune_graph(
            graph, data, self.data_type_key, self.data_input_key, self.data_output_key
        )

        # We use the inputs/outputs of the dataset to expand the graph into a flow graph
        # inputs = pruned_data.collect_features(self.data_type_key, self.data_input_key)
        # outputs = pruned_data.collect_features(
        #     self.data_type_key, self.data_output_key
        # )
        inputs = pruned_data.query.filter_features(
            lambda f: f.data.get(self.data_type_key, None) == self.data_input_key
        ).pluck_features()
        outputs = pruned_data.query.filter_features(
            lambda f: f.data.get(self.data_type_key, None) == self.data_output_key
        ).pluck_features()
        flow_graph = create_flow_graph(pruned_graph, inputs, outputs)
        return flow_graph, pruned_data

    def create_flow_based_problem(self, flow_problem, graph: BaseGraph, data: Data):
        """Create the optimization problem with flow-based constraints.

        Sets up an integer linear programming problem by:
        1. Creating binary variables for edge activations and inhibitions.
        2. Defining signal propagation constraints.
        3. Enforcing acyclic signal flow.
        4. Incorporating experimental measurements into the objective.

        Args:
            flow_problem: The base optimization problem to build upon.
            graph (BaseGraph): The preprocessed network graph.
            data (Data): The experimental dataset.

        Returns:
            The configured optimization problem.
        """
        lb_dist = []
        unreachable_vertices_per_sample_idx = []
        if self.use_heuristic_bfs:
            vertex_idx = {v: i for i, v in enumerate(graph.V)}
            graph_vertices = frozenset(vertex_idx.keys())
            for sample in data.samples.values():
                # sample_inputs = sample.filter_values_by(
                #    self.data_type_key, self.data_input_key
                # )
                # sample_outputs = sample.filter_values_by(
                #    self.data_type_key, self.data_output_key
                # )
                # sample_inputs = list(sample_inputs.keys())
                # sample_outputs = list(sample_outputs.keys())
                sample_inputs = sample.query.select(lambda f: f.data[self.data_type_key] == self.data_input_key).pluck()
                sample_outputs = sample.query.select(
                    lambda f: f.data[self.data_type_key] == self.data_output_key
                ).pluck()
                # print(len(sample_inputs), len(sample_outputs))
                # Get the distance between inputs and outputs
                dist_dict = graph.bfs(sample_inputs, sample_outputs)
                pruned_g = graph.prune(sample_inputs, sample_outputs)
                unreachable = graph_vertices - set(pruned_g.V) - sample_inputs
                print(f"Unreachable vertices for sample: {len(unreachable)}")
                lb_dist.append(dist_dict)
                unreachable_vertices_per_sample_idx.append([vertex_idx[v] for v in unreachable])
            self.vertex_lb_dist = lb_dist

        # Alias for convenience and extract key constants
        problem = flow_problem
        num_experiments = len(data.samples)
        ones = np.ones((1, num_experiments))

        # Get incidence matrices and interactions from the graph
        At, Ah = get_incidence_matrices_of_edges(graph, sparse=True)
        interaction = get_interactions(graph)

        # Create binary variables for edge activations and inhibitions
        Eact = self.backend.Variable("edge_activates", (graph.num_edges, num_experiments), vartype=VarType.BINARY)
        Einh = self.backend.Variable("edge_inhibits", (graph.num_edges, num_experiments), vartype=VarType.BINARY)
        # Prevent an edge from activating and inhibiting simultaneously
        problem += Eact + Einh <= 1

        # PICOS requires a constant for a sparse matrix on the
        # left hand side, otherwise it fails.
        At_c = self.backend.Constant(At)
        Ah_c = self.backend.Constant(Ah)
        # Calculate vertex signal: activations minus inhibitions
        Va = At_c @ Eact
        Vi = At_c @ Einh
        V = Va - Vi

        # Unreachable vertices are set to 0 (if heuristics are used)
        if self.use_heuristic_bfs:
            for sample_idx, unreachable in enumerate(unreachable_vertices_per_sample_idx):
                if len(unreachable) == 0:
                    continue
                problem += V[unreachable, sample_idx] == 0
                problem += Va[unreachable, sample_idx] == 0
                problem += Vi[unreachable, sample_idx] == 0

        # Optionally enforce exclusive signal paths
        if self.exclusive_signal_paths:
            problem += Va + Vi <= 1

        # Register key variables for later use
        problem.register("vertex_value", V)
        problem.register("vertex_activated", Va)
        problem.register("vertex_inhibited", Vi)
        problem.register("edge_value", Eact - Einh)
        problem.register("edge_has_signal", Eact + Einh)

        # Add acyclic constraints to prevent cycles in signal propagation
        problem = self.backend.Acyclic(
            graph,
            problem,
            indicator_positive_var_name="edge_has_signal",
            vertex_lb_dist=self.vertex_lb_dist,
        )
        # Alias for default _dag_layer (to be changed in the future)
        problem.register("vertex_max_depth", problem.expr._dag_layer)

        # Identify edges with outgoing connections (heads)
        edges_with_head = np.flatnonzero(np.sum(np.abs(Ah), axis=0) > 0)

        # Extend flows across experiments; propagate signals only where flow exists.
        F = problem.expr.flow.reshape((Eact.shape[0], 1)) @ ones
        # Note: we might require scaling F, since this imposes that every edge
        # with signal needs to have a flow equal or greater than 1. Upper bound
        # of flow constrains the maximum size of the union graph.
        problem += Eact + Einh <= F

        # Broadcast and sparsify the interaction matrix for all experiments
        Int = sparsify(np.reshape(interaction, (interaction.shape[0], 1)) @ ones)

        # Precompute upstream signal contributions for edges with heads
        upstream_Va = (Ah_c.T @ Va)[edges_with_head, :]
        upstream_Vi = (Ah_c.T @ Vi)[edges_with_head, :]

        # Constrain activations based on upstream signals
        cond_act = (Int[edges_with_head, :] > 0).astype(int)
        cond_inh = (Int[edges_with_head, :] < 0).astype(int)
        problem += Eact[edges_with_head, :] <= upstream_Va.multiply(cond_act) + upstream_Vi.multiply(cond_inh)

        # Constrain inhibitions (swapping conditions)
        cond_act_inv = (Int[edges_with_head, :] < 0).astype(int)
        cond_inh_inv = (Int[edges_with_head, :] > 0).astype(int)
        problem += Einh[edges_with_head, :] <= upstream_Va.multiply(cond_act_inv) + upstream_Vi.multiply(cond_inh_inv)

        # Pre-collect all input features for use in designated perturbation constraints
        # all_inputs = data.collect_features(self.data_type_key, self.data_input_key)
        all_inputs = data.query.filter_features(
            lambda f: f.data[self.data_type_key] == self.data_input_key
        ).pluck_features()

        for i, (sample_name, sample) in enumerate(data.samples.items()):
            # --- Input Perturbation Constraints ---
            # sample_inputs = sample.filter_values_by(
            #    self.data_type_key, self.data_input_key
            # )
            sample_inputs = dict(
                sample.query.select(lambda f: f.data[self.data_type_key] == self.data_input_key).pluck(
                    lambda f: (f.id, f.value)
                )
            )

            # For multiple experiments, activate only designated perturbation inputs.
            # This version uses one flow and multiple acyclic signals across the
            # subgraph with flow, so it can block signal but not flow edges.
            if num_experiments > 1:
                p_nodes_set = set(sample_inputs.keys())
                other_inputs = all_inputs - p_nodes_set
                other_input_edges = [
                    idx for v in other_inputs for (idx, _) in graph.in_edges(v) if len(graph.get_edge(idx)[0]) == 0
                ]
                if other_input_edges:
                    problem += Eact[other_input_edges, i] == 0
                    problem += Einh[other_input_edges, i] == 0

            # Enforce equality constraints on nonzero input perturbations
            p_nodes = list(sample_inputs.keys())
            p_values = list(sample_inputs.values())
            p_positions = [graph.V.index(node) for node in p_nodes]
            # Filter out zero perturbations and only use nonzero signals
            nonzero_positions = [pos for pos, val in zip(p_positions, p_values) if val != 0]
            nonzero_signs = [np.sign(val) for val in p_values if val != 0]
            if nonzero_positions:
                problem += V[np.array(nonzero_positions), i] == np.array(nonzero_signs)

            # --- Objective: Error Terms from Experimental Outputs ---
            # sample_outputs = sample.filter_values_by(
            #    self.data_type_key, self.data_output_key
            # )
            sample_outputs = dict(
                sample.query.select(lambda f: f.data[self.data_type_key] == self.data_output_key).pluck(
                    lambda f: (f.id, f.value)
                )
            )

            m_nodes = list(sample_outputs.keys())
            m_values = np.array(list(sample_outputs.values()))
            m_positions = [graph.V.index(node) for node in m_nodes]
            # Choose vertex values based on experiment count
            # val = V[m_positions, i] if num_experiments > 1 else V[m_positions]
            error_expr = create_signed_error_expression(
                problem,
                m_values,
                index_of_vertices=m_positions,
                condition_index=i,
                vertex_variable=V,
            )
            # ones = np.ones((len(m_nodes), 1))  # Column vector of ones
            # problem.add_objectives(sum(error_expr))
            # problem.add_objectives(error_expr @ ones)
            problem.add_objective(error_expr.sum(), name=f"error_{sample_name}_{i}")
            if self.indirect_rule_penalty > 0:
                # Penalize more indirect rules:
                # A -> B interaction, but edge activity = -1 or
                # A -| B interaction, but edge activity = 1
                # i.e., in the first case, B is inhibited just
                # because A is inhibited and B is active just
                # because A is inhibited
                activatory_interactions = (Int[:, i] > 0).astype(int)
                inhibitory_interactions = (Int[:, i] < 0).astype(int)
                penalty_rule1 = Einh[:, i].T @ activatory_interactions
                penalty_rule2 = Eact[:, i].T @ inhibitory_interactions
                problem.add_objective(
                    penalty_rule1 + penalty_rule2,
                    weight=self.indirect_rule_penalty,
                    name=f"penalty_indirect_rules_{i}",
                )

        return problem

    @staticmethod
    def references():
        """Returns citation keys for this method.

        Returns:
            A list of citation keys that can be used to lookup BibTeX entries.
        """
        return ["rodriguez2024unified", "liu2019expression"]

    @staticmethod
    def description():
        """Returns a description of the method.

        Returns:
            A string describing the method.
        """
        return (
            "Extension of CARNIVAL for intracellular network inference "
            "that uses integer linear programming to model signal propagation."
        )


class CarnivalILP(_CarnivalUserInputs, Method):
    """Multi-condition implementation of the CARNIVAL ILP formulation.

    Each sample in :class:`corneto.data.Data` receives an independent signaling
    state on a shared network. Binary variables and ILP constraints model signal
    propagation without the flow formulation used by :class:`CarnivalFlow`.

    Args:
        beta_weight: Regularization term weight. Default: 0.2
        max_dist: Max distance between vertices. If None, uses vertex count.
            Default: None
        penalize: What to regularize - 'nodes'/'edges'/'both'. Default: 'edges'
        use_perturbation_weights: Include perturbation weights. Default: False
        interaction_graph_attribute: Edge attribute for interactions.
            Default: 'interaction'
        disable_acyclicity: Skip acyclicity constraints. Default: False
        backend: Optimization backend. Default: None
    """

    def __init__(
        self,
        beta_weight: float = 0.2,
        max_dist: Optional[int] = None,
        penalize: str = "edges",
        use_perturbation_weights: bool = False,
        interaction_graph_attribute: str = "interaction",
        disable_acyclicity: bool = False,
        data_type_key: str = "role",
        data_input_key: str = "input",
        data_output_key: str = "output",
        backend: Optional[Backend] = None,
    ):
        super().__init__(lambda_reg=0, backend=backend, disable_structured_sparsity=True)
        self.beta_weight = beta_weight
        self.max_dist = max_dist
        self.penalize = penalize
        self.use_perturbation_weights = use_perturbation_weights
        self.interaction_graph_attribute = interaction_graph_attribute
        self.disable_acyclicity = disable_acyclicity
        self.data_type_key = data_type_key
        self.data_input_key = data_input_key
        self.data_output_key = data_output_key

    def preprocess(self, graph: BaseGraph, data: Data) -> Tuple[BaseGraph, Data]:
        """Preprocess the input graph and dataset before optimization.

        This method performs:
        1. Graph pruning based on input conditions to remove irrelevant vertices
        2. Data standardization for optimization

        Args:
            graph: The input network graph
            data: Experimental dataset with inputs/outputs

        Returns:
            A tuple containing preprocessed graph and dataset
        """
        pruned_graph, pruned_data = prune_graph(
            graph, data, self.data_type_key, self.data_input_key, self.data_output_key
        )
        return pruned_graph, pruned_data

    def create_problem(self, graph: BaseGraph, data: Data):
        """Create the ILP optimization problem.

        This method implements the core CARNIVAL optimization problem by:
        1. Creating binary variables for vertex and edge states
        2. Setting up consistency constraints
        3. Adding acyclicity constraints if enabled
        4. Incorporating measurements into the objective

        Args:
            graph: The preprocessed network graph
            data: The preprocessed dataset

        Returns:
            The configured optimization problem
        """
        max_dist = self.max_dist if self.max_dist is not None else graph.num_vertices
        num_conditions = len(data.samples)
        is_multicondition = num_conditions > 1

        def condition_shape(size):
            return (size, num_conditions) if is_multicondition else (size,)

        def at_condition(variable, index, condition):
            if is_multicondition:
                return variable[index, condition]
            return variable[index]

        # Create the problem
        P = self.backend.Problem()

        # Each condition has its own signaling state on the shared graph. Keep
        # one-dimensional variables for the single-condition case so existing
        # result-access code remains compatible.
        V_act = self.backend.Variable(
            "vertex_activated",
            shape=condition_shape(len(graph.V)),
            vartype=VarType.BINARY,
        )
        V_inh = self.backend.Variable(
            "vertex_inhibited",
            shape=condition_shape(len(graph.V)),
            vartype=VarType.BINARY,
        )
        E_act = self.backend.Variable(
            "edge_activating",
            shape=condition_shape(len(graph.E)),
            vartype=VarType.BINARY,
        )
        E_inh = self.backend.Variable(
            "edge_inhibiting",
            shape=condition_shape(len(graph.E)),
            vartype=VarType.BINARY,
        )
        V_pos = self.backend.Variable(
            "vertex_position",
            shape=condition_shape(len(graph.V)),
            lb=0,
            ub=max_dist,
            vartype=VarType.CONTINUOUS,
        )

        V_index = {v: i for i, v in enumerate(graph.V)}

        # A vertex can be activated or inhibited, but not both
        P += V_act + V_inh <= 1
        # An edge can activate or inhibit, but not both
        P += E_act + E_inh <= 1

        for edge_index, (s, t) in enumerate(graph.E):
            s = list(s)
            t = list(t)
            if len(s) == 0:
                continue
            if len(s) > 1:
                raise ValueError("Only one source vertex allowed")
            if len(t) > 1:
                raise ValueError("Only one target vertex allowed")
            s = s[0]
            t = t[0]
            # An edge can activate its downstream (target vertex) (E_act=1, E_inh=0),
            # inhibit it (E_act=0, E_inh=1), or do nothing (E_act=0, E_inh=0)
            si = V_index[s]
            ti = V_index[t]
            interaction = int(graph.get_attr_edge(edge_index).get(self.interaction_graph_attribute))
            if interaction not in {-1, 1}:
                raise ValueError(f"Invalid interaction value for edge {edge_index}: {interaction}")
            for condition in range(num_conditions):
                edge_activates = at_condition(E_act, edge_index, condition)
                edge_inhibits = at_condition(E_inh, edge_index, condition)
                source_activated = at_condition(V_act, si, condition)
                source_inhibited = at_condition(V_inh, si, condition)

                if interaction == 1:
                    P += edge_activates <= source_activated
                    P += edge_inhibits <= source_inhibited
                else:
                    P += edge_activates <= source_inhibited
                    P += edge_inhibits <= source_activated

                if not self.disable_acyclicity:
                    edge_selected = edge_activates + edge_inhibits
                    source_position = at_condition(V_pos, si, condition)
                    target_position = at_condition(V_pos, ti, condition)
                    P += target_position - source_position >= 1 - max_dist * (1 - edge_selected)

        for condition, (sample_name, sample) in enumerate(data.samples.items()):
            perturbations = dict(
                sample.query.select(lambda f: f.data[self.data_type_key] == self.data_input_key).pluck(
                    lambda f: (f.id, f.value)
                )
            )
            measurements = dict(
                sample.query.select(lambda f: f.data[self.data_type_key] == self.data_output_key).pluck(
                    lambda f: (f.id, f.value)
                )
            )

            for vertex in graph.V:
                in_edge_indices = [i for i, _ in graph.in_edges(vertex)]
                vertex_index = V_index[vertex]
                vertex_activated = at_condition(V_act, vertex_index, condition)
                vertex_inhibited = at_condition(V_inh, vertex_index, condition)
                incoming_activating = sum(at_condition(E_act, edge_index, condition) for edge_index in in_edge_indices)
                incoming_inhibiting = sum(at_condition(E_inh, edge_index, condition) for edge_index in in_edge_indices)
                if in_edge_indices:
                    P += incoming_activating + incoming_inhibiting <= 1

                perturbed = vertex in perturbations
                P += vertex_activated <= int(perturbed) + incoming_activating
                P += vertex_inhibited <= int(perturbed) + incoming_inhibiting

                perturbed_value = np.sign(perturbations.get(vertex, 0))
                if perturbed_value > 0:
                    P += vertex_activated == 1
                    P += vertex_inhibited == 0
                elif perturbed_value < 0:
                    P += vertex_activated == 0
                    P += vertex_inhibited == 1

            objective_data = measurements.copy()
            if self.use_perturbation_weights:
                objective_data.update(perturbations)

            error_terms = []
            for vertex, value in objective_data.items():
                vertex_index = V_index[vertex]
                prediction = at_condition(V_act, vertex_index, condition) - at_condition(V_inh, vertex_index, condition)
                sign = np.sign(value)
                if sign > 0:
                    error_terms.append(np.abs(value) * (sign - prediction))
                elif sign < 0:
                    error_terms.append(np.abs(value) * (prediction - sign))
            P.add_objective(sum(error_terms), name=f"error_{sample_name}")

        if self.beta_weight > 0:
            if self.penalize == "nodes":
                reg = V_act.sum() + V_inh.sum()
            elif self.penalize == "edges":
                reg = E_act.sum() + E_inh.sum()
            elif self.penalize == "both":
                reg = V_act.sum() + V_inh.sum() + E_act.sum() + E_inh.sum()
            else:
                raise ValueError("penalize must be 'nodes', 'edges', or 'both'")
            P.add_objective(reg, weight=self.beta_weight, name=f"regularization_{self.penalize}")

        # Finally, register some aliases for convenience
        P.register("vertex_values", V_act - V_inh)
        P.register("edge_values", E_act - E_inh)

        return P

    @staticmethod
    def references():
        """Returns citation keys for this method.

        Returns:
            A list of citation keys that can be used to lookup BibTeX entries.
        """
        return ["liu2019expression", "rodriguez2025unifying"]
