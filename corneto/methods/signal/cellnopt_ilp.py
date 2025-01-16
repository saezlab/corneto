import re
from typing import Literal, Optional

import numpy as np

import corneto as cn
from corneto.backend._base import EXPR_NAME_FLOW


def clip_quantiles(arr, q):
    if q < 0 or q > 1:
        raise ValueError(f"Clipping value must be between 0 and 1, got {q}")
    # compute the quantiles at clipping and 1-clipping and clip the flow
    q = np.quantile(arr, [q, 1 - q])
    return np.clip(arr, q[0], q[1])


def cno_style(
    P,
    max_edge_width: float = 5,
    min_edge_width: float = 0.25,
    flow_name: str = EXPR_NAME_FLOW,
    positive_color: str = "dodgerblue4",
    negative_color: str = "firebrick4",
    zero_flow_threshold: float = 1e-6,
    scale: Optional[Literal["log", "std"]] = "log",
    clip_quantil: Optional[float] = 0.05,
    iexp=0,
):
    flow = np.array(P.expr[flow_name].value)[:, iexp]
    flow[np.abs(flow) < zero_flow_threshold] = 0
    if scale is not None:
        if scale == "log":
            flow = np.log10(np.abs(flow) + 1e-6) * np.sign(flow)
        elif scale == "std":
            flow = flow / np.std(flow)
        else:
            raise ValueError(f"Unknown normalization method: {scale}")
    if clip_quantil is not None:
        flow = clip_quantiles(flow, clip_quantil)
    max_flow = max(np.max(np.abs(flow)), 1e-6)
    edge_attrs = dict()
    for i, v in enumerate(flow):
        # Apply threshold edge width
        if abs(v) > 0:
            edge_width = max_edge_width
        else:
            edge_width = min_edge_width
        if scale is not None:
            edge_width = min_edge_width + (max_edge_width - min_edge_width) * abs(
                v / max_flow
            )
        # bound = P.expr[flow_name].ub[i] if v >= 0 else P.expr[flow_name].lb[i]
        edge_attrs[i] = {"penwidth": str(edge_width)}
        if flow[i] > 0:
            edge_attrs[i]["color"] = positive_color
        elif flow[i] < 0:
            edge_attrs[i]["color"] = negative_color
        else:
            edge_attrs[i]["color"] = "black"
    return edge_attrs


def get_interactions(G):
    """Get the sign of interactions from the graph G. I in [1, -1]"""
    return np.array(G.get_attr_from_edges("interaction", 1))


def get_AND_gate_nodes(G):
    """Get the indices of nodes that represent AND gates in the graph G.

    Parameters:
    - G (Graph): The input graph.

    Returns:
    - np.array: An array containing the indices of nodes that represent AND gates.
    """
    # find AND gates with regular expression: AND[0+9]+
    pattern = re.compile(r"^(AND|And|and)[0-9]+$")
    V_is_and = [bool(pattern.match(v)) for v in G.V]

    return np.array(V_is_and)


def get_incidence_matrices_of_edges(G, as_dataframe=False):
    """Get the mapping matrices A, At, Ah from the graph G.

    Parameters:
    - G: The graph object.
    - as_dataframe: Whether to return the matrices as pandas DataFrames. Default is False.

    Returns:
    - At: The tail-incidence matrix.
    - Ah: The head-incidence matrix.
    """
    A = G.vertex_incidence_matrix().astype(int)  # V x E
    At = np.clip(A, 0, 1)  # V x E, 1 if vertex appears as tail of edge
    Ah = np.clip(-A, 0, 1)  # V x E, 1 if vertex appears as head of edge

    if as_dataframe:
        import pandas as pd

        Ah = pd.DataFrame(Ah, index=G.V, columns=G.E)
        At = pd.DataFrame(At, index=G.V, columns=G.E)

    return At, Ah


def get_egdes_with_head(G):
    """Get the indices of edges with a head node.

    Parameters:
        G (graph): The input graph.

    Returns:
        edges_with_head (array): An array containing the indices of edges with a head node.
    """
    At, Ah = get_incidence_matrices_of_edges(G)
    edges_with_head = np.flatnonzero(np.sum(np.abs(Ah), axis=0) > 0)
    return edges_with_head


def get_inhibited_nodes(G, exp_list):
    """Returns an array, with shape = (len(G.V), len(exp_list)), where each column is a boolean array
    indicating if the node is inhibited in the corresponding experiment.
    """
    V_is_inhibited = np.full((len(G.V), len(exp_list)), False)

    for exp, iexp in zip(exp_list, range(len(exp_list))):
        if "inhibition" not in exp_list[exp]:
            continue
        i_nodes = list(exp_list[exp]["inhibition"].keys())
        V_is_inhibited[:, iexp] = np.array([v in i_nodes for v in G.V])
    return V_is_inhibited


def presolve_report(G, exp_list):
    At, Ah = get_incidence_matrices_of_edges(G, as_dataframe=True)
    interaction = get_interactions(G)
    edges_with_head = get_egdes_with_head(G)
    V_is_and = get_AND_gate_nodes(G)

    print("Vertex order:")
    print(G.V)
    print("AND gates:")
    print(V_is_and)
    print("Tails of interactions:")
    print(At)
    print("Head of interactions:")
    print(Ah)
    print("Sign of interactions:")
    print(interaction)
    print("Edges with head:")
    print(edges_with_head)


def expand_graph_for_flows(G, exp_list):
    """Expand the graph G with the perturbations and measurements from the experiments in exp_list."""
    G1 = G.copy()
    output_names = list(
        {key for exp in exp_list.values() for key in exp["output"].keys()}
    )
    input_names = list(
        {key for exp in exp_list.values() for key in exp["input"].keys()}
    )

    output_names = list(set(output_names))
    input_names = list(set(input_names))

    for node in output_names:
        G1.add_edge(node, ())
    for node in input_names:
        G1.add_edge((), node)

    return G1


def check_exp_graph_consistency(G, exp_list):
    """Check if the experiments are consistent with the graph G."""
    for exp in exp_list:
        for node in exp_list[exp]["input"]:
            if node not in G.V:
                raise ValueError(
                    f"Node {node} in experiment {exp} is not in the graph."
                )
        for node in exp_list[exp]["output"]:
            if node not in G.V:
                raise ValueError(
                    f"Node {node} in experiment {exp} is not in the graph."
                )
        if "inhibition" in exp_list[exp]:
            for node in exp_list[exp]["inhibition"]:
                if node not in G.V:
                    raise ValueError(
                        f"Node {node} in experiment {exp} is not in the graph."
                    )


def cellnoptILP(G, exp_list, solver=None, alpha_flow=1e-3, verbose=False):
    """Create and solves the ILP model for the given graph G and the list of experiments exp_list.

    Parameters:
    - G: The graph representing the network.
    - exp_list: The list of experiments.
    - solver: The solver to use for solving the ILP problem. Default is None.
    - alpha_flow: The weight of the penalty of flow in the objective. Default is 1e-3.
    - verbose: Whether to print verbose output. Default is False.

    Returns:
    - P: The ILP model.
    """
    check_exp_graph_consistency(G, exp_list)

    At, Ah = get_incidence_matrices_of_edges(G)
    interaction = get_interactions(G)
    edges_with_head = get_egdes_with_head(G)
    V_is_and = get_AND_gate_nodes(G)
    V_is_inhibited = get_inhibited_nodes(G, exp_list)

    # let's start with acyclic flow
    P = cn.K.AcyclicFlow(G)

    # vertex value is binary (0 and 1)
    V = cn.K.Variable(
        "vertex_value", (G.num_vertices, len(exp_list)), vartype=cn.VarType.BINARY
    )
    # edge activation is also binary:
    Eact = cn.K.Variable(
        "edge_activates", (G.num_edges, len(exp_list)), vartype=cn.VarType.BINARY
    )

    M = 100  # a large number, so sum(incoming edges)/M is always less than 1

    # Dummy variable for the linearization of the absolute deviation objective function
    Z = cn.K.Variable(
        "dummy", (G.num_vertices, len(exp_list)), vartype=cn.VarType.CONTINUOUS
    )
    P += Z >= 0

    # some basic constraints
    P += V >= 0
    P += V <= 1

    # Rule 1: Edge can be activated only if they carry flow
    for exp, iexp in zip(exp_list, range(len(exp_list))):
        P += Eact[:, iexp] <= P.expr.with_flow

        # Rule 2: The edges can take the value of the upstream vertex, if the sign is 1. If the sign is -1, then they take the value of 1 - upstream vertex
        for exp, iexp in zip(exp_list, range(len(exp_list))):
            # this should keep Eact in [0,1] interval:

            # signed value of source/head node for an edge:
            V_head = (Ah.T @ V)[edges_with_head, iexp].multiply(
                interaction[edges_with_head] > 0
            ) + (1 - (Ah.T @ V)[edges_with_head, iexp]).multiply(
                interaction[edges_with_head] < 0
            )
            # an edge is active if there is flow AND there head node is also active
            # logical AND is translated to ILP as:
            # y >= x1 + x2 - 1  ; y <= x1 ; y <= x2
            P += (
                Eact[edges_with_head, iexp]
                >= P.expr.with_flow[edges_with_head] + V_head - 1
            )
            P += Eact[edges_with_head, iexp] <= V_head
            # P += Eact[edges_with_head,iexp] <=  P.expr.with_flow[edges_with_head] - we dont need this, see Rule 1

    # Rule 3: propagate the active edges to the vertices

    # This is for general nodes that are not AND gates
    #
    for exp, iexp in zip(exp_list, range(len(exp_list))):
        is_regular_node = np.logical_and(~V_is_and, ~V_is_inhibited[:, iexp])

        P += (
            V[is_regular_node, iexp] >= ((At @ Eact) / M)[is_regular_node, iexp]
        )  # when there is at least one active edge, the vertex becomes 1 (larger than  someValue/M)
        P += (
            V[is_regular_node, iexp] <= (At @ Eact)[is_regular_node, iexp]
        )  # but it has an upper constraint, so it takes 0 when all input are 0
        if V_is_inhibited[:, iexp].any():
            P += V[V_is_inhibited[:, iexp], iexp] == np.zeros(
                sum(V_is_inhibited[:, iexp])
            )

    # AND relation expressed as follows:
    # - we only define these constraints for the AND gates
    # - we count the sum of flows (selected edges) and sum of activated edges
    # - if sum of flow equal to the sum of incoming edge activation, i.e. all edge are activated, then the vertex is activated:
    # - to ensure that the AND gate is not active when there is no flow (an no active edge), we add the second constraint:
    if V_is_and.any():
        sum_of_flow = At[V_is_and, :] @ P.expr.with_flow
        sum_of_edge_activation = At[V_is_and, :] @ Eact

        #  (sum_of_flow - sum_of_edge_activation)/M is always less than 1 and it is 0 if  all edges are activated.
        for exp, iexp in zip(exp_list, range(len(exp_list))):
            P += (
                V[V_is_and, iexp]
                <= 1 - (sum_of_flow - sum_of_edge_activation[:, iexp]) / M
            )
            P += V[V_is_and, iexp] <= sum_of_flow

        P.register("sum_of_flow", sum_of_flow)
        P.register("sum_of_edge_activation", sum_of_edge_activation)

    for exp, iexp in zip(exp_list, range(len(exp_list))):
        # activation:
        p_nodes = list(exp_list[exp]["input"].keys())
        p_values = list(exp_list[exp]["input"].values())
        p_nodes_positions = [G.V.index(key) for key in p_nodes]

        P += V[p_nodes_positions, iexp] == p_values

        # measurements:
        m_nodes = list(exp_list[exp]["output"].keys())
        m_values = np.array(list(exp_list[exp]["output"].values()))
        m_nodes_positions = [G.V.index(key) for key in m_nodes]

        # linearization of the ABS function: https://lpsolve.sourceforge.net/5.1/absolute.htm
        P += V[m_nodes_positions, iexp] - m_values <= Z[m_nodes_positions, iexp]
        P += -(V[m_nodes_positions, iexp] - m_values) <= Z[m_nodes_positions, iexp]

        P.add_objectives(sum(Z[m_nodes_positions, iexp]))

    P.add_objectives(alpha_flow * sum(P.expr.with_flow))

    P.solve(solver=solver, verbosity=verbose)

    return P


def report_solution_tables(G, exp_list, P):
    import pandas as pd

    for iexp in range(len(exp_list)):
        print("--------- iexp: ", iexp, " ---------")
        print(pd.DataFrame({"V": G.V, "value": P.expr.vertex_value.value[:, iexp]}))
        print(
            pd.DataFrame(
                {
                    "E": G.E,
                    "flow": P.expr.with_flow.value,
                    "Eact": P.expr.edge_activates.value[:, iexp],
                }
            )
        )


def plot_solution_network_active_edges(G, P, iexp):
    G.plot(
        custom_edge_attr=cno_style(P, flow_name="edge_activates", scale=None, iexp=iexp)
    )


def plot_fitness(G, exp_list, P, measured_only=False, **kwargs):
    """Plot the fitness of the model simulation vs measurements.

    PARAMETERS:
    - G: corneto.Graph object
    - exp_list: dictionary of experiments
    - P: solution of the ILP model
    - measured_only: if True, plot only the measured nodes, otherwise plot all nodes
    - **kwargs arguments are passed to subplots and figures of matplotlib

    TODO: there are some assumptions, like the first experiment is the reference experiment and it is called 'exp0'
    """
    import matplotlib.pyplot as plt

    N_exps = len(exp_list)

    # Ensure that all experiments have the input and output variables
    for exp in exp_list.values():
        if "input" not in exp:
            raise ValueError("Input not found in experiment")
        if "output" not in exp:
            raise ValueError("Output not found in experiment")

    # Collect the input and output variables
    input_matrix, input_vars = collect_field_into_matrix(exp_list, "input")
    output_matrix, output_vars = collect_field_into_matrix(exp_list, "output")

    # Check if inhibition is present in any of the experiments
    inhibition_present = any("inhibition" in exp for exp in exp_list.values())
    if inhibition_present:
        inhibition_matrix, inhibition_vars = collect_field_into_matrix(
            exp_list, "inhibition"
        )
        perturbation_matrix = np.hstack((input_matrix, inhibition_matrix))
        perturbation_vars = input_vars + inhibition_vars
    else:
        perturbation_matrix = input_matrix
        perturbation_vars = input_vars

    # Create the figure
    # Set colors: input colors are blue, inhibition colors are red
    perturbation_colors = ["blue"] * len(input_vars)
    if inhibition_present:
        perturbation_colors += ["red"] * len(inhibition_vars)

    N_nodes = len(G.V)
    output_names = list(
        {key for exp in exp_list.values() for key in exp["output"].keys()}
    )

    # depending on the flag measured_only, we can plot only the measured nodes or all nodes
    if measured_only:
        fig, axs = plt.subplots(
            N_exps - 1, len(output_names) + 1, squeeze=False, **kwargs
        )
    else:
        fig, axs = plt.subplots(N_exps - 1, N_nodes + 1, squeeze=False, **kwargs)

    fig.tight_layout(pad=0.0)

    # Adjust the space between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for exp, iexp in zip(exp_list, range(N_exps)):
        if iexp == 0:
            continue

        if measured_only:
            for imarker in range(len(output_names)):
                # output_names[imarker] is the name of the output node, find the position in the graph
                imarker_inG = G.V.index(output_names[imarker])

                axs[iexp - 1, imarker].plot(
                    [0, 10],
                    [
                        P.expr.vertex_value.value[imarker_inG, 0],
                        min(P.expr.vertex_value.value[imarker_inG, iexp], 1),
                    ],
                    "bo-",
                    label=G.V[imarker_inG],
                )

                if G.V[imarker_inG] in exp_list[exp]["output"].keys():
                    axs[iexp - 1, imarker].plot(
                        [0, 10],
                        [
                            exp_list["exp0"]["output"][G.V[imarker_inG]],
                            exp_list[exp]["output"][G.V[imarker_inG]],
                        ],
                        "ro-",
                    )
                axs[iexp - 1, imarker].set_ylim([-0.05, 1.1])
                if iexp == 1:
                    axs[iexp - 1, imarker].set_title(output_names[imarker])
                if iexp != N_exps - 1:
                    axs[iexp - 1, imarker].set_xticks([])
                if imarker == 0:
                    axs[iexp - 1, imarker].set_ylabel(f"Exp. {iexp}")
                else:
                    axs[iexp - 1, imarker].set_yticks([])
        else:
            for imarker in range(N_nodes):
                axs[iexp - 1, imarker].plot(
                    [0, 10],
                    [
                        P.expr.vertex_value.value[imarker, 0],
                        min(P.expr.vertex_value.value[imarker, iexp], 1),
                    ],
                    "bo-",
                    label=G.V[imarker],
                    # color="blue",
                    # linestyle="o-",
                )

                if G.V[imarker] in exp_list[exp]["output"].keys():
                    axs[iexp - 1, imarker].plot(
                        [0, 10],
                        [
                            exp_list["exp0"]["output"][G.V[imarker]],
                            exp_list[exp]["output"][G.V[imarker]],
                        ],
                        "ro-",
                    )
                axs[iexp - 1, imarker].set_ylim([-0.05, 1.1])
                if iexp == 1:
                    axs[iexp - 1, imarker].set_title(G.V[imarker])
                if iexp != N_exps - 1:
                    axs[iexp - 1, imarker].set_xticks([])
                if imarker == 0:
                    axs[iexp - 1, imarker].set_ylabel(f"Exp. {iexp}")
                else:
                    axs[iexp - 1, imarker].set_yticks([])
        # Plot perturbation
        if measured_only:
            plot_location = len(output_names)
        else:
            plot_location = N_nodes

        axs[iexp - 1, plot_location].bar(
            range(len(perturbation_vars)),
            perturbation_matrix[iexp],
            color=perturbation_colors,
        )
        if iexp == N_exps - 1:
            axs[iexp - 1, plot_location].set_xticks(range(len(perturbation_vars)))
            axs[iexp - 1, plot_location].set_xticklabels(
                perturbation_vars, rotation=45
            )
        else:
            # No xtick label
            axs[iexp - 1, plot_location].set_xticks([])

        axs[iexp - 1, plot_location].set_ylim([-0.01, 1.1])
        axs[iexp - 1, plot_location].set_yticks([])
        if iexp == 1:
            axs[iexp - 1, plot_location].set_title("Pert.")

    plt.show()


def collect_field_into_matrix(experiments, field_name="input"):
    """Collects the field_name values into matrix.

    Collects the field_name values (input, inhibition etc) from a dictionary
    of experiments and returns them as a numpy array.

    PARAMETERS:
    - experiments: dictionary of experiments containing input values
    - field_name: name of the field to collect (default: 'input')

    Returns:
    - input_matrix: numpy array of input values
    - input_vars: list of unique input variable names

    """
    # Collect all unique input names
    input_vars = set()
    for exp in experiments.values():
        # ensure field_name exists
        if field_name not in exp:
            raise ValueError("Field name (" + field_name + ")  not found in experiment")
        input_vars.update(exp[field_name].keys())

    input_vars = sorted(input_vars)  # Sorting to keep a consistent order
    data = []

    # Collect input values for each experiment
    for exp in experiments.values():
        row = [exp[field_name].get(var, 0) for var in input_vars]
        data.append(row)

    # Convert the data to a numpy array
    input_matrix = np.array(data)

    return input_matrix, input_vars


def plot_data(exp_list):
    """Plot the data.

    PARAMETERS:
    - exp_list: dictionary of experiments

    """
    import matplotlib.pyplot as plt

    N_exps = len(exp_list)

    # Ensure that all experiments have the input and output variables
    for exp in exp_list.values():
        if "input" not in exp:
            raise ValueError("Input not found in experiment")
        if "output" not in exp:
            raise ValueError("Output not found in experiment")

    # Collect the input and output variables
    input_matrix, input_vars = collect_field_into_matrix(exp_list, "input")
    output_matrix, output_vars = collect_field_into_matrix(exp_list, "output")

    # Check if inhibition is present in any of the experiments
    inhibition_present = any("inhibition" in exp for exp in exp_list.values())
    if inhibition_present:
        inhibition_matrix, inhibition_vars = collect_field_into_matrix(
            exp_list, "inhibition"
        )
        perturbation_matrix = np.hstack((input_matrix, inhibition_matrix))
        perturbation_vars = input_vars + inhibition_vars
    else:
        perturbation_matrix = input_matrix
        perturbation_vars = input_vars

    # Create the figure
    # Set colors: input colors are blue, inhibition colors are red
    perturbation_colors = ["blue"] * len(input_vars)
    if inhibition_present:
        perturbation_colors += ["red"] * len(inhibition_vars)

    fig, axs = plt.subplots(N_exps - 1, len(output_vars) + 1, squeeze=False)

    fig.tight_layout(pad=0.0)
    # Adjust the space between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for exp, iexp in zip(exp_list, range(N_exps)):
        if iexp == 0:
            continue

        for imarker in range(len(output_vars)):
            # output_names[imarker] is the name of the output node, find the position in the graph
            imarker_name = output_vars[imarker]

            axs[iexp - 1, imarker].plot(
                [0, 10],
                [output_matrix[0, imarker], output_matrix[iexp, imarker]],
                "ro-",
            )
            axs[iexp - 1, imarker].set_ylim([-0.01, 1.1])

            if iexp == 1:
                axs[iexp - 1, imarker].set_title(imarker_name)
            if iexp != N_exps - 1:
                axs[iexp - 1, imarker].set_xticks([])
            if imarker == 0:
                axs[iexp - 1, imarker].set_ylabel(f"Exp. {iexp}")
            else:
                axs[iexp - 1, imarker].set_yticks([])

        # Plot perturbation
        axs[iexp - 1, len(output_vars)].bar(
            range(len(perturbation_vars)),
            perturbation_matrix[iexp],
            color=perturbation_colors,
        )
        if iexp == N_exps - 1:
            axs[iexp - 1, len(output_vars)].set_xticks(range(len(perturbation_vars)))
            axs[iexp - 1, len(output_vars)].set_xticklabels(
                perturbation_vars, rotation=45
            )
        else:
            # No xtick label
            axs[iexp - 1, len(output_vars)].set_xticks([])

        axs[iexp - 1, len(output_vars)].set_ylim([-0.01, 1.1])
        axs[iexp - 1, len(output_vars)].set_yticks([])
        if iexp == 1:
            axs[iexp - 1, len(output_vars)].set_title("Pert.")
    plt.show()
