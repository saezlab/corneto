import numpy as np


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
    import re

    # find AND gates with regular expression: AND[0+9]+
    pattern = re.compile(r"^(AND|And|and)[0-9]+$")
    V_is_and = [bool(pattern.match(v)) for v in G.V]

    return np.array(V_is_and)


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
