"""Utility functions for data generation.

This module provides utility functions for generating random networks.

Functions
---------
generate_preferential_attachment_network : Generate a random network using the
    Barabasi-Albert preferential attachment model.
generate_duplication_divergence_network : Generate a random network using the
    duplication-divergence model.
generate_random_signalling_network : Generate a random biological signaling network
    with configurable properties such as inhibitory edge probability.
generate_random_duplication_divergence_signalling_network : Generate a random biological
    signaling network using the duplication-divergence model.
"""

import numpy as np


def generate_preferential_attachment_network(n, m, seed=None):
    """Generate a random network using the Barabasi-Albert preferential attachment model.

    Parameters
    ----------
    n : int
        Total number of nodes in the network (must be greater than m).
    m : int
        Number of edges each new node attaches with, controlling network density.
    seed : int or None, optional
        Random seed for reproducibility, default is None.

    Returns:
        list of tuples
            Each edge is represented as (source, target) where source and target
            are vertex labels like "v1", "v2", ...

    Notes:
        The algorithm implements a Barabasi-Albert preferential attachment model,
        which produces scale-free networks similar to those observed in biological
        systems. The initial network consists of a complete graph among the first
        m nodes.
    """
    if m >= n:
        raise ValueError("m must be smaller than n")
    rng = np.random.default_rng(seed)
    edges = []
    # Track degree counts for nodes 0 to n-1.
    degrees = np.zeros(n, dtype=int)
    # Create initial complete graph on the first m nodes.
    for i in range(m):
        for j in range(i + 1, m):
            edges.append((i, j))
            # For undirected graphs, update degrees for both nodes.
            degrees[i] += 1
            degrees[j] += 1
    # Add remaining nodes using preferential attachment.
    for new_node in range(m, n):
        # Compute attachment probabilities for existing nodes [0, new_node).
        current_degrees = degrees[:new_node]
        total_degree = np.sum(current_degrees)
        if total_degree == 0:
            probs = np.ones(new_node) / new_node
        else:
            probs = current_degrees / total_degree
        # Select m distinct nodes from existing nodes with the computed probabilities.
        chosen_nodes = rng.choice(new_node, size=m, replace=False, p=probs)
        for target in chosen_nodes:
            # Convert NumPy integer to plain int for consistency.
            target = int(target)
            edges.append((new_node, target))
            # Update degrees (for undirected graphs, update both nodes).
            degrees[new_node] += 1
            degrees[target] += 1
    # Convert numeric node identifiers to names like "v1", "v2", ...
    named_edges = [(f"v{u + 1}", f"v{v + 1}") for u, v in edges]
    return named_edges


def generate_duplication_divergence_network(n, p_retain=0.3, seed=None):
    """Generate a random network using the duplication-divergence model.

    Parameters
    ----------
    n : int
        Total number of nodes in the network (n must be at least 2).
    p_retain : float, optional
        Probability to retain each edge during duplication (default is 0.3).
    seed : int or None, optional
        Random seed for reproducibility (default is None).

    Returns:
        list of tuples
            Each edge is represented as (source, target) where source and target
            are vertex labels like "v1", "v2", ...
    """
    if n < 2:
        raise ValueError("n must be at least 2")

    rng = np.random.default_rng(seed)
    edges = []

    # Seed graph: two nodes connected by one edge
    edges.append((0, 1))

    # Grow the network by duplicating nodes
    for new_node in range(2, n):
        # Randomly choose a parent node from existing nodes [0, new_node)
        parent = rng.choice(new_node)

        # Get the set of parent's neighbors (treating edges as undirected)
        parent_neighbors = set()
        for u, v in edges:
            if u == parent:
                parent_neighbors.add(v)
            elif v == parent:
                parent_neighbors.add(u)

        # For each neighbor of the parent, add an edge from the new node with probability p_retain
        for neighbor in parent_neighbors:
            if rng.random() < p_retain:
                edges.append((new_node, neighbor))

        # Always add an edge between the new node and its parent
        edges.append((new_node, parent))

    # Convert numeric node identifiers to names like "v1", "v2", ...
    named_edges = [(f"v{u + 1}", f"v{v + 1}") for u, v in edges]
    return named_edges


def _add_signaling_signs(edges, p_inhibitory=0.3, seed=None):
    """Add signaling signs (+1/-1) to network edges.

    Parameters
    ----------
    edges : list of tuples
        List of edges as (source, target) pairs.
    p_inhibitory : float, optional
        Probability that an edge is inhibitory (-1), default is 0.3.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns:
        list of tuples
            Each edge as (source, sign, target) where sign is -1 (inhibitory)
            or +1 (activation).
    """
    rng = np.random.default_rng(seed)
    return [(src, -1 if rng.random() < p_inhibitory else 1, tgt) for src, tgt in edges]


def generate_random_signalling_network(n, m, p_inhibitory=0.3, seed=None):
    """Generate a random signaling network using a preferential attachment model.

    Parameters
    ----------
    n : int
        Total number of nodes in the network (must be greater than m).
    m : int
        Number of edges each new node attaches with, controlling network density.
    p_inhibitory : float, optional
        Probability that an edge is inhibitory (-1), default is 0.3.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns:
        list of tuples
            Each edge is represented as (source, sign, target) where:
            - source and target are vertex labels like "v1", "v2", ...
            - sign is -1 (inhibitory) or +1 (activation)

    Examples:
        >>> from corneto.data.util import generate_random_signalling_network
        >>> # Generate a small signaling network with 30% inhibitory edges
        >>> network = generate_random_signalling_network(
        ...     n=10, m=2, p_inhibitory=0.3, seed=42
        ... )
        >>> print(f"Generated {len(network)} edges")
        >>> # Extract all inhibitory interactions
        >>> inhibitory = [edge for edge in network if edge[1] == -1]
        >>> print(f"Network has {len(inhibitory)} inhibitory relationships")
    """
    base_network = generate_preferential_attachment_network(n, m, seed)
    return _add_signaling_signs(base_network, p_inhibitory, seed)


def generate_random_duplication_divergence_signalling_network(n, p_retain=0.3, p_inhibitory=0.3, seed=None):
    """Generate a random signaling network using the duplication-divergence model.

    Parameters
    ----------
    n : int
        Total number of nodes in the network (n must be at least 2).
    p_retain : float, optional
        Probability to retain each edge during duplication (default is 0.3).
    p_inhibitory : float, optional
        Probability that an edge is inhibitory (-1), default is 0.3.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns:
        list of tuples
            Each edge is represented as (source, sign, target) where:
            - source and target are vertex labels like "v1", "v2", ...
            - sign is -1 (inhibitory) or +1 (activation)
    """
    base_network = generate_duplication_divergence_network(n, p_retain, seed)
    return _add_signaling_signs(base_network, p_inhibitory, seed)
