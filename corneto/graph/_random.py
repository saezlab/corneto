"""Utility functions for data generation.

This module provides utility functions for generating random networks.

Functions
---------
generate_preferential_attachment_network : Generate a random network using the
    Barabasi-Albert preferential attachment model.
generate_duplication_divergence_network : Generate a random network using the
    duplication-divergence model.
"""

import numpy as np


def preferential_attachment_network(n, m, interactions=None, probs=None, seed=None):
    """Generate a random network using the Barabasi-Albert preferential attachment model.

    Parameters
    ----------
    n : int
        Total number of nodes in the network (must be greater than m).
    m : int
        Number of edges each new node attaches with, controlling network density.
    interactions : list or None, optional
        List of possible interaction types for edges (e.g., [1, -1]). If None,
        no interaction types will be assigned to edges.
    probs : list or None, optional
        List of probabilities for each interaction type in the 'interactions' list.
        Must sum to 1 and have the same length as 'interactions'. If None and
        interactions is provided, uniform probabilities will be used.
    seed : int or None, optional
        Random seed for reproducibility, default is None.

    Returns:
        list of tuples
            If interactions is None, each edge is represented as (source, target)
            where source and target are vertex labels like "v1", "v2", ...
            Otherwise, each edge is represented as (source, interaction, target) where
            interaction is one of the values from the interactions list.

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

    # Validate interactions and probs parameters
    if interactions is not None:
        if not isinstance(interactions, (list, tuple, np.ndarray)):
            raise ValueError("interactions must be a list, tuple, or numpy array")
        if len(interactions) == 0:
            raise ValueError("interactions list cannot be empty")

        if probs is None:
            # Use uniform probabilities if not provided
            probs = [1.0 / len(interactions)] * len(interactions)
        else:
            if len(probs) != len(interactions):
                raise ValueError("probs must have the same length as interactions")
            if not np.isclose(sum(probs), 1.0):
                raise ValueError("probabilities must sum to 1")

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
            probs_attachment = np.ones(new_node) / new_node
        else:
            probs_attachment = current_degrees / total_degree
        # Select m distinct nodes from existing nodes with the computed probabilities.
        chosen_nodes = rng.choice(new_node, size=m, replace=False, p=probs_attachment)
        for target in chosen_nodes:
            # Convert NumPy integer to plain int for consistency.
            target = int(target)
            edges.append((new_node, target))
            # Update degrees (for undirected graphs, update both nodes).
            degrees[new_node] += 1
            degrees[target] += 1

    # Convert numeric node identifiers to names like "v1", "v2", ...
    if interactions is not None:
        named_edges = []
        for u, v in edges:
            interaction_type = rng.choice(interactions, p=probs)
            named_edges.append((f"v{u + 1}", interaction_type, f"v{v + 1}"))
        return named_edges
    else:
        return [(f"v{u + 1}", f"v{v + 1}") for u, v in edges]


def duplication_divergence_network(n, p_retain=0.3, interactions=None, probs=None, seed=None):
    """Generate a random network using the duplication-divergence model.

    Parameters
    ----------
    n : int
        Total number of nodes in the network (n must be at least 2).
    p_retain : float, optional
        Probability to retain each edge during duplication (default is 0.3).
    interactions : list or None, optional
        List of possible interaction types for edges (e.g., [1, -1]). If None,
        no interaction types will be assigned to edges.
    probs : list or None, optional
        List of probabilities for each interaction type in the 'interactions' list.
        Must sum to 1 and have the same length as 'interactions'. If None and
        interactions is provided, uniform probabilities will be used.
    seed : int or None, optional
        Random seed for reproducibility (default is None).

    Returns:
        list of tuples
            If interactions is None, each edge is represented as (source, target)
            where source and target are vertex labels like "v1", "v2", ...
            Otherwise, each edge is represented as (source, interaction, target) where
            interaction is one of the values from the interactions list.
    """
    if n < 2:
        raise ValueError("n must be at least 2")

    rng = np.random.default_rng(seed)
    edges = []

    # Validate interactions and probs parameters
    if interactions is not None:
        if not isinstance(interactions, (list, tuple, np.ndarray)):
            raise ValueError("interactions must be a list, tuple, or numpy array")
        if len(interactions) == 0:
            raise ValueError("interactions list cannot be empty")

        if probs is None:
            # Use uniform probabilities if not provided
            probs = [1.0 / len(interactions)] * len(interactions)
        else:
            if len(probs) != len(interactions):
                raise ValueError("probs must have the same length as interactions")
            if not np.isclose(sum(probs), 1.0):
                raise ValueError("probabilities must sum to 1")

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
    if interactions is not None:
        named_edges = []
        for u, v in edges:
            interaction_type = rng.choice(interactions, p=probs)
            named_edges.append((f"v{u + 1}", interaction_type, f"v{v + 1}"))
        return named_edges
    else:
        return [(f"v{u + 1}", f"v{v + 1}") for u, v in edges]
