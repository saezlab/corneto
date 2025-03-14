"""Utility functions for data generation and manipulation.

This module provides utility functions for generating and manipulating
data structures for use with the corneto data package.

Functions
---------
generate_random_signalling_network : Generate a random biological signaling network
    with configurable properties such as inhibitory edge probability.
"""
import numpy as np


def generate_random_signalling_network(n, m, p_inhibitory=0.3, seed=None):
    """Generate a random signaling network using a preferential attachment model.
    
    Creates a network with biologically-inspired properties, where each edge represents
    a regulatory relationship (activation or inhibition) between genes or proteins.
    The network follows a scale-free topology commonly observed in biological systems.
    
    Parameters
    ----------
    n : int
        Total number of nodes in the network (must be greater than m).
    m : int
        Number of edges each new node attaches with, controlling network density.
    p_inhibitory : float, optional
        Probability that an edge is inhibitory (-1), default is 0.3.
        Values closer to 0 create more activation-dominated networks.
    seed : int or None, optional
        Random seed for reproducibility, default is None.
    
    Returns
    -------
    list of tuples
        Each edge is represented as (source, sign, target) where:
        - source and target are vertex labels like "v1", "v2", ...
        - sign is -1 (inhibitory) or +1 (activation)
    
    Examples
    --------
    >>> from corneto.data.util import generate_random_signalling_network
    >>> # Generate a small signaling network with 30% inhibitory edges
    >>> network = generate_random_signalling_network(n=10, m=2, p_inhibitory=0.3, seed=42)
    >>> print(f"Generated {len(network)} edges")
    >>> # Extract all inhibitory interactions
    >>> inhibitory = [edge for edge in network if edge[1] == -1]
    >>> print(f"Network has {len(inhibitory)} inhibitory relationships")
    
    Notes
    -----
    The algorithm implements a Barabási–Albert preferential attachment model,
    which produces scale-free networks similar to those observed in biological systems.
    The initial network consists of a complete graph among the first m nodes.
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
            sign = -1 if rng.random() < p_inhibitory else 1
            edges.append((i, sign, j))
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
            sign = -1 if rng.random() < p_inhibitory else 1
            edges.append((new_node, sign, target))
            # Update degrees (for undirected graphs, update both nodes).
            degrees[new_node] += 1
            degrees[target] += 1
    # Convert numeric node identifiers to names like "v1", "v2", ...
    named_edges = [(f"v{u + 1}", sign, f"v{v + 1}") for u, sign, v in edges]
    return named_edges
