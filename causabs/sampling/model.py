"""Sampling of models, abstractions, and datasets."""

import igraph as ig  # type: ignore
import numpy as np


def sample_dag(
    num_nodes: int,
    num_edges: int,
    graph_type: str,
    ordered: bool = False,
) -> np.ndarray:
    """
    Samples a random DAG with the expected number of edges,
    originally taken from https://github.com/xunzheng/notears.
    """

    def _random_permutation(matrix: np.ndarray) -> np.ndarray:
        # np.random.permutation permutes first axis only
        p_matrix = np.random.permutation(np.eye(matrix.shape[0]))
        return p_matrix.T @ matrix @ p_matrix

    def _random_acyclic_orientation(b_und):
        return np.tril(_random_permutation(b_und), k=-1)

    def _graph_to_adjmat(graph: ig.Graph) -> np.ndarray:
        return np.array(graph.get_adjacency().data)

    if graph_type == "ER":
        # Erdos-Renyi
        graph = ig.Graph.Erdos_Renyi(n=num_nodes, m=num_edges)
        binary = _graph_to_adjmat(graph)
        binary = _random_acyclic_orientation(binary)
    elif graph_type == "SF":
        # Scale-free, Barabasi-Albert
        graph = ig.Graph.Barabasi(
            n=num_nodes, m=int(round(num_edges / num_nodes)), directed=True
        )
        binary = _graph_to_adjmat(graph)
    elif graph_type == "BP":
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * num_nodes)
        graph = ig.Graph.Random_Bipartite(
            top, num_nodes - top, m=num_edges, directed=True, neimode=ig.OUT
        )
        binary = _graph_to_adjmat(graph)
    else:
        raise ValueError("unknown graph type")

    # Random permutation of the node labels
    binary = _random_permutation(binary)
    graph = ig.Graph.Weighted_Adjacency(binary.tolist())

    if ordered:
        # Get topological ordering
        order = graph.topological_sorting()
        # Reorder adjacency matrix
        binary = binary[order, :][:, order]
        graph = ig.Graph.Weighted_Adjacency(binary.tolist())

    return np.array(graph.get_adjacency().data)


def sample_uniform_coeffs(
    graph: np.ndarray,
    coeff_ranges: tuple[tuple[float, float], tuple[float, float]] = (
        (-3.0, -0.2),
        (0.2, 3.0),
    ),
) -> np.ndarray:
    """Given a graph as a binary adjacency matrix, returns a weight matrix"""
    weight_adj = np.zeros(graph.shape)
    sign_mat = np.random.randint(len(coeff_ranges), size=graph.shape)
    for i, (low, high) in enumerate(coeff_ranges):
        unif_sample = np.random.uniform(low=low, high=high, size=graph.shape)
        weight_adj += graph * (sign_mat == i) * unif_sample
    return weight_adj


def sample_normal_coeffs(
    graph: np.ndarray, mean: float = 1.5, min_abs: float = 0.2
) -> np.ndarray:
    """Given a graph, returns a weight matrix"""

    negative_coeffs = np.random.normal(loc=-mean, size=graph.shape) / np.sqrt(
        graph.shape[0]
    )
    negative_coeffs = np.where(negative_coeffs < -min_abs, negative_coeffs, -min_abs)
    positive_coeffs = np.random.normal(loc=mean, size=graph.shape) / np.sqrt(
        graph.shape[0]
    )
    positive_coeffs = np.where(positive_coeffs > min_abs, positive_coeffs, min_abs)

    positive_sign = np.random.choice([0, 1], size=graph.shape)

    coeffs = graph * (
        positive_sign * positive_coeffs + (1 - positive_sign) * negative_coeffs
    )

    return coeffs
