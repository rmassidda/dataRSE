"""Causal inference utilities: TCE estimation and adjustment-set identification."""

import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression

from causabs.graph import abstract_graph


def path_coefficients_tce(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    coeffs: np.ndarray,
) -> float:
    """Estimate the TCE (Total Causal Effect) of treatment on outcome using path coefficients."""
    all_paths = list(nx.all_simple_paths(dag, source=treatment, target=outcome))
    total_effect = 0.0
    for path in all_paths:
        path_effect = 1.0
        for x1, x2 in zip(path, path[1:]):
            path_effect *= coeffs[x1, x2]
        total_effect += path_effect
    return total_effect


def get_optimal_adjustment_set(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
) -> set[int]:
    """
    Identifies the optimal adjustment set for estimating the causal effect
    of treatment on outcome in a given DAG.
    Adapted from https://github.com/Matyasch/load/blob/main/load.py,
    also see:
    https://www.stats.ox.ac.uk/~evans/APTS/effadj.html#optimal-adjustment-set
    """
    descendants = nx.descendants(dag, treatment)
    ancestors = nx.ancestors(dag, outcome)
    mediators = descendants.intersection(ancestors)
    oset = set()
    for v in mediators | {outcome}:
        oset |= set(dag.predecessors(v))
    oset -= mediators | {treatment}
    return oset


def abstract_optimal_adjustment_tce(
    cnc_graph: nx.DiGraph,
    cnc_data: np.ndarray,
    relevants: list[list[int]],
    tau: dict[int, np.ndarray],
    treatment: int,
    outcome: int,
) -> float:
    """Estimate TCE using the optimal adjustment set directly on the abstract graph."""
    abs_graph = abstract_graph(cnc_graph, relevants)
    oset = get_optimal_adjustment_set(abs_graph, treatment, outcome)
    abs_data = np.zeros((cnc_data.shape[0], abs_graph.number_of_nodes()))
    for abs_node in abs_graph.nodes:
        R_j = relevants[abs_node]
        abs_data[:, abs_node] = cnc_data[:, R_j] @ tau[abs_node]
    X = abs_data[:, [treatment] + list(oset)]
    y = abs_data[:, outcome]
    lr = LinearRegression()
    lr.fit(X, y)
    return lr.coef_[0]
