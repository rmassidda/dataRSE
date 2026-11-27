"""Logging and basic graph utilities."""

import logging

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def log(msg: str) -> None:
    """Italic and yellow — delegates to the module logger at DEBUG level."""
    logger.debug("\033[3m\033[33m%s\033[0m", msg)


def digraph(nodes, edges) -> nx.DiGraph:
    """Construct a DiGraph from explicit node and edge lists."""
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def tau_coeffs_to_dict(
    tau_coeffs: np.ndarray,
    relevants: list[list[int]],
) -> dict[int, np.ndarray]:
    """Convert a tau coefficient matrix to a per-abstract-node dict.

    ``tau_coeffs[i, j]`` is the loading of concrete node *i* onto abstract
    node *j*.  Returns a dict mapping each abstract-node index to the
    coefficient vector over its relevant concrete nodes.
    """
    return {
        abs_node: tau_coeffs[relevants[abs_node], abs_node]
        for abs_node in range(len(relevants))
    }


def compare_solutions(sol: list[list[int]], gt: list[list[int]]) -> bool:
    """Return True if sol matches gt, i.e. each group in gt is contained in
    some group in sol."""
    if len(sol) != len(gt):
        return False
    sol_r = [frozenset(ry) for ry in sol]
    gt_r = [frozenset(ry) for ry in gt]
    return all(gt_group in sol_r for gt_group in gt_r)

def compare_subsolution(sol: list[list[int]], gt: list[list[int]]) -> bool:
    """Return True if sol is a subset of gt, i.e. each group in sol is contained in
    some group in gt. In other terms, we check if the solution is a further partitioning of the ground truth."""
    sol_r = [frozenset(ry) for ry in sol]
    gt_r = [frozenset(ry) for ry in gt]
    for sol_group in sol_r:
        if not any(sol_group.issubset(gt_group) for gt_group in gt_r):
            return False
    return True
