"""Utility functions"""

import random
from typing import Optional

import numpy as np


def seed_everything(seed: int, verbose: bool = False):
    """Seed everything."""
    np.random.seed(seed)
    random.seed(seed)
    if verbose:
        print(f"Seeded everything with {seed}")


def compute_direct_paths(adj_matrix: np.ndarray) -> np.ndarray:
    """Matrix of direct paths."""
    n_nodes = adj_matrix.shape[0]
    # set 1 if there is an arc, zero otherwise
    adj_matrix = np.where(np.abs(adj_matrix) > 0, 1, 0)

    mechanism = np.zeros((n_nodes, n_nodes))
    cumulator = np.eye(n_nodes)
    mechanism += cumulator
    for _ in range(n_nodes):
        cumulator = cumulator @ adj_matrix
        cumulator = np.where(cumulator > 0, 1, 0)
        mechanism += cumulator

    mechanism = np.where(mechanism > 0, 1, 0)

    return mechanism


def compute_mechanism(adj_matrix: np.ndarray) -> np.ndarray:
    """Reduced form of the linear ANM."""
    n_nodes = adj_matrix.shape[0]
    mechanism = np.zeros((n_nodes, n_nodes))
    cumulator = np.eye(n_nodes)
    mechanism += cumulator
    for _ in range(n_nodes):
        cumulator = cumulator @ adj_matrix
        mechanism += cumulator
    return mechanism


def check_cancelling_paths(adj_matrix: np.ndarray, min_abs: float = 1e-5) -> bool:
    """
    In a linear SCM, the model does not have cancelling paths
    whenever the direct paths coincide with the dependencies
    of the mechanism.
    """
    return np.allclose(
        compute_direct_paths(adj_matrix),
        np.abs(compute_mechanism(adj_matrix)) > min_abs,
    )


def linear_anm(
    weights: np.ndarray,
    exogenous: np.ndarray,
    intervention: Optional[dict] = None,
) -> np.ndarray:
    """Linear Additive Noise Model (ANM)"""
    endogenous = np.zeros_like(exogenous)
    n_nodes = weights.shape[0]
    assert weights.shape == (n_nodes, n_nodes)

    # max number of steps
    for _ in range(n_nodes):
        # update endogenous
        endogenous = endogenous @ weights + exogenous

        # apply intervention
        if intervention is not None:
            for target, value in intervention.items():
                endogenous[:, target] = value

    return endogenous


def block_slice(node: int, blocks: list[list[int]]) -> slice:
    """Slices the block."""
    start, end = min(blocks[node]), max(blocks[node]) + 1
    # Check that the block is contiguous
    assert list(range(start, end)) == blocks[node]
    return slice(start, end)
