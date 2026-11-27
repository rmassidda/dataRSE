"""
Identifiability-margin sampling for ground-truth abstract graphs.

Wraps ``sampling.sample_ground_truth`` with a rejection step that enforces
MaxSim(M, G_H) < gamma, where MaxSim mirrors the recursion of dataRSE over
G_H (\\Cref{app:datagen}).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .sampling import sample_ground_truth


def _abstract_targets(
    abs_graph: nx.DiGraph, y: int, known: set[int]
) -> set[int]:
    """Known variables reachable from y through unknown variables only.

    Abstract counterpart of the abstract targets of the relevant variables
    of y: a directed path stops at the first known variable it reaches.
    """
    targets: set[int] = set()
    visited: set[int] = {y}
    stack = [y]
    while stack:
        node = stack.pop()
        for child in abs_graph.successors(node):
            if child in known:
                targets.add(child)
            elif child not in visited:
                visited.add(child)
                stack.append(child)
    return targets


def max_similarity(
    abs_graph: nx.DiGraph, abs_coeff: np.ndarray, given: set[int]
) -> float:
    """
    MaxSim(M, G_H): mirror the recursion of dataRSE over the ground-truth
    abstract graph, starting from the given variables, and return the
    largest absolute cosine similarity checked by the identifiability
    condition of \\Cref{thm:maximalpartitioning}.

    At each step, with `known` the variables recovered so far:
        - an unknown variable is intersected if it targets at least two
          known variables, and V is the union of these targets;
        - the new parents are the intersected variables with a known child,
          i.e., the variables whose relevant sets lie in Z;
        - S are the new parents whose children are all in V.
    The condition requires the outgoing coefficients towards V of each
    variable in S to be non-collinear with those of every other new parent.
    Pairs of new parents outside S are not checked: their merges are
    discarded by dataRSE and resolved at a later step. Then, only S is
    added to the known variables, as dataRSE does.
    """
    known = set(given)
    sims: list[float] = []
    while True:
        unknown = set(abs_graph.nodes()) - known
        targets = {y: _abstract_targets(abs_graph, y, known) for y in unknown}
        intersected = {y for y in unknown if len(targets[y]) >= 2}
        if not intersected:
            break

        v_set = set().union(*(targets[y] for y in intersected))
        v_list = sorted(v_set)
        new_parents = [
            y for y in intersected if set(abs_graph.successors(y)) & known
        ]
        s_list = [
            y for y in new_parents if set(abs_graph.successors(y)) <= v_set
        ]
        if not s_list:
            break

        rows = {}
        for y in new_parents:
            row = abs_coeff[y, v_list]
            norm = np.linalg.norm(row)
            if norm > 0:
                rows[y] = row / norm
        for s in s_list:
            for h in new_parents:
                if h != s and s in rows and h in rows:
                    sims.append(float(np.abs(rows[s] @ rows[h])))

        known.update(s_list)
    return max(sims) if sims else 0.0


def sample_identifiable_ground_truth(
    abs_code: str,
    block_size: int,
    relevant_size: int,
    edge_ratio: float = 0.5,
    output_ratio: float = 0.2,
    max_sim: float = 0.9,
    tau_norm: bool = True,
    repeat_index: int = 0,
    max_trials: int = 100,
) -> tuple[
    nx.DiGraph,
    nx.DiGraph,
    list[list[int]],
    np.ndarray,
    np.ndarray,
    dict[int, np.ndarray],
    int,
    int,
    float,
]:
    """
    Like ``sample_ground_truth``, but resamples the high-level coefficients
    up to ``max_trials`` times to enforce the identifiability margin
    MaxSim(M, G_H) < ``max_sim`` (\\Cref{app:datagen}). Falls back to the
    last sampled instance, with a warning, if no resample satisfies the
    threshold within the budget.

    Raises ValueError if sampling fails due to a degenerate graph
    configuration.

    Returns the same tuple as ``sample_ground_truth``, plus the achieved
    MaxSim value.
    """
    n_trial = 0
    while True:
        (
            abs_graph,
            cnc_graph,
            relevants_gt,
            cnc_coeff_gt,
            abs_coeff_gt,
            tau_coeff_gt,
            treatment_abs,
            outcome_abs,
        ) = sample_ground_truth(
            abs_code,
            min_block_size=block_size,
            max_block_size=block_size,
            min_relevant_size=relevant_size,
            max_relevant_size=relevant_size,
            edge_ratio=edge_ratio,
            output_ratio=output_ratio,
            tau_norm=tau_norm,
        )

        achieved_max_sim = max_similarity(
            abs_graph, abs_coeff_gt, {treatment_abs, outcome_abs}
        )

        n_trial += 1
        if achieved_max_sim < max_sim:
            break
        if n_trial >= max_trials:
            print(
                f"WARNING: high max_sim={achieved_max_sim:.4f} "
                f"({relevant_size}, {block_size}) repeat={repeat_index}"
            )
            break

    return (
        abs_graph,
        cnc_graph,
        relevants_gt,
        cnc_coeff_gt,
        abs_coeff_gt,
        tau_coeff_gt,
        treatment_abs,
        outcome_abs,
        achieved_max_sim,
    )
