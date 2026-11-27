"""Ground-truth sampling for abstract/concrete causal graph experiments."""

from __future__ import annotations

import networkx as nx
import numpy as np

from causabs.sampling.concretization import (
    sample_blocks,
    sample_from_abstract_graph_relevants,
    sample_inter_block_edges,
    sample_intra_block_edges,
    sample_relevants_min_max,
)

from ._config import load_graph
from ._utils import log, tau_coeffs_to_dict


def sample_ground_truth(
    abs_code: str,
    min_block_size: int,
    max_block_size: int,
    min_relevant_size: int,
    max_relevant_size: int,
    edge_ratio: float = 0.5,
    output_ratio: float = 0.2,
    tau_norm: bool = True,
) -> tuple[
    nx.DiGraph,
    nx.DiGraph,
    list[list[int]],
    np.ndarray,
    np.ndarray,
    dict[int, np.ndarray],
    int,
    int,
]:
    """
    Sample a ground-truth abstract/concrete causal graph pair.

    Raises ValueError if sampling fails due to a degenerate graph configuration.
    """
    # High-Level Graph
    abs_graph, treatment_abs, outcome_abs = load_graph(abs_code)
    abs_graph_bin = nx.adjacency_matrix(abs_graph).toarray()

    log(f"Selected treatment: {treatment_abs}, outcome: {outcome_abs}")

    # Sample blocks and relevants
    blocks_gt = sample_blocks(
        abs_graph.number_of_nodes(), min_block_size, max_block_size
    )
    relevants_gt = sample_relevants_min_max(
        blocks_gt, min_relevant_size, max_relevant_size
    )

    # Sample concrete graph
    cnc_graph_bin = sample_intra_block_edges(
        blocks_gt, relevants_gt, edge_ratio=edge_ratio
    )
    cnc_graph_bin += sample_inter_block_edges(
        abs_graph_bin, blocks_gt, relevants_gt, output_ratio=output_ratio
    )
    cnc_graph = nx.DiGraph(cnc_graph_bin)

    # Sample coefficients (_blocks_gt unused beyond this call)
    cnc_coeff_gt, abs_coeff_gt, tau_coeff_gt, _ = sample_from_abstract_graph_relevants(
        abs_graph,
        cnc_graph,
        relevants_gt,
        blocks_gt,
        normalize=tau_norm,
    )

    return (
        abs_graph,
        cnc_graph,
        relevants_gt,
        cnc_coeff_gt,
        abs_coeff_gt,
        tau_coeffs_to_dict(tau_coeff_gt, relevants_gt),
        treatment_abs,
        outcome_abs,
    )

