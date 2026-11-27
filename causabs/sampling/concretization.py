"""
Copyright (c) 2024 Riccardo Massidda
Released under the MIT license, see LICENSE.
"""

import networkx as nx  # type: ignore
import numpy as np
from igraph._igraph import InternalError  # type: ignore

from causabs.sampling.model import (
    sample_dag,
    sample_normal_coeffs,
    sample_uniform_coeffs,
)

from ..utils import block_slice, check_cancelling_paths, compute_mechanism


def sample_blocks(
    abs_nodes: int,
    min_block_size: int = 2,
    max_block_size: int = 5,
    ignored: bool = True,
) -> list[list[int]]:
    """
    Samples blocks of concrete variables for each
    of the provided abstract variable. Always returns
    a further block of ignored variables, eventually
    empty.

    Parameters
    ----------
    abs_nodes : int
        Number of abstract nodes.
    min_block_size : int
        Minimum size of the block.
    max_block_size : int
        Maximum size of the block.
    ignored : bool
        Whether to include an ignored block.

    Returns
    -------
    list[list[int]]
        List of blocks for each of the abstract nodes.
    """
    # Sample the block sizes (the last block contains the ignored variables)
    block_size = [
        np.random.randint(min_block_size, max_block_size + 1)
        for _ in range(abs_nodes + 1)
    ]

    # Eventually remove ignored variables
    if not ignored:
        block_size[-1] = 0

    # Concrete model pointers
    block_start = np.cumsum([0] + block_size[:-1])
    blocks = [
        list(range(block_start[node], block_start[node] + block_size[node]))
        for node in range(abs_nodes + 1)
    ]

    return blocks


def sample_relevants_min_max(
    blocks: list[list[int]],
    min_relevant_size: int,
    max_relevant_size: int,
) -> list[list[int]]:
    # Do not sample relevants for the ignored block
    blocks = blocks[:-1]

    relevants = []
    for _, block in enumerate(blocks):
        # Number of irrelevant variables to sample
        n_relevant = np.random.randint(min_relevant_size, max_relevant_size + 1)
        n_irrelevant = len(block) - n_relevant
        # Sample indices of the irrelevant variables (last one always relevant)
        irrelevant = sorted(
            np.random.choice(len(block) - 1, n_irrelevant, replace=False)
        )
        # Select variables whose index is not irrelevant
        relevants.append([block[i] for i in range(len(block)) if i not in irrelevant])

    return relevants


def sample_intra_block_edges(
    blocks: list[list[int]],
    relevants: list[list[int]],
    edge_ratio: float = 0.5,
    max_trial: int = 100,
    safe_mode: bool = False,
) -> np.ndarray:
    """
    Given the blocks, the function samples the edges
    within the blocks. It assumes that the intra-block
    edges respect necessary conditions.
    """

    cnc_nodes = sum(len(block) for block in blocks)
    cnc_graph = np.zeros((cnc_nodes, cnc_nodes))
    abs_nodes = len(blocks) - 1

    # Internally connect ignored variables
    if len(blocks[-1]) > 0:
        block = block_slice(-1, blocks)
        block_size = len(blocks[-1])
        try:
            cnc_graph[block, block] = sample_dag(
                block_size, int(block_size * edge_ratio), graph_type="ER", ordered=True
            )
        except InternalError:
            pass

    def check_edge_ratio(subgraph: nx.DiGraph):
        n_edges = len(subgraph.edges())
        n_nodes = subgraph.number_of_nodes()
        return n_edges <= n_nodes * edge_ratio

    def check_consistency(subgraph: nx.DiGraph, node: int):
        # True if each non-relevant variable is
        # connected to at least one relevant variable
        for idx, block_var in enumerate(blocks[node]):
            if block_var in relevants[node]:
                continue
            connected_rels = [
                blocks[node].index(rel_var)
                for rel_var in relevants[node]
                if nx.has_path(subgraph, idx, blocks[node].index(rel_var))
            ]
            if len(connected_rels) == 0:
                return False

        return True

    for node in range(abs_nodes):
        # Fully connected sub-graph
        block = block_slice(node, blocks)
        block_size = len(blocks[node])
        subgraph: nx.DiGraph = nx.from_numpy_array(
            np.triu(np.ones((block_size, block_size)), k=1),
            create_using=nx.DiGraph,
        )

        if safe_mode:
            assert check_consistency(subgraph, node)

        trial = 0
        while not check_edge_ratio(subgraph) and trial < max_trial:
            # Sample a random edge from .edges()
            edges = list(subgraph.edges())
            edge_idx = int(np.random.randint(len(edges)))
            edge = edges[edge_idx]

            # Remove the edge
            subgraph.remove_edge(*edge)

            # Check if properties still hold
            if not check_consistency(subgraph, node):
                subgraph.add_edge(*edge)
                trial += 1

        # Update the adjacency matrix
        cnc_graph[block, block] = nx.to_numpy_array(subgraph)

    return cnc_graph


def sample_inter_block_edges(
    abs_graph: np.ndarray,
    blocks: list[list[int]],
    relevants: list[list[int]],
    output_ratio: float = 0.2,
) -> np.ndarray:
    """
    Given the blocks and the relevant variables, it returns the adjacency matrix containing only the edges across the blocks. It assumes that the intra-block edges respect necessary conditions.

    Each relevant variable must have at least one outgoing edge to the next block. The number of sampled edges is controlled by the output_ratio parameter, which sets the number of outgoing edges per relevant variable as a ratio of the target block size. For example, if output_ratio=0.2 and the target block has 10 variables, each relevant variable will have at most 2 outgoing edges to the target block.

    Parameters
    ----------
    abstract_graph: AdjacencyMatrix
        Adjacency matrix of the abstract graph.
    blocks : list[list[int]]
        List of blocks for each of the abstract nodes.
    relevant : list[list[int]]
        List of relevant variables for each of the abstract nodes.
    output_ratio : float
        Ratio of the maximum number of outgoing edges per relevant variable.

    Returns
    -------
    AdjacencyMatrix
        Adjacency matrix of the concrete graph.
    """

    cnc_nodes = sum(len(block) for block in blocks)
    cnc_graph = np.zeros((cnc_nodes, cnc_nodes))
    abs_nodes = len(blocks) - 1

    if output_ratio <= 0 or output_ratio > 1:
        raise ValueError("output_ratio must be in (0, 1].")

    for source in range(abs_nodes):
        relevant_i = relevants[source]
        for target in range(abs_nodes):
            # Skip if no abstract edge
            if not abs_graph[source, target]:
                continue
            block_j = blocks[target]
            n_sources = len(relevant_i)
            n_targets = len(block_j)

            # We code edges (i, j) as i * n_targets + j in a flat array.

            # Phase 1: one mandatory edge per relevant source node.
            mandatory = np.array(
                [r * n_targets + np.random.randint(n_targets) for r in range(n_sources)]
            )
            # Phase 2: distribute remaining budget over the rest of
            # the bipartite graph, excluding the mandatory indices.
            budget = max(0, int(n_targets * output_ratio) - n_sources)
            remaining = np.setdiff1d(np.arange(n_sources * n_targets), mandatory)
            edges = np.random.choice(
                remaining, size=min(budget, len(remaining)), replace=False
            )
            edges = np.concatenate([mandatory, edges])
            for idx in edges:
                xi = relevant_i[idx // n_targets]
                xj = block_j[idx % n_targets]
                cnc_graph[xi, xj] = 1

        # Connect to ignored variables
        block_ignored = blocks[abs_nodes]
        if block_ignored:
            block_i = blocks[source]
            block_j = blocks[-1]
            n_sources = len(block_i)
            n_targets = len(block_j)
            budget = int(n_targets * output_ratio)
            edges = np.random.choice(n_sources * n_targets, size=budget, replace=False)
            for idx in edges:
                xi = block_i[idx // n_targets]
                xj = block_j[idx % n_targets]
                cnc_graph[xi, xj] = 1

    return cnc_graph


def sample_concrete_coeffs(
    abs_coeff: np.ndarray,
    cnc_graph: np.ndarray,
    tau_coeff: np.ndarray,
    blocks: list[list[int]],
    relevants: list[list[int]],
    alpha: float = 1e4,
    style: str = "positive",
    inner: str = "normal",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given the concrete graph, the abstract model
    and the abstraction function, it return the
    concrete model and the corresponding exogenous
    abstraction function.
    """
    # Number of nodes
    abs_nodes = abs_coeff.shape[0]
    cnc_nodes = tau_coeff.shape[0]

    # Helper to get slices
    def _block(node: int):
        return block_slice(node, blocks)

    # Initialization
    cnc_coeff = np.zeros((cnc_nodes, cnc_nodes))
    cnc_mech = np.zeros((cnc_nodes, cnc_nodes))
    res_coeff = np.zeros((cnc_nodes, cnc_nodes))
    rho_coeff = np.zeros((abs_nodes, abs_nodes))
    gamma_coeff = np.zeros_like(tau_coeff)

    # Abstract mechanism
    abs_mech = compute_mechanism(abs_coeff)

    # Intra-block (mechanisms and exog. abstraction)
    for node in range(abs_nodes):
        block = _block(node)

        # Sample internal weights
        if inner == "normal":
            sample_inner_coeffs = sample_normal_coeffs
        elif inner == "uniform":
            sample_inner_coeffs = sample_uniform_coeffs  # type: ignore
        else:
            raise ValueError(f"Unknown inner style {inner}.")

        w_ii = sample_inner_coeffs(cnc_graph[block, block])

        # Compute internal mechanism
        f_ii = compute_mechanism(w_ii)

        # Compute exogenous abstraction
        t_i = tau_coeff[block, node]
        s_i = f_ii @ t_i

        # Assign results
        cnc_coeff[block, block] = w_ii
        cnc_mech[block, block] = f_ii
        gamma_coeff[block, node] = s_i

    # Check abstract model faithfulness
    if not check_cancelling_paths(abs_coeff):
        raise ValueError("The abstract model is not faithful.")

    # Inter-block Mechanism
    for target in range(abs_nodes):
        # Target params
        block_j = _block(target)
        s_j = gamma_coeff[block_j, target]

        for source in reversed(range(target)):
            # Source params
            block_i = _block(source)
            t_i = tau_coeff[block_i, source]

            # Source-target params
            g_ij = abs_mech[source, target]
            m_ij = abs_coeff[source, target]

            # Update the remainder
            for k in range(source + 1, target):
                block_k = _block(k)
                w_ik = cnc_coeff[block_i, block_k]
                w_kj = cnc_coeff[block_k, block_j]
                r_kj = res_coeff[block_k, block_j]
                f_kk = cnc_mech[block_k, block_k]
                m_ik = abs_coeff[source, k]
                m_kj = abs_coeff[k, target]
                p_kj = rho_coeff[k, target]
                res_coeff[block_i, block_j] += w_ik @ f_kk @ (w_kj + r_kj)
                rho_coeff[source, target] += m_ik * (m_kj + p_kj)

            # Get the remainder
            r_ij = res_coeff[block_i, block_j]

            # Relevant sources
            for x_source in relevants[source]:
                # Get the block index
                x_source_idx = blocks[source].index(x_source)

                # Get the targets from the concrete graph.
                # We store the relative index within the target block.
                cnc_targets = []
                for x_target in blocks[target]:
                    x_target_idx = blocks[target].index(x_target)
                    if cnc_graph[block_i, block_j][x_source_idx, x_target_idx]:
                        cnc_targets.append(x_target_idx)

                if not cnc_targets:
                    # If there are no targets for a source,
                    # there is no abstract edge from the
                    # source to the target.
                    break


                # Assignment vector
                v = np.zeros_like(s_j)
                if style == "uniform":
                    dist = np.array(1 / len(cnc_targets))
                elif style == "positive":
                    dist = np.random.dirichlet(np.ones(len(cnc_targets)) * alpha)
                else:
                    raise ValueError(f"Unknown style {style}.")

                # Right-inverse of s_j
                s_j_pinv = np.zeros_like(s_j)
                v[cnc_targets] = dist
                s_j_pinv[cnc_targets] = v[cnc_targets] / s_j[cnc_targets]

                # compute block
                cnc_coeff[block_i, block_j][x_source_idx, :] = (
                    m_ij * t_i[x_source_idx] * s_j_pinv
                )

            # check weights closed form
            w_ij = cnc_coeff[block_i, block_j]
            s_j = s_j.reshape((len(s_j), 1))
            t_i = t_i.reshape((len(t_i), 1))
            if not np.allclose(w_ij @ s_j, g_ij * t_i - r_ij @ s_j):
                print("WARNING", source, target)
                print(w_ij @ s_j)
                print(g_ij * t_i - r_ij @ s_j)
                print("w_ij", w_ij)
                print("s_j", s_j)
                print("t_i", t_i)
                print("g_ij", g_ij)
                print("r_ij", r_ij)
                print("s_j", s_j)
            assert np.allclose(w_ij @ s_j, g_ij * t_i - r_ij @ s_j)

            s_j = s_j.reshape((len(s_j),))

    # Eventually handle ignored variables
    if len(blocks[-1]) > 0:
        # From each variable to ignored
        block_ignored = _block(abs_nodes)
        size_ignored = len(blocks[-1])
        for node in range(abs_nodes):
            block = _block(node)
            size_block = len(blocks[node])
            # random weights from block_y to block_ignored
            cnc_coeff[block, block_ignored] = (
                np.random.normal(size=(size_block, size_ignored))
                / np.sqrt(size_block)
                * cnc_graph[block, block_ignored]
            )

        # Internal ignored connections
        cnc_coeff[block_ignored, block_ignored] = (
            np.random.normal(size=(size_ignored, size_ignored))
            / np.sqrt(size_ignored)
            * cnc_graph[block_ignored, block_ignored]
        )

    # Check cancelling paths
    if not check_cancelling_paths(cnc_coeff):
        raise ValueError("The concrete model is not faithful.")

    # Check properties
    cnc_mech_bis = compute_mechanism(cnc_coeff)
    for node in range(abs_nodes):
        block_i = _block(node)
        f_ii = cnc_mech[block_i, block_i]
        f_ii_bis = cnc_mech_bis[block_i, block_i]
        if not np.allclose(f_ii, f_ii_bis):
            print("WARNING (---)", node)
            print(f_ii)
            print(f_ii_bis)
    cnc_mech = cnc_mech_bis

    for target in range(abs_nodes):
        for source in range(target):
            p_ij = rho_coeff[source, target]
            m_ij = abs_coeff[source, target]
            g_ij = abs_mech[source, target]
            if not np.allclose(g_ij, m_ij + p_ij):
                print(f"WARNING {source} -> {target} ({g_ij} != {m_ij} + {p_ij})")

    for target in range(abs_nodes):
        for source in range(target):
            block_i = _block(source)
            block_j = _block(target)

            # Consistency of the mechanisms: F_ij t_j = s_i g_ij
            f_ii = cnc_mech[block_i, block_i]
            f_ij = cnc_mech[block_i, block_j]
            t_j = tau_coeff[block_j, target]
            s_i = gamma_coeff[block_i, source]
            g_ij = abs_mech[source, target]
            if not np.allclose(f_ij @ t_j, s_i * g_ij):
                print("WARNING (F_ij t_j = s_i g_ij)", source, target)
                print(f_ij @ t_j)
                print(s_i * g_ij)

            # F_ii (W_ij + R_ij) F_jj t_j = F_ii t_i (m_ij + p_ij)
            f_ii = cnc_mech[block_i, block_i]
            w_ij = cnc_coeff[block_i, block_j]
            r_ij = res_coeff[block_i, block_j]
            f_jj = cnc_mech[block_j, block_j]
            m_ij = abs_coeff[source, target]
            p_ij = rho_coeff[source, target]
            t_i = tau_coeff[block_i, source]
            if not np.allclose(
                f_ii @ (w_ij + r_ij) @ f_jj @ t_j, f_ii @ t_i * (m_ij + p_ij)
            ):
                print("WARNING (F_ii (W_ij + R_ij) F_jj t_j = F_ii t_i (m_ij + p_ij))", source, target)
                print(f_ii @ (w_ij + r_ij) @ f_jj @ t_j)
                print(f_ii @ t_i * (m_ij + p_ij))

            # (W_ij + R_ij) s_j = t_i (m_ij + p_ij)
            s_j = gamma_coeff[block_j, target]
            if not np.allclose((w_ij + r_ij) @ s_j, t_i * (m_ij + p_ij)):
                print("WARNING ((W_ij + R_ij) s_j = t_i (m_ij + p_ij))", source, target)
                print((w_ij + r_ij) @ s_j)
                print(t_i * (m_ij + p_ij))

            # W_ij s_j = m_ij t_i
            if not np.allclose(w_ij @ s_j, m_ij * t_i):
                print("WARNING (W_ij s_j = m_ij t_i)", source, target)
                print(w_ij @ s_j)
                print(m_ij * t_i)

            # Closed form of the inter-block weights and of the remainder:
            # W_ij s_j = g_ij t_i - R_ij s_j and R_ij s_j = p_ij t_i
            if not np.allclose(w_ij @ s_j, g_ij * t_i - r_ij @ s_j):
                print("WARNING (W_ij s_j = g_ij t_i - R_ij s_j)", source, target)
                print(w_ij, s_j)
                print(w_ij @ s_j)
                print(g_ij, t_i, r_ij, s_j)
                print(g_ij * t_i - r_ij @ s_j)
            if not np.allclose(r_ij @ s_j, p_ij * t_i):
                print("WARNING (R_ij s_j = p_ij t_i)", source, target)
                print(w_ij @ s_j)
                print(g_ij * t_i - r_ij @ s_j)

    # Check consistency FT = SG
    cnc_model = compute_mechanism(cnc_coeff)
    assert np.allclose(cnc_model @ tau_coeff, gamma_coeff @ abs_mech, atol=1e-4)

    # Check closed form
    for source in range(abs_nodes):
        for target in range(abs_nodes):
            if source == target:
                continue
            block_i = _block(source)
            block_j = _block(target)
            w_ij = cnc_coeff[block_i, block_j]
            f_ii = cnc_mech[block_i, block_i]
            m_ij = abs_coeff[source, target]
            t_i = tau_coeff[block_i, source]
            s_j = gamma_coeff[block_j, target]
            assert np.allclose(np.linalg.pinv(f_ii) @ f_ii, np.eye(f_ii.shape[0]))
            assert np.allclose(f_ii @ w_ij @ s_j, m_ij * f_ii @ t_i)
            assert np.allclose(w_ij @ s_j, m_ij * t_i)

    return cnc_coeff, gamma_coeff


def sample_from_abstract_graph_relevants(
    abs_graph: nx.DiGraph | np.ndarray,
    cnc_graph: nx.DiGraph | np.ndarray,
    relevants: list[list[int]],
    blocks: list[list[int]],
    alpha: float = 1e4,
    style: str = "positive",
    normalize: bool = True,
    max_trial: int = 100,
    verbose: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Given an abstract structure samples coefficients
    and structure of the models and the abstraction.

    Parameters
    ----------
    abs_graph : nx.DiGraph | np.ndarray
        The abstract causal graph.
    cnc_graph : nx.DiGraph | np.ndarray
        The concrete causal graph.
    relevants : list[list[int]]
        The relevant nodes for each abstract node.
    blocks : list[list[int]]
        The blocks of nodes in the concrete graph.
    alpha : float, optional
        The alpha parameter for the concrete coefficients.
    style : str, optional
        The style for the concrete coefficients.
    normalize : bool, optional
        Whether to normalize the abstraction coefficients.
    max_trial : int, optional
        The maximum number of trials.

    Returns
    -------
    cnc_coeff : np.ndarray
        The concrete coefficients.
    abs_coeff : np.ndarray
        The abstract coefficients.
    tau_coeff : np.ndarray
        The abstraction coefficients.
    gamma_coeff : np.ndarray
        The exogenous abstraction coefficients.
    """

    # Fail if no more trials available
    if max_trial == 0:
        raise ValueError("Can't sample a concrete model.")

    # networkx graph to numpy array
    if isinstance(abs_graph, nx.DiGraph):
        abs_graph = nx.to_numpy_array(abs_graph)
    if isinstance(cnc_graph, nx.DiGraph):
        cnc_graph = nx.to_numpy_array(cnc_graph)
    assert isinstance(cnc_graph, np.ndarray)
    assert isinstance(abs_graph, np.ndarray)

    # Abstract coefficients
    b = abs_graph.shape[0]
    # Sample magnitudes uniformly in (0.5, 3)
    magnitudes = np.random.uniform(0.5, 3, size=(b, b))
    # Assign random signs
    signs = np.random.choice([-1, 1], size=(b, b))
    # Remove sign to zero entries
    signs[abs_graph == 0] = 0
    # Combine and mask with G
    abs_coeff = signs * magnitudes * abs_graph

    tau_coeff = np.zeros((cnc_graph.shape[0], abs_graph.shape[0]))
    for y, ry in enumerate(relevants):
        d = len(ry)
        eps = 1e-1
        while True:
            if normalize:
                v = np.random.randn(d)
                v = v / np.linalg.norm(v)
            else:
                magnitudes = np.random.uniform(0.5, 3, size=(d,))
                signs = np.random.choice([-1, 1], size=(d,))
                v = magnitudes * signs
            if np.all(np.abs(v) >= eps):
                break
        tau_coeff[ry, y] = v

    # Concrete parameters
    n_trials = 0
    while n_trials < max_trial:
        try:
            cnc_coeff, gamma_coeff = sample_concrete_coeffs(
                abs_coeff,
                cnc_graph,
                tau_coeff,
                blocks,
                relevants,
                alpha=alpha,
                style=style,
            )
            return (
                cnc_coeff,
                abs_coeff,
                tau_coeff,
                gamma_coeff,
            )
        except ValueError as e:
            n_trials += 1
            # Print the error
            if verbose:
                print(e, f"Retrying ({n_trials + 1}th time)")

    # If no results raise an error
    raise ValueError(f"Failure after {max_trial} trials.")
