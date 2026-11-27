"""
Copyright (c) 2025 Riccardo Massidda
Released under the MIT license, see LICENSE.

Implementation of the Relevant Sufficiency
Extender (RSE) algorithm from Massidda et al.
"""

from typing import Generator

import networkx as nx

from causabs.graph import blocks_from_relevants, graphical_consistency
from causabs.rse.utils import (
    abstract_intersections,
    cnc_intersection,
    multi_partitions,
)


def get_new_relevants(
    intersected_vars: set[int],
    non_intersected_vars: set[int],
    cnc_graph: nx.DiGraph,
    verbose: bool = False,
) -> set[int]:
    """Computes the set of new relevant variables given a set of low-level variables in an intersection."""
    new_relevants = set()
    for xi in intersected_vars:
        xi_children = set(cnc_graph.successors(xi))
        out_children = non_intersected_vars.intersection(xi_children)
        if out_children:
            new_relevants.add(xi)
            if verbose:
                print(f"{xi} new relevant as it reaches {out_children} directly")
    return new_relevants


def compute_abstract_targets(
    new_relevants: set[int],
    abs_nodes: int,
    blocks: list[list[int]],
    non_intersected_vars: set[int],
    cnc_graph: nx.DiGraph,
    verbose: bool = False,
) -> dict[int, frozenset[int]]:
    """For each new relevant variable, computes the existing abstract
    variables it directly reaches through the non-intersected part of
    their blocks."""
    q_targets = {}
    for x_rel in new_relevants:
        q_x = set()
        for yj in range(abs_nodes):
            # check if x_rel is directly connecting to a variable
            # in the non-intersected block of Yj
            bj_no_inter = non_intersected_vars.intersection(blocks[yj])
            for child in cnc_graph.successors(x_rel):
                if child in bj_no_inter:
                    if verbose:
                        print(f"X{x_rel} -> Y{yj} (X{child} in {bj_no_inter})")
                    q_x.add(yj)
                    break

        q_targets[x_rel] = frozenset(q_x)

        # this set should never be empty
        assert len(q_targets[x_rel]) > 0

    return q_targets


def recursive_RSE(
    cnc_graph: nx.DiGraph,
    relevants: list[list[int]],
    q_heuristic: bool = False,
    verbose: bool = False,
) -> Generator[list[list[int]], None, None]:
    """RSE algorithm. With q_heuristic, only the new relevant variables with
    the same abstract targets can be clustered together (qRSE)."""

    # stop if not graphically consistent
    if not graphical_consistency(cnc_graph, relevants):
        return

    # compute block intersections
    blocks = blocks_from_relevants(cnc_graph, relevants)
    intersections = abstract_intersections(blocks)
    intersections = sorted(intersections, key=len)

    # found a sufficient solution
    if len(intersections) == 0:
        yield relevants
        return

    # get intersected and non intersected vars
    all_vars = set(cnc_graph.nodes())
    all_intersection_vars = set().union(
        *(cnc_intersection(i, blocks) for i in intersections)
    )
    non_intersected_vars = all_vars - all_intersection_vars

    # select the first intersection
    intersection = intersections[0]
    intersected_vars = cnc_intersection(intersection, blocks)

    # select candidate relevants
    new_relevants = get_new_relevants(
        intersected_vars, non_intersected_vars, cnc_graph, verbose
    )

    # all new relevants potentially clustered together
    m_targets = [list(new_relevants)]

    # apply q_heuristic to try to cluster together only
    # new relevants with the same abstract targets
    if q_heuristic:
        q_targets = compute_abstract_targets(
            new_relevants,
            len(relevants),
            blocks,
            non_intersected_vars,
            cnc_graph,
            verbose,
        )
        # cluster together variables with the same q targets
        q_clusters: dict[frozenset, list[int]] = {}
        for x_l in new_relevants:
            abs_targets = q_targets[x_l]
            if abs_targets not in q_clusters:
                q_clusters[abs_targets] = []
            q_clusters[abs_targets].append(x_l)
        m_targets = list(q_clusters.values())

    # recurse on every partition of each group of new relevant variables
    for new_abs in multi_partitions(m_targets):
        yield from recursive_RSE(
            cnc_graph,
            relevants + new_abs,  # list[list[int]] + list[list[int]]
            q_heuristic,
            verbose,
        )


def recursive_qRSE(
    cnc_graph: nx.DiGraph,
    relevants: list[list[int]],
    verbose: bool = False,
) -> Generator[list[list[int]], None, None]:
    """Recursive RSE with the q heuristic (qRSE)."""
    return recursive_RSE(
        cnc_graph=cnc_graph,
        relevants=relevants,
        q_heuristic=True,
        verbose=verbose,
    )
