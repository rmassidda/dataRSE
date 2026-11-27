"""Methods to compute blocks and abstract graph from the concrete graph and the set of relevant variables."""

import networkx as nx


def abstract_graph(cnc_graph: nx.DiGraph, relevants: list[list[int]]) -> nx.DiGraph:
    """
    Generates the abstract graph from the concrete graph and the blocks.

    If a relevant variable of Yi has a direct edge to a block variable of Yj in the concrete graph, then there is an edge from Yi to Yj in the abstract graph.

    Parameters
    ----------
    cnc_graph : nx.DiGraph
        The concrete graph with d nodes.
    relevants : list[list[int]]
        The set of relevant variables with length b.

    Returns
    -------
    nx.DiGraph
        The abstract graph with b nodes.
    """

    blocks = blocks_from_relevants(cnc_graph, relevants)
    node_to_blocks: dict[int, set[int]] = {}
    for block_idx, block in enumerate(blocks):
        for node in block:
            if node not in node_to_blocks:
                node_to_blocks[node] = set()
            node_to_blocks[node].add(block_idx)

    node_to_relevant: dict[int, int] = {}
    for rel_idx, rel in enumerate(relevants):
        for node in rel:
            node_to_relevant[node] = rel_idx

    abstract_edges = set()
    for a, Yi in node_to_relevant.items():
        for b in cnc_graph.successors(a):
            for Yj in node_to_blocks.get(b, set()):
                if Yi != Yj:
                    abstract_edges.add((Yi, Yj))

    abs_graph = nx.DiGraph()
    abs_graph.add_nodes_from(range(len(blocks)))
    abs_graph.add_edges_from(abstract_edges)

    return abs_graph


def blocks_from_relevants(
    cnc_graph: nx.DiGraph, relevants: list[list[int]]
) -> list[list[int]]:
    """
    Computes the blocks of variables from the set of relevant variables.

    Parameters
    ----------
    cnc_graph : nx.DiGraph
        The concrete graph with d nodes.
    relevants : list[list[int]]
        The set of relevant variables with length b.

    Returns
    -------
    list[list[int]]
        The blocks of variables with length b.
    """
    all_relevants: set[int] = set()
    for rel in relevants:
        all_relevants.update(rel)

    blocks: list[list[int]] = []

    for rel in relevants:
        visited: set[int] = set(rel)
        stack = list(rel)

        while stack:
            node = stack.pop()
            for pred in cnc_graph.predecessors(node):
                if pred not in visited and pred not in all_relevants:
                    visited.add(pred)
                    stack.append(pred)

        blocks.append(sorted(visited))

    return blocks


def clean_ignorable(
    cnc_graph: nx.DiGraph,
    relevants: list[list[int]],
) -> nx.DiGraph:
    """Return a copy of the CNC graph with ignorable variables removed."""
    blocks = blocks_from_relevants(cnc_graph, relevants)
    all_vars = set(cnc_graph.nodes())
    block_vars = set(sum(blocks, []))
    ignorable_vars = all_vars - block_vars
    cnc_graph = cnc_graph.copy()
    for x_ignored in ignorable_vars:
        cnc_graph.remove_node(x_ignored)
    return cnc_graph


def graphical_consistency(
    cnc_graph: nx.DiGraph,
    relevants: list[list[int]],
    verbose: bool = False,
) -> bool:
    abs_graph = abstract_graph(cnc_graph, relevants)

    # Set of relevant variables
    all_relevants = set()
    for ry in relevants:
        all_relevants = all_relevants.union(set(ry))

    def is_connected(xi: int, rj: list[int]) -> bool:
        """Check if xi is connected to any variable in rj."""
        visited = {xi}
        stack = [xi]
        while stack:
            xj = stack.pop()
            for child in cnc_graph.successors(xj):
                if child in visited:
                    continue
                visited.add(child)
                if child in all_relevants:
                    if child in rj:
                        return True
                else:
                    stack.append(child)
        return False

    # yi -> yj iff exists xi --> xj
    for yi, yj in abs_graph.edges:
        # Get relevants
        ri, rj = relevants[yi], relevants[yj]
        # Check all relevants
        for xi in ri:
            if not is_connected(xi, rj):
                if verbose:
                    print(f"X{xi} (Y{yi}: {ri}) is not connected to any in Y{yj}: {rj}")
                return False
    return True
