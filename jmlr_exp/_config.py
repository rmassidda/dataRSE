"""Config file loading utilities."""

import os

import networkx as nx
import yaml

from ._utils import digraph


def load_graph(
    abs_code: str, config_dir: str = "config/"
) -> tuple[nx.DiGraph, int, int]:
    """Loads the graph and the treatment and outcome IDs"""
    with open(config_dir + "graphs.yaml") as f:
        data = yaml.safe_load(f)
    spec = data["graphs"][abs_code]
    graph = digraph(spec["nodes"], [tuple(e) for e in spec["edges"]])
    treatment, outcome = spec["targets"]
    return graph, treatment, outcome


def load_sweeps(
    config_dir: str = "config/", filename: str = "datarse_sweeps.yaml"
) -> dict[str, list]:
    """Load sweep parameter lists from a sweep config file.

    Returns a dict with keys ``block_size``, ``relevant_size``,
    ``edge_ratio``, and ``output_ratio`` (plus ``n_samples`` for
    ``datarse_sweeps.yaml``), each mapping to a sorted list of values to
    iterate over. dataRSE and qRSE use separate files (see
    ``datarse_sweeps.yaml`` and ``qrse_sweeps.yaml``) since qRSE's running
    time blows up combinatorially outside a narrower, known-safe range.
    """
    with open(os.path.join(config_dir, filename)) as f:
        return yaml.safe_load(f)


def load_arg(name, default, sweeps):
    if default is not None:
        return [default]
    elif name in sweeps and sweeps[name] is not None:
        return sweeps[name]
    else:
        raise ValueError(f"Sweep parameter '{name}' is missing and has no default")
