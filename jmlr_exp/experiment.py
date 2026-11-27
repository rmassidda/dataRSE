"""
Experiment runner for JMLR RSE experiments.

python -m jmlr_exp run [options]
"""

from __future__ import annotations

import itertools
import os
import sys
import time
from collections.abc import Callable, Generator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from causabs.graph import clean_ignorable, graphical_consistency
from causabs.rse import data_RSE
from causabs.utils import linear_anm, seed_everything

from ._config import load_arg, load_sweeps
from ._utils import compare_solutions, compare_subsolution
from .causal import abstract_optimal_adjustment_tce, path_coefficients_tce
from .sampling_general import sample_identifiable_ground_truth

# Generators yield (relevants, tau) pairs; tau is None for combinatorial variants.
VariantRSE = Callable[
    [nx.DiGraph, list[list[int]], bool], Generator[Any, None, None]
]


def measure_variant(
    rse_fn: VariantRSE,
    cnc_graph: nx.DiGraph,
    relevants_start: list[list[int]],
    relevants_gt: list[list[int]],
    max_solutions: int | None = None,
    verbose: bool = False,
    cnc_data: np.ndarray | None = None,
    gt_tce: float | None = None,
) -> dict[str, Any]:
    # patch graphical_consistency to count calls without modifying the original
    counter = [0]
    _real_gc = graphical_consistency

    def _counting(*args, **kwargs):
        counter[0] += 1
        return _real_gc(*args, **kwargs)

    patched = [
        mod
        for mod in sys.modules.values()
        if mod is not None
        and getattr(mod, "graphical_consistency", None) is graphical_consistency
    ]
    for mod in patched:
        mod.graphical_consistency = _counting  # type: ignore

    # instantiate the method
    gen = rse_fn(cnc_graph, relevants_start, verbose)
    if max_solutions is not None:
        gen = itertools.islice(gen, max_solutions)

    # run the method and capture metrics
    n_solutions = 0
    failed = 0
    found = 0
    subfound = 0
    abs_var_ratio = 0.0
    first_rels: list[list[int]] | None = None
    first_tau: dict | None = None
    t0 = time.perf_counter()
    try:
        for rels, tau in gen:
            n_solutions += 1
            if compare_solutions(rels, relevants_gt):
                found = 1
            if compare_subsolution(rels, relevants_gt):
                subfound = 1
            abs_var_ratio = len(rels) / len(relevants_gt)
            if first_rels is None:
                first_rels = rels
                first_tau = tau
    except RecursionError:
        failed = 1
    finally:
        elapsed = time.perf_counter() - t0
        for mod in patched:
            mod.graphical_consistency = graphical_consistency  # type: ignore

    # no solutions count as failure
    if n_solutions == 0:
        failed = 1

    # cap the number of solutions if max_solutions is set
    capped = 0
    if max_solutions is not None and n_solutions == max_solutions:
        capped = 1

    # TCE error on the first returned solution
    tce_err: float | None = None
    if (
        cnc_data is not None
        and gt_tce is not None
        and first_rels is not None
        and first_tau is not None
    ):
        tce_hat = abstract_optimal_adjustment_tce(
            cnc_graph, cnc_data, first_rels, first_tau, 0, 1
        )
        tce_err = abs(tce_hat - gt_tce)

    # Store the normalized abstraction functions
    if first_tau is not None:
        tau_hat = [ty / np.linalg.norm(ty) for ty in first_tau.values()]
    else:
        tau_hat = []

    if verbose:
        tce_str = f"tce_err={tce_err:.4f}" if tce_err is not None else "tce_err=N/A"
        print(
            f" {rse_fn.__name__:<12}  "
            + f"solutions={n_solutions}  "
            + f"gc_calls={counter[0]}  "
            + f"time={elapsed:.4f}s  "
            + f"failed={failed}  "
            + f"found={found}  "
            + f"subfound={subfound}  "
            + f"abs_var_ratio={abs_var_ratio:.4f}  "
            + f"capped={capped}  "
            + f"{tce_str}"
        )

    return {
        "time": elapsed,
        "n_solutions": n_solutions,
        "n_gc_calls": counter[0],
        "failed": failed,
        "capped": capped,
        "found": found,
        "subfound": subfound,
        "abs_var_ratio": abs_var_ratio,
        "tau_hat": tau_hat,
        "tce_err": tce_err,
    }


@dataclass
class RepeatTask:
    relevant_size: int
    block_size: int
    edge_ratio: float
    output_ratio: float
    max_sim: float
    n_samples: int
    abs_code: str
    max_solutions: int | None
    repeat_index: int
    repeat_seed: int
    verbose: bool
    tau_norm: bool
    delta: float = 0.25
    lambda_ridge: float = 1e-5
    # dataRSE variants to run on the same ground truth and data (DATA_VARIANTS)
    variants: tuple[str, ...] = ("dataRSE",)


# Rules to select the number of clusters k* in dataRSE, passed to data_RSE.
# "dataRSE" is the default BIC-based rule, "rrr" is the same rule under the
# name used by the comparison with the rank-based rule ("plateau").
DATA_VARIANTS: dict[str, dict] = {
    "dataRSE": {},
    "plateau": {"k_selection": "plateau"},
    "rrr": {"k_selection": "rrr"},
}


def _make_data_variant(
    vname,
    max_cc,
    cnc_data,
    tau_start,
    delta=0.25,
    lambda_ridge=1e-5,
    rule_kwargs: dict | None = None,
) -> tuple[str, Callable[[nx.DiGraph, list[list[int]]], Generator]]:
    rule_kwargs = rule_kwargs or {}

    def _data_rse_fn(
        g: nx.DiGraph, rels: list[list[int]], verbose: bool = False
    ) -> Generator:
        for rels_out, tau_out in data_RSE(
            g,
            cnc_data,
            tau_start,
            rels,
            max_collinearity_constraints=max_cc,
            delta=delta,
            lambda_ridge=lambda_ridge,
            verbose=verbose,
            **rule_kwargs,
        ):
            yield rels_out, tau_out

    _data_rse_fn.__name__ = vname
    return (vname, _data_rse_fn)


def _run_repeat(task: RepeatTask) -> list[dict]:
    """Executed in a worker process."""
    seed_everything(task.repeat_seed)

    try:
        (
            _abs_graph,
            cnc_graph,
            relevants_gt,
            cnc_coeff_gt,
            _abs_coeff,
            tau_gt,
            treatment_abs,
            outcome_abs,
            _max_sim,
        ) = sample_identifiable_ground_truth(
            task.abs_code,
            block_size=task.block_size,
            relevant_size=task.relevant_size,
            edge_ratio=task.edge_ratio,
            output_ratio=task.output_ratio,
            max_sim=task.max_sim,
            tau_norm=task.tau_norm,
            repeat_index=task.repeat_index,
        )
    except ValueError:
        print(
            f"WARNING: skipping combo "
            f"({task.relevant_size}, {task.block_size}) repeat={task.repeat_index} "
            f"due to sampling error"
        )
        return []

    # initial conditions
    relevants_start = [relevants_gt[treatment_abs], relevants_gt[outcome_abs]]
    tau_start = {0: tau_gt[treatment_abs], 1: tau_gt[outcome_abs]}
    gt_tce = path_coefficients_tce(_abs_graph, treatment_abs, outcome_abs, _abs_coeff)
    cnc_graph_hat = clean_ignorable(cnc_graph.copy(), relevants_start)
    exog = np.random.normal(size=(task.n_samples, cnc_graph.number_of_nodes()))
    cnc_data = linear_anm(cnc_coeff_gt, exog)

    if task.verbose:
        for y, ty in tau_gt.items():
            with np.printoptions(precision=4, suppress=True):
                print(f"tau_gt[{y}] = {ty / np.linalg.norm(ty)}")
        with np.printoptions(precision=4, suppress=True):
            print(f"ABS graph coeff:\n{_abs_coeff}")

    # store the normalized ground-truth abstraction functions
    normalized_tau_gt = [ty / np.linalg.norm(ty) for ty in tau_gt.values()]

    base_row = {
        "graph_code": task.abs_code,
        "block_size": task.block_size,
        "relevant_size": task.relevant_size,
        "edge_ratio": task.edge_ratio,
        "output_ratio": task.output_ratio,
        "max_sim": task.max_sim,
        "n_cnc_nodes": cnc_graph_hat.number_of_nodes(),
        "repeat": task.repeat_index,
        "n_samples": task.n_samples,
        "delta": task.delta,
        "lambda_ridge": task.lambda_ridge,
        "tau_gt": normalized_tau_gt,
    }

    # every variant runs on the same ground truth and data, so that accuracy
    # and running time are paired across variants
    rows = []
    for vname in task.variants:
        variant_name, variant_fn = _make_data_variant(
            vname,
            max_cc=1,
            cnc_data=cnc_data,
            tau_start=tau_start,
            delta=task.delta,
            lambda_ridge=task.lambda_ridge,
            rule_kwargs=DATA_VARIANTS[vname],
        )

        metrics = measure_variant(
            variant_fn,
            cnc_graph_hat,
            relevants_start,
            relevants_gt,
            max_solutions=task.max_solutions,
            verbose=task.verbose,
            cnc_data=cnc_data,
            gt_tce=gt_tce,
        )
        rows.append(
            {
                **base_row,
                "variant": variant_name,
                **metrics,
            }
        )

    return rows


def _run(tasks: list[RepeatTask], n_workers: int) -> list[dict]:
    # run the experiment
    pbar = tqdm(total=len(tasks), desc="graph_exp")

    all_rows: list[dict] = []

    def _collect(repeat_rows: list[dict]) -> None:
        all_rows.extend(repeat_rows)
        pbar.update(1)

    if n_workers == 1:
        for task in tasks:
            _collect(_run_repeat(task))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_repeat, t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    _collect(fut.result())
                except Exception as exc:  # type: ignore
                    print(f"Task {futures[fut]} raised: {exc}")

    pbar.close()
    return all_rows


def run(
    abs_codes: list[str] = ["A", "B", "C", "D"],
    n_repeats: int = 15,
    max_solutions: int | None = None,
    block_size: int | None = None,
    relevant_size: int | None = None,
    edge_ratio: float = 0.5,
    output_ratio: float = 0.2,
    n_samples: int | None = None,
    max_sim: float = 0.9,
    config_dir: str = "config/",
    results_dir: str = "results",
    seed: int = 42,
    verbose: bool = False,
    n_workers: int = 1,
    tau_norm: bool = True,
    delta: float = 0.25,
    lambda_ridge: float = 1e-5,
    exp_name: str = "rse",
    variants: list[str] = ["dataRSE"],
) -> None:
    unknown = set(variants) - set(DATA_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants {unknown}, choose among {list(DATA_VARIANTS)}")

    # load sweep configs but override defaults
    sweeps = load_sweeps(config_dir, filename="datarse_sweeps.yaml")
    block_sizes: list[int] = load_arg("block_size", block_size, sweeps)
    relevant_sizes: list[int] = load_arg("relevant_size", relevant_size, sweeps)
    edge_ratios: list[float] = load_arg("edge_ratio", edge_ratio, sweeps)
    output_ratios: list[float] = load_arg("output_ratio", output_ratio, sweeps)
    n_samples_list: list[int] = load_arg("n_samples", n_samples, sweeps)

    # generate combinations of the sweeps
    rng = np.random.default_rng(seed)
    raw_combos = list(
        itertools.product(
            relevant_sizes,
            block_sizes,
            edge_ratios,
            output_ratios,
            n_samples_list,
            abs_codes,
        )
    )
    combo_seeds = rng.integers(0, 10_000, size=len(raw_combos))
    raw_combos = list(zip(raw_combos, combo_seeds))

    # generate tasks
    tasks: list[RepeatTask] = []
    for (rel_size, blk_size, e_ratio, o_ratio, n_samp, code), c_seed in raw_combos:
        if rel_size >= blk_size:
            print(f"Skip rel_size>=block_size ({rel_size}>={blk_size})")
            continue

        # derive per-repeat seeds from the combo seed (fully deterministic)
        combo_rng = np.random.default_rng(int(c_seed))
        repeat_seeds = combo_rng.integers(0, 10_000, size=n_repeats)

        for i_rep in range(n_repeats):
            tasks.append(
                RepeatTask(
                    relevant_size=rel_size,
                    block_size=blk_size,
                    edge_ratio=e_ratio,
                    output_ratio=o_ratio,
                    max_sim=max_sim,
                    n_samples=n_samp,
                    abs_code=code,
                    max_solutions=max_solutions,
                    repeat_index=i_rep,
                    repeat_seed=int(repeat_seeds[i_rep]),
                    tau_norm=tau_norm,
                    verbose=verbose,
                    delta=delta,
                    lambda_ridge=lambda_ridge,
                    variants=tuple(variants),
                )
            )

    # collect results
    all_rows = _run(tasks, n_workers)

    # save results
    df = pd.DataFrame(all_rows)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{exp_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


