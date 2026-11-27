"""
Measure the size of the class of Minimal Sufficient Extensions (MiSE) via
the purely combinatorial qRSE algorithm.

Unlike dataRSE (see experiment.py), this does not use any low-level data or
abstraction coefficients: it only measures how many minimal sufficient
extensions qRSE enumerates for a given abstract/concrete graph pair.

python -m jmlr_exp run_mise [options]
"""

from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from causabs.graph import clean_ignorable
from causabs.rse import recursive_qRSE
from causabs.utils import seed_everything

from ._config import load_arg, load_sweeps
from .experiment import measure_variant
from .sampling_general import sample_identifiable_ground_truth


@dataclass
class MiseTask:
    relevant_size: int
    block_size: int
    edge_ratio: float
    output_ratio: float
    max_sim: float
    abs_code: str
    max_solutions: int | None
    repeat_index: int
    repeat_seed: int
    verbose: bool
    tau_norm: bool


def _run_repeat(task: MiseTask) -> list[dict]:
    """Executed in a worker process."""
    seed_everything(task.repeat_seed)

    try:
        (
            _abs_graph,
            cnc_graph,
            relevants_gt,
            _cnc_coeff_gt,
            _abs_coeff,
            _tau_gt,
            treatment_abs,
            outcome_abs,
            max_sim,
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

    relevants_start = [relevants_gt[treatment_abs], relevants_gt[outcome_abs]]
    cnc_graph_hat = clean_ignorable(cnc_graph.copy(), relevants_start)

    metrics = measure_variant(
        lambda g, r, v: ((s, None) for s in recursive_qRSE(g, r, verbose=v)),
        cnc_graph_hat,
        relevants_start,
        relevants_gt,
        max_solutions=task.max_solutions,
        verbose=task.verbose,
    )

    row = {
        "graph_code": task.abs_code,
        "block_size": task.block_size,
        "relevant_size": task.relevant_size,
        "edge_ratio": task.edge_ratio,
        "output_ratio": task.output_ratio,
        "max_sim": max_sim,
        "n_cnc_nodes": cnc_graph_hat.number_of_nodes(),
        "repeat": task.repeat_index,
        "variant": "qRSE",
        **metrics,
    }
    return [row]


def _run(tasks: list[MiseTask], n_workers: int) -> list[dict]:
    pbar = tqdm(total=len(tasks), desc="mise_exp")

    all_rows: list[dict] = []

    def _collect(repeat_rows: list[dict]) -> None:
        all_rows.extend(repeat_rows)
        pbar.update(len(repeat_rows))

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


def run_mise(
    abs_codes: list[str] = ["A", "B", "C", "D"],
    n_repeats: int = 15,
    max_solutions: int | None = None,
    block_size: int | None = None,
    relevant_size: int | None = None,
    edge_ratio: float = 0.5,
    output_ratio: float = 0.2,
    max_sim: float = 0.9,
    config_dir: str = "config/",
    results_dir: str = "results",
    seed: int = 42,
    verbose: bool = False,
    n_workers: int = 1,
    tau_norm: bool = True,
    exp_name: str = "mise",
) -> None:
    """Measure the size of the MiSE, with qRSE, over a sweep of configurations."""
    # load sweep configs but override defaults; qRSE uses its own, narrower
    # sweep range since its running time blows up combinatorially outside it
    sweeps = load_sweeps(config_dir, filename="qrse_sweeps.yaml")
    block_sizes: list[int] = load_arg("block_size", block_size, sweeps)
    relevant_sizes: list[int] = load_arg("relevant_size", relevant_size, sweeps)
    edge_ratios: list[float] = load_arg("edge_ratio", edge_ratio, sweeps)
    output_ratios: list[float] = load_arg("output_ratio", output_ratio, sweeps)

    # generate combinations of the sweeps
    rng = np.random.default_rng(seed)
    raw_combos = list(
        itertools.product(
            relevant_sizes,
            block_sizes,
            edge_ratios,
            output_ratios,
            abs_codes,
        )
    )
    combo_seeds = rng.integers(0, 10_000, size=len(raw_combos))
    raw_combos = list(zip(raw_combos, combo_seeds))

    # generate tasks
    tasks: list[MiseTask] = []
    for (rel_size, blk_size, e_ratio, o_ratio, code), c_seed in raw_combos:
        if rel_size >= blk_size:
            print(f"Skip rel_size>=block_size ({rel_size}>={blk_size})")
            continue

        # derive per-repeat seeds from the combo seed (fully deterministic)
        combo_rng = np.random.default_rng(int(c_seed))
        repeat_seeds = combo_rng.integers(0, 10_000, size=n_repeats)

        for i_rep in range(n_repeats):
            tasks.append(
                MiseTask(
                    relevant_size=rel_size,
                    block_size=blk_size,
                    edge_ratio=e_ratio,
                    output_ratio=o_ratio,
                    max_sim=max_sim,
                    abs_code=code,
                    max_solutions=max_solutions,
                    repeat_index=i_rep,
                    repeat_seed=int(repeat_seeds[i_rep]),
                    tau_norm=tau_norm,
                    verbose=verbose,
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
