# Learning Sufficient Extensions of Linear Causal Abstractions

Code to replicate the experiments of the paper. It contains the
implementation of `RSE`, `qRSE`, and `dataRSE` (package `causabs`) and the
experimental pipeline (package `jmlr_exp`).

## Installation

The code requires Python 3.12 or later.

```bash
pip install -r requirements.txt
```

Alternatively, `docker build -t rse .` builds an image whose default command
runs the quick check below.

## Quick check

```bash
./run_all.sh quick
```

This runs `dataRSE` on a small sweep (`config/quick/`) and plots the results in
`plots/quick_ms90_<timestamp>/`. It takes a few minutes with 6 workers.

## Replicating the experiments

```bash
./run_all.sh     # results/<experiment>.csv
./plot_all.sh    # plots/<experiment>/ and figures/
```

`run_all.sh` runs every experiment of the paper and skips the ones whose
results already exist. By default it uses 64 worker processes, which can be
changed with `N_WORKERS=<n> ./run_all.sh`. `plot_all.sh` plots the results
and collects every figure of the paper into `figures/`, with the same file
names used in the LaTeX source.

The results and plots of the runs reported in the paper are available on
Zenodo, at https://doi.org/10.5281/zenodo.24581034. To regenerate the
figures from them without running the experiments, extract
`rse_results.zip` into `results/` and run `./plot_all.sh`.

| Experiment      | Paper                                                         |
|-----------------|---------------------------------------------------------------|
| `qRSE_ms90`     | Size of the MiSE (Figure 3)                                   |
| `rse_ms90`      | Main results, $\gamma=0.9$, normalized (main text and Appendix) |
| `rse_ms90_ntn`  | $\gamma=0.9$, not normalized (Appendix)                       |
| `rse_ms100`     | $\gamma=1.00$, no collinearity margin (Appendix)              |
| `rse_ms90_efgh` | Additional graphs E–H (Appendix)                              |
| `compare_rule`  | BIC-based vs. rank-based partitioning (Appendix)              |

All experiments are seeded and deterministic, except for running times.
The running times of `compare_rule` are only comparable within the same run,
so it should run with a fixed `N_WORKERS` and without other jobs on the same
machine.

## Correspondence with the paper

| Paper                                         | Code                                                                 |
|-----------------------------------------------|----------------------------------------------------------------------|
| `RSE`, `qRSE`                                 | `causabs.rse.recursive_RSE`, `causabs.rse.recursive_qRSE`            |
| `dataRSE`                                     | `causabs.rse.data_RSE`                                               |
| BIC-based maximal partitioning                | `k_selection="rrr"` (default); variant `dataRSE` or `rrr`            |
| Rank-based partitioning, threshold $\delta$   | `k_selection="plateau"`, `delta=0.25`; variant `plateau`             |
| Ridge regularization $\lambda=10^{-5}$        | `lambda_ridge=1e-5`                                                  |
| Collinearity margin $\gamma$                  | `--max-sim` (`--max-sim=1.01` disables the check, i.e., $\gamma=1.00$) |
| `normalized` flag                             | `--tau-norm`                                                         |
| Edge ratio $\rho_e$, output ratio $\rho_o$    | `--edge-ratio=0.5`, `--output-ratio=0.2`                             |
| Graphs A–H                                    | `config/graphs.yaml`, where Graph H is stored as `I`                 |
| Sweeps of $n$, $\lvert\mathbf{R}_Y\rvert$, $\lvert\mathbf{B}_Y\rvert$ | `config/datarse_sweeps.yaml`, `config/qrse_sweeps.yaml`, `config/compare/` |

## Structure

```
causabs/
  graph.py              abstract graph, blocks, graphical consistency
  rse/rse.py            RSE and qRSE
  rse/data_rse.py       dataRSE and the maximal partitioning
  sampling/             sampling of the ground-truth abstractions
  utils.py              linear ANMs and seeding
jmlr_exp/
  experiment.py         dataRSE experiments (python -m jmlr_exp run)
  mise.py               size of the MiSE (python -m jmlr_exp run_mise)
  sampling*.py          ground truths with the collinearity margin
  causal.py             total causal effect (TCE) estimation
  plot.py               figures
config/                 abstract graphs and sweeps
```
