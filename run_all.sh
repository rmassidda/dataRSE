#!/usr/bin/env bash
# Run every experiment reported in the paper. Results are written to
# results/<name>.csv; experiments whose results already exist are skipped.
#
#   ./run_all.sh          full experiments (N_WORKERS=64 by default)
#   ./run_all.sh quick    small sweep for a quick local check (N_WORKERS=6)
#
# The number of parallel worker processes can be set with N_WORKERS.
# Graph I of config/graphs.yaml is shown as Graph H in the paper.

set -eu

# Quick mode: small sweep (config/quick/) on a subset of the graphs, followed
# by the plots. Outputs are timestamped, so that each run produces a new
# results file and plot folder instead of being skipped.
if [ "${1:-}" = "quick" ]; then
    QUICK_WORKERS="${N_WORKERS:-6}"
    QUICK_CODES="A,B,C,D,E,I"
    QUICK_NAME="quick_ms90_$(date +%Y%m%d_%H%M%S)"

    python -m jmlr_exp run --exp-name="${QUICK_NAME}" --n-workers="${QUICK_WORKERS}" \
        --abs-codes="${QUICK_CODES}" --max-sim=0.90 --n-repeats=5 \
        --config-dir="config/quick/"

    python -m jmlr_exp summary_grid --exp-name="${QUICK_NAME}" --abs-codes="${QUICK_CODES}" --n-workers="${QUICK_WORKERS}"
    python -m jmlr_exp plot_abs --exp-name="${QUICK_NAME}" --abs-codes="${QUICK_CODES}" --n-workers="${QUICK_WORKERS}"

    echo "Results: results/${QUICK_NAME}.csv, plots: plots/${QUICK_NAME}/"
    exit 0
fi

N_WORKERS="${N_WORKERS:-64}"

run_experiment() {
    local cmd="$1"
    local out_file="$2"
    shift 2
    if [ ! -f "results/${out_file}.csv" ]; then
        python -m jmlr_exp "${cmd}" --exp-name="${out_file}" --n-workers="${N_WORKERS}" "$@"
    else
        echo "results/${out_file}.csv already exists, skipping"
    fi
}

# size of the MiSE computed with qRSE (Figure 3)
run_experiment run_mise "qRSE_ms90" --abs-codes="A,B,C,D" --max-sim=0.90 --max-solutions=10000

# dataRSE, main configuration: gamma = 0.9, normalized abstraction vectors
run_experiment run "rse_ms90" --abs-codes="A,B,C,D" --max-sim=0.90

# dataRSE, appendix: not normalized abstraction vectors
run_experiment run "rse_ms90_ntn" --abs-codes="A,B,C,D" --max-sim=0.90 --tau-norm=False

# dataRSE, appendix: no collinearity margin (gamma = 1.00)
run_experiment run "rse_ms100" --abs-codes="A,B,C,D" --max-sim=1.01

# dataRSE, appendix: additional graphs E, F, G, and H (stored as I)
run_experiment run "rse_ms90_efgh" --abs-codes="E,F,G,I" --max-sim=0.90

# dataRSE, appendix: paired comparison of the BIC-based ("rrr") and the
# rank-based ("plateau") rules selecting the number of clusters. Running
# times are only comparable within this run: keep N_WORKERS fixed and avoid
# running other jobs at the same time.
run_experiment run "compare_rule" --abs-codes="A,B,C,D,E,F,G,I" \
    --variants='["plateau","rrr"]' --max-sim=0.90 --config-dir="config/compare/"
