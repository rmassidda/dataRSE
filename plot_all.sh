#!/usr/bin/env bash
# Plot the results of run_all.sh into plots/<name>/, and collect every
# figure of the paper into figures/, named as in the LaTeX source.
# Graph I of config/graphs.yaml is shown as Graph H in the figures.

set -eu
N_WORKERS="${N_WORKERS:-6}"
FIG="figures"
mkdir -p "${FIG}/appendix"

# copy a figure into figures/, warning instead of failing if it is missing
# (e.g., tau_notfound_* is only produced when some run returns an incorrect
# solution, which might not happen on small sweeps)
MISSING=0
copy() {
    if [ -f "$1" ]; then
        cp "$1" "$2"
    else
        echo "WARNING: $1 not found" >&2
        MISSING=$((MISSING + 1))
    fi
}

require() {
    if [ ! -f "results/$1.csv" ]; then
        echo "results/$1.csv not found, run ./run_all.sh first" >&2
        exit 1
    fi
}

# size of the MiSE (Figure 3)
require qRSE_ms90
python -m jmlr_exp figure3 --exp-name=qRSE_ms90 --abs-codes="A,B,C,D"

# dataRSE: summary grids and recovery of the abstraction coefficients
for exp in rse_ms90 rse_ms90_ntn rse_ms100; do
    require "${exp}"
    python -m jmlr_exp summary_grid --exp-name="${exp}" --abs-codes="A,B,C,D" --n-workers="${N_WORKERS}"
    python -m jmlr_exp plot_abs --exp-name="${exp}" --abs-codes="A,B,C,D" --n-workers="${N_WORKERS}"
done
require rse_ms90_efgh
python -m jmlr_exp summary_grid --exp-name=rse_ms90_efgh --abs-codes="E,F,G,I" --n-workers="${N_WORKERS}"
python -m jmlr_exp plot_abs --exp-name=rse_ms90_efgh --abs-codes="E,F,G,I" --n-workers="${N_WORKERS}"

# comparison with rank-based partitioning
require compare_rule
python -m jmlr_exp plot_compare --exp-name=compare_rule --abs-codes='["A","B","C","D","E","F","G","I"]'

# main body: size of the MiSE and results (gamma = 0.9, normalized)
copy plots/qRSE_ms90/figure3.pdf "${FIG}/figure3.pdf"
for f in summary_grid_n20000 summary_grid_nsamples \
    tau_full_A_n5000 tau_full_B_n5000 tau_full_C_n5000 tau_full_D_n5000; do
    copy "plots/rse_ms90/${f}.pdf" "${FIG}/${f}.pdf"
done

# appendix: gamma = 0.9 breakdown, gamma = 0.9 without normalization, and
# gamma = 1.00 (no collinearity margin)
A="${FIG}/appendix"
copy plots/rse_ms90/summary_grid_full_nsamples.pdf "${A}/summary_grid_full_ms90.pdf"
copy plots/rse_ms90_ntn/summary_grid_full_nsamples.pdf "${A}/summary_grid_full_ms90_ntn.pdf"
copy plots/rse_ms100/summary_grid_full_nsamples.pdf "${A}/summary_grid_full_ms100.pdf"
for g in ms90 ms100; do
    for n in 1000 5000 10000; do
        copy "plots/rse_${g}/summary_grid_n${n}.pdf" "${A}/summary_grid_n${n}_${g}.pdf"
    done
done
for c in A B C D; do
    copy "plots/rse_ms100/tau_full_${c}_n5000.pdf" "${A}/tau_full_${c}_ms100.pdf"
done
# appendix: abstraction-coefficient recovery split into correct and incorrect
# solutions, with and without the collinearity margin
for g in ms90 ms100; do
    for c in A B C D; do
        copy "plots/rse_${g}/tau_${c}_n5000.pdf" "${A}/tau_found_${c}_${g}.pdf"
        copy "plots/rse_${g}/tau_notfound_${c}_n5000.pdf" "${A}/tau_notfound_${c}_${g}.pdf"
    done
done

# appendix: additional graphs E-H (Graph H is stored as I in the results)
EFGH="plots/rse_ms90_efgh"
copy "${EFGH}/summary_grid_nsamples.pdf" "${A}/summary_grid_nsamples_efgh.pdf"
copy "${EFGH}/summary_grid_full_nsamples.pdf" "${A}/summary_grid_full_efgh.pdf"
for c in E F G; do
    copy "${EFGH}/tau_full_${c}_n5000.pdf" "${A}/tau_full_${c}_efgh.pdf"
done
copy "${EFGH}/tau_full_I_n5000.pdf" "${A}/tau_full_H_efgh.pdf"

# appendix: comparison with rank-based partitioning
copy plots/compare_rule/compare_rule_ABCD.pdf "${A}/compare_rule_ABCD.pdf"
copy plots/compare_rule/compare_rule_EFGH.pdf "${A}/compare_rule_EFGH.pdf"

echo "Figures collected in ${FIG}/ (${MISSING} missing)"
