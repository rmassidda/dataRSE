"""
Plotting utilities for JMLR RSE experiment results.

python -m jmlr_exp figure3      [options]
python -m jmlr_exp summary_grid [options]
python -m jmlr_exp plot_abs     [options]
python -m jmlr_exp plot_compare [options]
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: faster and avoids GUI overhead

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ._plot import _WONG_PALETTE, build_style_maps


def _run_jobs(fn, jobs: list[dict], n_workers: int = 1) -> None:
    """Run `fn(**job)` for each job dict in `jobs`.

    Each job is fully independent (produces its own figure(s) to disk), so
    when `n_workers > 1` the jobs are dispatched to a process pool instead of
    running sequentially. Exceptions raised in worker processes are
    re-raised in the main process.
    """
    if n_workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            fn(**job)
        return

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(fn, **job) for job in jobs]
        for fut in as_completed(futures):
            fut.result()


def _add_combo_group_lines(ax, combo_order):
    """Draw a vertical line marking the start of each new relevant-size
    regime on a categorical (relevant_size, block_size) x-axis.
    """
    if not combo_order:
        return
    prev_relevant = None
    for i, combo in enumerate(combo_order):
        relevant = combo.strip("()").split(",")[0].strip()
        if prev_relevant is not None and relevant != prev_relevant:
            ax.axvline(
                i, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, zorder=0
            )
        prev_relevant = relevant


def _combo_regime_groups(combo_order: list[str]) -> list[list[str]]:
    """Split `combo_order` into contiguous groups that share the same
    relevant_size, so lines can be drawn separately within each group and
    never connect points across a relevant-size regime change."""
    groups: list[list[str]] = []
    prev_relevant = None
    for combo in combo_order:
        relevant = combo.strip("()").split(",")[0].strip()
        if relevant != prev_relevant:
            groups.append([])
        groups[-1].append(combo)
        prev_relevant = relevant
    return groups


def _combo_block_size(combo: str) -> str:
    return combo.strip("()").split(",")[1].strip()


def _combo_relevant_size(combo: str) -> str:
    return combo.strip("()").split(",")[0].strip()


def _set_two_level_combo_xticks(
    ax, combo_order, fontsize: int = 9, grid_sizes: bool = False, fig3: bool = False
) -> None:
    """Replace the default "(relevant, block)" x tick labels on a combo axis
    with a two-level layout: the block_size directly on the ticks, and the
    relevant_size grouped under a bracket spanning all combos that share it.
    """
    if not combo_order:
        return

    if fig3:
        fontsize = 15

    positions = list(range(len(combo_order)))
    ax.set_xticks(positions)
    ax.set_xticklabels([_combo_block_size(c) for c in combo_order])
    # reduce the font size of the tick labels
    if not fig3:
        ax.tick_params(axis="x", labelsize=fontsize * 0.55)

    trans = ax.get_xaxis_transform()  # x: data coords, y: axes-fraction coords
    y_bracket = -0.10
    y_text = -0.13
    tick_h = 0.025

    if grid_sizes:
        y_bracket = -0.13
        y_text = -0.15
        tick_h = 0.025
    elif fig3:
        y_bracket = -0.15
        y_text = -0.18
        tick_h = 0.03

    start = 0
    for group in _combo_regime_groups(combo_order):
        end = start + len(group) - 1
        x0, x1 = start, end
        relevant = _combo_relevant_size(group[0])

        ax.plot(
            [x0, x1],
            [y_bracket, y_bracket],
            color="black",
            linewidth=0.8,
            transform=trans,
            clip_on=False,
        )
        if start == 0 and fig3:
            ax.text(x0, y_bracket - tick_h / 3, f"$|\\mathbf{{B}}_Y|=$", ha="right", va="bottom", transform=trans, clip_on=False, fontsize=10)
        ax.plot(
            [x0, x0],
            [y_bracket, y_bracket + tick_h],
            color="black",
            linewidth=0.8,
            transform=trans,
            clip_on=False,
        )
        ax.plot(
            [x1, x1],
            [y_bracket, y_bracket + tick_h],
            color="black",
            linewidth=0.8,
            transform=trans,
            clip_on=False,
        )
        ax.text(
            (x0 + x1) / 2,
            y_text,
            f"$|\\mathbf{{R}}_Y|={relevant}$",
            ha="center",
            va="top",
            transform=trans,
            clip_on=False,
            fontsize=fontsize * 0.85,
        )

        start = end + 1


def _plot_by_combo(
    df: pd.DataFrame,
    key: str,
    y_label: str,
    abs_codes: list[str],
    plots_dir: str,
    exp_name: str,
    combo_order: list[str],
    palette: dict,
    dashes: dict,
    markers: dict,
    log_scale_y: bool = False,
    out_file: str | None = None,
    label_size: int = 10,
    exclude_capped: bool = False,
) -> None:
    """Core: one figure of `key` vs combo for the given (already filtered) DataFrame."""
    plot_df = df[df["graph_code"].isin(abs_codes)]
    if exclude_capped and "capped" in plot_df.columns:
        plot_df = plot_df[plot_df["capped"] == 0]
    fig, ax = plt.subplots(figsize=(12, 3.5))

    combo_groups = _combo_regime_groups(combo_order) if combo_order else [None]
    for group in combo_groups:
        group_df = plot_df if group is None else plot_df[plot_df["combo"].isin(group)]
        sns.lineplot(
            data=group_df,  # type: ignore
            x="combo",
            y=key,
            hue="graph_code",
            style="graph_code",
            hue_order=abs_codes,
            style_order=abs_codes,
            palette=palette,
            dashes=dashes,  # type: ignore
            markers=markers,
            ax=ax,
            legend=False,
        )

    if log_scale_y:
        ax.set_yscale("log")

    _add_combo_group_lines(ax, combo_order)

    _set_two_level_combo_xticks(ax, combo_order, fontsize=label_size, fig3=True)
    ax.set_xlabel("", fontsize=label_size)
    ax.set_ylabel(y_label, fontsize=label_size)

    legend_handles = []
    for code in abs_codes:
        color = palette[code]
        dash = dashes[code]
        marker = markers[code]
        ls: object = (
            "-" if dash == "" else ((0, dash) if isinstance(dash, tuple) else dash)
        )
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=color,
                linestyle=ls,
                marker=marker,
                markersize=7,
                linewidth=1.8,
                label=f"Graph {_graph_label(code)}",
            )
        )

    ax.legend(handles=legend_handles, loc="best", frameon=True, fontsize=label_size)

    out_dir = os.path.join(plots_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{key}_by_combo.pdf" if out_file is None else out_file
    path = os.path.join(out_dir, fname)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {path}")


def _parse_tau_list(value) -> list[np.ndarray]:
    """Parse a `tau_hat`/`tau_gt` cell (repr of a list of numpy arrays) back
    into an actual list of numpy arrays."""
    if isinstance(value, list):
        return value
    return eval(str(value), {"array": np.array, "__builtins__": {}})


def _match_tau_hat_to_gt(
    tau_hat: list[np.ndarray], tau_gt: list[np.ndarray]
) -> list[np.ndarray]:
    """Adjust `tau_hat` so it has exactly `len(tau_gt)` vectors.

    - If `tau_hat` has fewer vectors than `tau_gt`, pad it with zero vectors
      (of the same dimensionality as the tau_gt vectors) so the padded slots
      have zero cosine similarity with everything.
    - If `tau_hat` has more vectors than `tau_gt`, keep only the
      `len(tau_gt)` vectors with the highest best-match abs-cosine-similarity
      to any tau_gt vector, dropping the rest.
    """
    n_hat = len(tau_hat)
    n_gt = len(tau_gt)

    if n_hat == n_gt:
        return list(tau_hat)

    if n_hat < n_gt:
        dim = tau_gt[0].shape
        return list(tau_hat) + [np.zeros(dim) for _ in range(n_gt - n_hat)]

    # n_hat > n_gt: keep the vectors that best match some tau_gt vector.
    best_sim = []
    for i in range(n_hat):
        sims = [
            abs(np.dot(tau_hat[i], tau_gt[j]))
            for j in range(n_gt)
            if tau_hat[i].shape == tau_gt[j].shape
        ]
        best_sim.append(max(sims) if sims else 0.0)

    keep_idx = sorted(np.argsort(best_sim)[::-1][:n_gt])
    return [tau_hat[i] for i in keep_idx]


def _avg_abs_cosine_matrix(
    sub_df: pd.DataFrame,
    col_i: str,
    col_j: str,
    hungarian: bool = False,
) -> np.ndarray:
    """Average abs-cosine-similarity matrix between the vector lists stored
    in `col_i` and `col_j` across all rows of `sub_df`.

    The `col_i` vector list is always first padded with zero vectors (if
    shorter) or trimmed to its best-matching vectors (if longer) so it has
    exactly the same length as the `col_j` vector list. This keeps every
    row's matrix square (required for the Hungarian realignment below) and
    gives all rows a consistent shape to average over, even for instances
    where the two lists differ in length (e.g. not-found, or partially
    found, instances).

    If `hungarian` is True, each row's matrix is first individually realigned
    via the optimal linear assignment (Hungarian algorithm) so that matched
    pairs lie on the diagonal, *before* averaging across rows. This ensures
    the averaging combines already-aligned matrices instead of averaging raw
    (unaligned) matrices and aligning the result, which would wash out the
    per-instance correspondence.
    """
    from scipy.optimize import linear_sum_assignment

    c_list = []
    for _, row in sub_df.iterrows():
        vecs_i = _parse_tau_list(row[col_i])
        vecs_j = _parse_tau_list(row[col_j])

        # If dataRSE returns no solution, its output does not contain the
        # provided abstraction coefficients of the treatment and the outcome.
        # As described in the paper, we keep them in this case. They are the
        # last two vectors of the ground truth vecs_j.
        if len(vecs_i) == 0 and len(vecs_j) >= 2:
            vecs_i.append(vecs_j[-2])
            vecs_i.append(vecs_j[-1])

        # Always normalize lengths so every row's matrix is square (n_j x
        # n_j): this is required for the Hungarian realignment below, and
        # gives all rows a consistent shape so they can be averaged
        # together, even when vecs_i and vecs_j differ in length.
        vecs_i = _match_tau_hat_to_gt(vecs_i, vecs_j)

        n_j = len(vecs_j)
        c_k = np.zeros((n_j, n_j))
        for i in range(n_j):
            for j in range(n_j):
                if vecs_i[i].shape == vecs_j[j].shape:
                    c_k[i, j] = abs(np.dot(vecs_i[i], vecs_j[j]))

        if hungarian:
            # Optimal assignment (Hungarian algorithm) matching row indices
            # to column indices so as to maximize total similarity, for this
            # instance alone.
            row_ind, col_ind = linear_sum_assignment(c_k, maximize=True)
            order = np.empty_like(row_ind)
            order[col_ind] = row_ind
            c_k = c_k[order, :]

        c_list.append(c_k)

    return np.mean(c_list, axis=0)


def _render_abs_heatmap(
    c: np.ndarray,
    n_instances: int,
    title: str,
    xlabel: str,
    ylabel: str,
    out_file: str,
) -> None:
    """Plot and save the abs-cosine-similarity heatmap."""

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        c,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Absolute Cosine Similarity"},
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}")


def _plot_abs_combo(
    code: str,
    n: int,
    sub_df: pd.DataFrame,
    all_df: pd.DataFrame,
    hungarian: bool,
    out_dir: str,
) -> None:
    """Render all three abs-cosine-similarity heatmaps for one
    (graph code, n_samples) combo. Split out from `plot_abs` so each combo
    can be dispatched as an independent parallel job."""
    n_instances = len(sub_df)

    c = _avg_abs_cosine_matrix(sub_df, "tau_hat", "tau_gt", hungarian=hungarian)
    _render_abs_heatmap(
        c,
        n_instances=n_instances,
        title=f"Graph {_graph_label(code)}, n_samples={n}",
        xlabel="Ground-Truth Tau",
        ylabel="dataRSE Tau",
        out_file=os.path.join(out_dir, f"tau_{code}_n{n}.pdf"),
    )

    c_gt = _avg_abs_cosine_matrix(sub_df, "tau_gt", "tau_gt", hungarian=hungarian)
    _render_abs_heatmap(
        c_gt,
        n_instances=n_instances,
        title=f"Graph {_graph_label(code)}, n_samples={n} (Ground-Truth vs itself)",
        xlabel="Ground-Truth Tau",
        ylabel="Ground-Truth Tau",
        out_file=os.path.join(out_dir, f"tau_gt_{code}_n{n}.pdf"),
    )

    n_instances_full = len(all_df)
    c_full = _avg_abs_cosine_matrix(
        all_df, "tau_hat", "tau_gt", hungarian=hungarian
    )
    _render_abs_heatmap(
        c_full,
        n_instances=n_instances_full,
        title=f"Graph {_graph_label(code)}",
        xlabel="Ground-Truth Tau",
        ylabel="dataRSE Tau",
        out_file=os.path.join(out_dir, f"tau_full_{code}_n{n}.pdf"),
    )

    not_found_df = all_df[all_df["found"] == 0]
    if not not_found_df.empty:
        n_instances_not_found = len(not_found_df)
        c_not_found = _avg_abs_cosine_matrix(
            not_found_df, "tau_hat", "tau_gt", hungarian=hungarian
        )
        _render_abs_heatmap(
            c_not_found,
            n_instances=n_instances_not_found,
            title=f"Graph {_graph_label(code)}, n_samples={n} (not-found instances)",
            xlabel="Ground-Truth Tau",
            ylabel="dataRSE Tau",
            out_file=os.path.join(out_dir, f"tau_notfound_{code}_n{n}.pdf"),
        )
    else:
        print(f"No found=False instance for graph {code}, n_samples={n}")


def plot_abs(
    exp_name: str = "rse_ms90",
    abs_codes: list[str] = ["A", "B", "C", "D"],
    results_dir: str = "results",
    plots_dir: str = "plots",
    hungarian: bool = True,
    n_workers: int = 1,
) -> None:
    """For each graph code and sample size, take all found=True instances and
    plot the abs-cosine-similarity matrix between tau_hat and tau_gt,
    averaged across all of them.

    Also produces a reference heatmap comparing tau_gt against itself
    (averaged the same way), saved as `tau_gt_{code}_n{n}.pdf`, showing how
    distinguishable the ground-truth vectors are from one another.

    Additionally produces `tau_full_{code}_n{n}.pdf`, which uses *all*
    instances (not just found=True). Since not-found instances can have
    tau_hat with a different number of vectors than tau_gt, tau_hat is first
    adjusted to match tau_gt's length: padded with zero vectors if it has
    fewer, or trimmed to its best-matching vectors (dropping those with
    smaller cosine similarity) if it has more.

    Also produces `tau_notfound_{code}_n{n}.pdf`, the same kind of heatmap
    but restricted to found=False instances only (with the same length
    adjustment applied), showing how tau_hat diverges from tau_gt
    specifically in failure cases.

    If `hungarian` is True, each instance's matrix is individually realigned
    via the optimal linear assignment (Hungarian algorithm) so that matched
    pairs lie on the diagonal *before* averaging across instances. If False,
    the raw (unpermuted) matrices are averaged as-is.
    """
    csv_path = os.path.join(results_dir, f"{exp_name}.csv")
    df = pd.read_csv(csv_path)

    out_dir = os.path.join(plots_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    jobs: list[dict] = []
    for code in abs_codes:
        code_df = df[df["graph_code"] == code]
        for n_samples in sorted(code_df["n_samples"].unique()):
            n = int(n_samples)
            all_df = code_df[code_df["n_samples"] == n_samples]
            sub_df = all_df[all_df["found"] == 1]
            if sub_df.empty:
                print(f"No found=True instance for graph {code}, n_samples={n}")
                continue

            jobs.append(
                dict(
                    code=code,
                    n=n,
                    sub_df=sub_df,
                    all_df=all_df,
                    hungarian=hungarian,
                    out_dir=out_dir,
                )
            )

    _run_jobs(_plot_abs_combo, jobs, n_workers)


def _compact_summary_grid_figure(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    abs_codes: list[str],
    palette: dict,
    dashes: dict,
    markers: dict,
    row_specs: list[tuple[str, str]],
    log_scale_time: bool,
    out_file: str,
    combo_order: list[str] | None = None,
) -> None:
    """Grid of all graph codes overlaid per metric (2 per row), legend in
    first subplot."""

    n_cols = 2
    n_rows = -(-len(row_specs) // n_cols)  # ceil division
    h_space = 0.25
    w_space = 0.25
    w_size = 5 * n_cols + w_space * (n_cols - 1)
    h_size = 3.2 * n_rows + h_space * (n_rows - 1)
    print(f"Rendering summary grid figure {out_file} ({n_rows}x{n_cols})")
    print(f"Figure size: {h_size:.1f} x {w_size:.1f} inches")

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(w_size, h_size),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=h_space, wspace=w_space)

    for idx, (metric, y_label) in enumerate(row_specs):
        r, c = divmod(idx, n_cols)
        is_last_row = r == n_rows - 1
        ax = axes[r][c]

        combo_groups = (
            _combo_regime_groups(combo_order)
            if (x_col == "combo" and combo_order)
            else [None]
        )
        for group in combo_groups:
            plot_df = (
                df[df["graph_code"].isin(abs_codes)]
                if group is None
                else df[df["graph_code"].isin(abs_codes) & df["combo"].isin(group)]
            )
            sns.lineplot(
                data=plot_df,  # type: ignore
                x=x_col,
                y=metric,
                hue="graph_code",
                style="graph_code",
                hue_order=abs_codes,
                style_order=abs_codes,
                palette=palette,
                dashes=dashes,  # type: ignore
                markers=markers,
                ax=ax,
                legend=False,
            )

        if metric == "time" and log_scale_time:
            ax.set_yscale("log")
        if x_col == "n_samples":
            ax.set_xscale("log")

        ax.set_ylabel(y_label)

        if x_col == "combo":
            _add_combo_group_lines(ax, combo_order)
            _set_two_level_combo_xticks(ax, combo_order, grid_sizes=True)

        if is_last_row:
            ax.set_xlabel("" if x_col == "combo" else x_label)
        else:
            ax.set_xlabel("")

    # legend in the lower left corner of the first subplot (accuracy)
    legend_handles = []
    for code in abs_codes:
        color = palette[code]
        dash = dashes[code]
        marker = markers[code]
        ls: object = (
            "-" if dash == "" else ((0, dash) if isinstance(dash, tuple) else dash)
        )
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=color,
                linestyle=ls,
                marker=marker,
                markersize=7,
                linewidth=1.8,
                label=f"Graph {_graph_label(code)}",
            )
        )
    axes[0][0].legend(
        handles=legend_handles, loc="lower left", frameon=True, fontsize=9
    )

    fig.savefig(out_file, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {out_file}")


def _summary_grid_batch(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    suffix: str,
    abs_codes: list[str],
    palette: dict,
    dashes: dict,
    markers: dict,
    log_scale_time: bool,
    out_dir: str,
    combo_order: list[str] | None = None,
) -> None:
    """Produce the summary-grid figures for one (sub_df, x_col) combo. Split
    out from `summary_grid` so each combo can be dispatched as an
    independent parallel job."""
    if x_col == "n_samples":
        # over the number of samples, TCE error is more informative than
        # running time (which is essentially flat in n_samples)
        _compact_summary_grid_figure(
            df=df,
            x_col=x_col,
            x_label=x_label,
            abs_codes=abs_codes,
            palette=palette,
            dashes=dashes,
            markers=markers,
            row_specs=[('found', 'Accuracy'), ('tce_err', 'TCE Abs. Error')],
            log_scale_time=log_scale_time,
            out_file=os.path.join(out_dir, f"summary_grid_{suffix}.pdf"),
            combo_order=combo_order,
        )
        # full 2x2 grid (accuracy, time, tce, failure) over the number of
        # samples, graph codes overlaid per metric (not exploded into columns)
        _compact_summary_grid_figure(
            df=df,
            x_col=x_col,
            x_label=x_label,
            abs_codes=abs_codes,
            palette=palette,
            dashes=dashes,
            markers=markers,
            row_specs=[
                ('found', 'Accuracy'),
                ('time', 'Running Time (s)'),
                ('tce_err', 'TCE Abs. Error'),
                ('failed', 'Failure'),
            ],
            log_scale_time=log_scale_time,
            out_file=os.path.join(out_dir, f"summary_grid_full_{suffix}.pdf"),
            combo_order=combo_order,
        )
    else:
        _compact_summary_grid_figure(
            df=df,
            x_col=x_col,
            x_label=x_label,
            abs_codes=abs_codes,
            palette=palette,
            dashes=dashes,
            markers=markers,
            row_specs=[('found', 'Accuracy'), ("time", "Running Time (s)")],
            log_scale_time=log_scale_time,
            out_file=os.path.join(out_dir, f"summary_grid_{suffix}.pdf"),
            combo_order=combo_order,
        )


def summary_grid(
    abs_codes: list[str] | None,
    results_dir: str = "results",
    plots_dir: str = "plots",
    exp_name: str = "rse",
    log_scale_time: bool = False,
    n_workers: int = 1,
) -> None:
    """Summary grids of accuracy, running time, TCE error, and failure rate,
    graph codes overlaid per metric. Produces one figure per unique n_samples
    (combo x-axis) plus figures with n_samples as the x-axis."""
    csv_path = os.path.join(results_dir, f"{exp_name}.csv")
    df = pd.read_csv(csv_path)

    if abs_codes is None:
        abs_codes = ["A", "B", "C", "D"]

    palette = {
        code: _WONG_PALETTE[i % len(_WONG_PALETTE)] for i, code in enumerate(abs_codes)
    }
    dashes, markers = build_style_maps(abs_codes)

    # combo x-axis, sorted by (relevant_size, block_size)
    df["combo"] = df.apply(
        lambda row: f"({int(row['relevant_size'])}, {int(row['block_size'])})", axis=1
    )
    combo_order: list[str] = (
        df[["relevant_size", "block_size", "combo"]]
        .drop_duplicates()
        .sort_values(["relevant_size", "block_size"])["combo"]
        .tolist()
    )
    df["combo"] = pd.Categorical(df["combo"], categories=combo_order, ordered=True)

    out_dir = os.path.join(plots_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    jobs: list[dict] = []

    # one figure per n_samples value, combo on x-axis
    unique_sample_sizes: Iterable[int] = df["n_samples"].unique()
    for n_samples in sorted(unique_sample_sizes):
        sub_df = df[df["n_samples"] == n_samples]
        n = int(n_samples)
        jobs.append(
            dict(
                df=sub_df,
                x_col="combo",
                x_label="(Relevant size, Block size)",
                suffix=f"n{n}",
                abs_codes=abs_codes,
                palette=palette,
                dashes=dashes,
                markers=markers,
                log_scale_time=log_scale_time,
                out_dir=out_dir,
                combo_order=combo_order,
            )
        )

    # one figure across all data, n_samples on x-axis
    jobs.append(
        dict(
            df=df,
            x_col="n_samples",
            x_label="Number of Samples",
            suffix="nsamples",
            abs_codes=abs_codes,
            palette=palette,
            dashes=dashes,
            markers=markers,
            log_scale_time=log_scale_time,
            out_dir=out_dir,
        )
    )

    _run_jobs(_summary_grid_batch, jobs, n_workers)

def figure3(
    abs_codes: list[str] | None = None,
    results_dir: str = "results",
    plots_dir: str = "plots",
    exp_name: str = "rse",
    log_scale_y: bool = False,
) -> None:
    """MiSE size vs (relevant_size, block_size) combos, one line per graph
    code coloured distinctly."""
    if abs_codes is None:
        abs_codes = ["A", "B", "C", "D"]

    csv_path = os.path.join(results_dir, f"{exp_name}.csv")
    df = pd.read_csv(csv_path)

    df["combo"] = df.apply(
        lambda row: f"({int(row['relevant_size'])}, {int(row['block_size'])})", axis=1
    )
    combo_order: list[str] = (
        df[["relevant_size", "block_size", "combo"]]
        .drop_duplicates()
        .sort_values(["relevant_size", "block_size"])["combo"]
        .tolist()
    )
    df["combo"] = pd.Categorical(df["combo"], categories=combo_order, ordered=True)

    palette = {
        code: _WONG_PALETTE[i % len(_WONG_PALETTE)] for i, code in enumerate(abs_codes)
    }
    dashes, markers = build_style_maps(abs_codes)

    _plot_by_combo(
        df=df,
        key="n_solutions",
        y_label="MiSE Size",
        abs_codes=abs_codes,
        plots_dir=plots_dir,
        exp_name=exp_name,
        combo_order=combo_order,
        palette=palette,
        dashes=dashes,
        markers=markers,
        log_scale_y=log_scale_y,
        out_file="figure3.pdf",
        label_size=13,
    )


# ---------------------------------------------------------------------------
# Comparison of the rules selecting the number of clusters in dataRSE
# ---------------------------------------------------------------------------

# As in the other figures of the paper, each graph keeps its color from the
# Wong palette (by position in the figure); the two rules are distinguished
# by line style and marker, so that the comparison reads in grayscale too.
_RULE_STYLE = {
    "rrr": {
        "dashes": "",
        "marker": "o",
        "fill": True,
        "label": "Least squares + BIC",
    },
    "plateau": {
        "dashes": (4, 1.5),
        "marker": "s",
        "fill": False,
        "label": "Rank-one partitioning",
    },
}
_RULE_ORDER = ["rrr", "plateau"]
_KEYS = ["graph_code", "block_size", "relevant_size", "repeat", "n_samples"]

# display names of the graph codes: Graph I of config/graphs.yaml is shown
# as Graph H in the paper
_GRAPH_LABELS = {"I": "H"}


def _graph_label(code: str) -> str:
    return _GRAPH_LABELS.get(code, code)


def _paired_time_ratio(df: pd.DataFrame, num: str, den: str) -> pd.DataFrame:
    """Per-run ratio of the running times of two variants on the same data."""
    wide = df.pivot_table(index=_KEYS, columns="variant", values="time").reset_index()
    wide = wide.dropna(subset=[num, den])
    wide["time_ratio"] = wide[num] / wide[den].clip(lower=1e-12)
    return wide


def _rule_line(ax, data, y, color, rule):
    style = _RULE_STYLE[rule]
    sns.lineplot(
        data=data,
        x="n_samples",
        y=y,
        color=color,
        marker=style["marker"],
        markersize=7,
        markerfacecolor=color if style["fill"] else "white",
        markeredgecolor=color,
        markeredgewidth=1.5,
        linewidth=1.8,
        errorbar=("ci", 95),
        ax=ax,
    )
    if style["dashes"]:
        ax.lines[-1].set_dashes(style["dashes"])


def _compare_column(axes, df_g, ratio_g, variants, color, show_ylabels, label_size):
    """Fill one column of panels (one graph) of the comparison figure."""
    rows = [
        ("found", "Accuracy", False),
        ("tce_err", "TCE Abs. Error", True),
        ("time", "Running Time (s)", True),
    ]
    for ax, (metric, y_label, log_y) in zip(axes, rows):
        for rule in variants:
            _rule_line(ax, df_g[df_g["variant"] == rule], metric, color, rule)
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        if metric == "found":
            ax.set_ylim(0, 1.02)
        ax.set_ylabel(y_label if show_ylabels else "", fontsize=label_size)
        ax.set_xlabel("")

    # paired slowdown of rrr with respect to plateau
    ax = axes[len(rows)]
    if ratio_g is not None and len(ratio_g):
        _rule_line(ax, ratio_g, "time_ratio", color, "rrr")
    ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Time Ratio (LS+BIC / Rank-one)" if show_ylabels else "", fontsize=label_size)
    # plain tick labels (e.g., 5, 10, 20) instead of powers of ten
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Number of Samples", fontsize=label_size)


def _compare_legend(fig, variants, label_size):
    handles = []
    for rule in variants:
        style = _RULE_STYLE[rule]
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="#333333",
                marker=style["marker"],
                markerfacecolor="#333333" if style["fill"] else "white",
                markeredgecolor="#333333",
                markersize=7,
                linewidth=1.8,
                linestyle="-" if not style["dashes"] else (0, style["dashes"]),
                label=style["label"],
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=label_size,
        bbox_to_anchor=(0.5, 1.0),
    )


def plot_compare(
    exp_name: str = "compare_rule",
    abs_codes: list[str] | None = None,
    groups: list[list[str]] | None = None,
    results_dir: str = "results",
    plots_dir: str = "plots",
    label_size: int = 10,
) -> None:
    """
    Compare the rrr and plateau rules of dataRSE, run on the same ground
    truth and data (``run --variants='["plateau","rrr"]'``). For each graph,
    plots accuracy, TCE error, and running time against the number of
    samples, averaged over block and relevant sizes (95% CI), plus the paired
    ratio of running times rrr / plateau.

    Saves one figure per group of graphs (default: A-D and E-G plus the
    counterexample I, shown as H), named compare_rule_<labels>.pdf, one
    figure per graph, and a summary table.
    """
    df = pd.read_csv(os.path.join(results_dir, f"{exp_name}.csv"))
    variants = [v for v in _RULE_ORDER if v in set(df["variant"])]
    if not variants:
        raise ValueError("No plateau/rrr variants found in the results")
    available = list(df["graph_code"].unique())
    if abs_codes is None:
        abs_codes = sorted(available)
    if groups is None:
        groups = [
            [c for c in ["A", "B", "C", "D"] if c in abs_codes],
            [c for c in ["E", "F", "G", "I"] if c in abs_codes],
        ]
        groups = [g for g in groups if g]

    ratio = (
        _paired_time_ratio(df, "rrr", "plateau")
        if {"plateau", "rrr"} <= set(variants)
        else None
    )

    out_dir = os.path.join(plots_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    def _figure(codes: list[str], out_file: str) -> None:
        n_cols = len(codes)
        fig, axes = plt.subplots(
            nrows=4,
            ncols=n_cols,
            figsize=(3.4 * n_cols, 10.5),
            squeeze=False,
            sharex=True,
        )
        fig.subplots_adjust(hspace=0.12, wspace=0.28, top=0.93)
        for c, code in enumerate(codes):
            df_g = df[(df["graph_code"] == code) & (df["variant"].isin(variants))]
            ratio_g = None if ratio is None else ratio[ratio["graph_code"] == code]
            _compare_column(
                [axes[r][c] for r in range(4)],
                df_g,
                ratio_g,
                variants,
                color=_WONG_PALETTE[c % len(_WONG_PALETTE)],
                show_ylabels=(c == 0),
                label_size=label_size,
            )
            axes[0][c].set_title(f"Graph {_graph_label(code)}", fontsize=label_size)
        _compare_legend(fig, variants, label_size)
        path = os.path.join(out_dir, out_file)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(f"Saved {path}")

    for codes in groups:
        labels = "".join(_graph_label(c) for c in codes)
        _figure(codes, f"compare_rule_{labels}.pdf")
    for code in abs_codes:
        _figure([code], f"compare_rule_graph_{_graph_label(code)}.pdf")

    # summary table: mean accuracy and median running time per graph and n
    table = (
        df[df["variant"].isin(variants)]
        .assign(graph=lambda d: d["graph_code"].map(_graph_label))
        .groupby(["graph", "n_samples", "variant"])
        .agg(accuracy=("found", "mean"), median_time=("time", "median"))
        .unstack("variant")
    )
    table_path = os.path.join(out_dir, "compare_rule.csv")
    table.to_csv(table_path)
    print(table.to_string())
    print(f"Saved {table_path}")
