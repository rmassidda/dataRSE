"""
Copyright (c) 2026 Riccardo Massidda
Released under the MIT license, see LICENSE.

Implementation of the data-driven RSE algorithm (dataRSE).
"""

from typing import Generator

import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from causabs.graph import abstract_graph, blocks_from_relevants, graphical_consistency
from causabs.rse.rse import get_new_relevants
from causabs.rse.utils import (
    abstract_intersections,
    canonical,
    cnc_intersection,
)


def _solve_regression(X: np.ndarray, y: np.ndarray, lambda_ridge: float) -> np.ndarray:
    """OLS (lambda_ridge<=0) or standardized ridge regression of y on X."""
    if lambda_ridge <= 0:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    else:
        X_std = X.std(axis=0, keepdims=True)
        X_std = np.where(X_std < 1e-12, 1.0, X_std)
        X_scaled = X / X_std
        XtX = X_scaled.T @ X_scaled + lambda_ridge * np.eye(X_scaled.shape[1])
        XtY = X_scaled.T @ y
        coeffs = np.linalg.solve(XtX, XtY)
        coeffs = coeffs / X_std.flatten()  # rescale back
    return coeffs


def recover_anm_coefficients(
    X: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    parent_mask: np.ndarray,
    z_mask: np.ndarray | None = None,
    lambda_ridge: float = 1e-5,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover structural coefficients in a linear ANM where:
        - X: (n, p) new relevant variables, available as regressors subject
          to z_mask, since their parent structure is not yet known.
        - W: (n, r) known abstract variables -- the known parents K together
          with the targets Y themselves, in the same causal (topological)
          order -- available as regressors subject to parent_mask.
        - Y: (n, q) target variables V, in the same causal order as the V
          part of W.
        - parent_mask: (r, q) boolean; parent_mask[i, k] is True iff W[:, i]
          is an actual parent, in the abstract graph, of Y[:, k] (always
          False for Y[:, k]'s own column and any later column in causal
          order).
        - z_mask: (p, q) boolean; z_mask[l, k] is True iff X[:, l] targets
          Y[:, k], i.e., X[:, l] is in Z^{(k)}. Only these new relevant
          variables are regressors of Y[:, k], since the others have zero
          coefficient and might be correlated with its exogenous term. If
          None, every new relevant variable is a regressor of every target.
        - lambda_ridge: regularization strength for ridge regression.

    Model for each k:
        Y[:,k] = X[:, z_mask[:,k]] @ A[z_mask[:,k], k]
                 + W[:, parent_mask[:,k]] @ M[parent_mask[:,k], k] + eps_k

    Returns:
        A: (p, q)
            matrix where A[:, k] are the direct effects of X on Y[:, k];
            A[l, k] = 0 whenever z_mask[l, k] is False
        M: (r, q)
            matrix where M[i, k] is the direct effect of W[:, i] on Y[:, k];
            M[i, k] = 0 whenever parent_mask[i, k] is False
    """
    assert (
        X.shape[0] == W.shape[0] == Y.shape[0]
    ), "X, W, and Y must have the same number of samples"
    assert lambda_ridge >= 0, "lambda_ridge must be non-negative"
    p = X.shape[1]
    r = W.shape[1]
    q = Y.shape[1]

    if z_mask is None:
        z_mask = np.ones((p, q), dtype=bool)
    assert z_mask.shape == (p, q), "z_mask must have shape (p, q)"

    A = np.zeros((p, q))
    M = np.zeros((r, q))

    for k in range(q):
        x_idx = np.where(z_mask[:, k])[0]
        w_idx = np.where(parent_mask[:, k])[0]
        X_aug = np.column_stack([X[:, x_idx], W[:, w_idx]])
        if X_aug.shape[1] == 0:
            continue
        coeffs = _solve_regression(X_aug, Y[:, k], lambda_ridge)

        A[x_idx, k] = coeffs[: len(x_idx)]
        M[w_idx, k] = coeffs[len(x_idx) :]

        if verbose:
            print(f"Recovered coefficients for target {k}:")
            print(f"A[:, {k}] = {A[:, k]}")
            print(f"M[{list(w_idx)}, {k}] = {M[w_idx, k]}")

    return A, M


def extract_abstract_coefficients(
    A: np.ndarray, relevants: list[list[int]]
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Extract abstract coefficients for each group of relevant rows."""
    tau: dict[int, np.ndarray] = {}
    abs_coeff: dict[int, np.ndarray] = {}
    for y, r_y in enumerate(relevants):
        sub = A[r_y, :]  # (|R_y|, q)

        # robust direction via first left singular vector
        U, sigma, Vt = np.linalg.svd(sub, full_matrices=False)

        # signed projections s_i = a_{g,i} * ||b_g||
        ty = U[:, 0]
        my = sigma[0] * Vt[0]

        # sign convention: largest |a| is positive
        if ty[np.argmax(np.abs(ty))] < 0:
            ty = -ty
            my = -my

        tau[y] = ty
        abs_coeff[y] = my

    return tau, abs_coeff


def cluster_columns(A: np.ndarray, k: int) -> list[list[int]]:
    """
    Cluster the rows of the matrix A into k clusters
    composed of mutually collinear rows.

    Parameters
    ----------
    A : (p, q) coefficient matrix.
    k : number of clusters.

    Returns
    -------
    labels : (p,) integer array of cluster assignments.
    """
    p = A.shape[0]

    # When k >= p every row is its own cluster, no clustering needed.
    if k >= p:
        return [[i] for i in range(p)]

    # normalize the rows
    A_norm = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    # compute the similarity matrix
    S = np.abs(A_norm @ A_norm.T)
    S = np.clip(S, 0, 1)
    # compute the distance matrix
    D = 1 - S

    clustering = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="complete"
    )

    labels = clustering.fit_predict(D)

    clusters = [list(np.where(labels == j)[0]) for j in range(labels.max() + 1)]

    return clusters


def _fit_rank_one_clusters(
    X: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    parent_mask: np.ndarray,
    z_mask: np.ndarray | None,
    clusters: list[list[int]],
    t_init: list[np.ndarray],
    n_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[np.ndarray, int, list[np.ndarray]]:
    """
    Least-squares fit of the k-cluster model (reduced-rank regression):

        Y[:, j] = sum_h m_hj X[:, P_h] t_h + W[:, pa(j)] M[pa(j), j] + eps_j,

    where m_hj = 0 whenever no variable of P_h targets Y[:, j] (z_mask).
    Alternates between the regression of each target on the induced abstract
    variables X[:, P_h] t_h and its parents, and a weighted least-squares
    update of the abstraction vectors t_h, with weights 1 / sigma_j^2 to
    approximate the maximum-likelihood fit. Starts from t_init (e.g., the
    first left singular vectors of the blocks of A_hat).

    Returns the residual variance of each target and the number of free
    parameters of the abstract variables (sum_h (|P_h| - 1) + number of
    non-zero m_hj), which is used by BIC, and the fitted abstraction vectors
    (unit norm, largest entry in absolute value positive).
    """
    n, p = X.shape
    q = Y.shape[1]
    k = len(clusters)
    zm = np.ones((p, q), dtype=bool) if z_mask is None else z_mask
    allowed = np.array([[bool(zm[c, j].any()) for j in range(q)] for c in clusters])

    order = [l for c in clusters for l in c]
    owner = np.array([h for h, c in enumerate(clusters) for _ in c])
    X_ord = X[:, order]
    G = X_ord.T @ X_ord
    t = [np.asarray(t0, dtype=float).copy() for t0 in t_init]

    weights = np.ones(q)
    prev_obj = np.inf
    rss = np.zeros(q)
    for _ in range(n_iter):
        # m-step: regression of each target on the induced variables and
        # its parents
        Y_h = np.column_stack([X[:, c] @ t[h] for h, c in enumerate(clusters)])
        m = np.zeros((k, q))
        M_par = np.zeros((W.shape[1], q))
        residuals = np.empty_like(Y)
        for j in range(q):
            hs = np.where(allowed[:, j])[0]
            ws = np.where(parent_mask[:, j])[0]
            D = np.column_stack([Y_h[:, hs], W[:, ws]])
            if D.shape[1] == 0:
                residuals[:, j] = Y[:, j]
                continue
            coef, _, _, _ = np.linalg.lstsq(D, Y[:, j], rcond=None)
            m[hs, j] = coef[: len(hs)]
            M_par[ws, j] = coef[len(hs) :]
            residuals[:, j] = Y[:, j] - D @ coef
        rss = (residuals**2).sum(axis=0) / n
        obj = float(np.sum(np.log(np.maximum(rss, 1e-300))))
        if prev_obj - obj < tol * max(abs(prev_obj), 1.0):
            break
        prev_obj = obj
        weights = 1.0 / np.maximum(rss, 1e-300)

        # t-step: weighted least squares for the stacked abstraction vectors
        R = Y - W @ M_par
        XtR = X_ord.T @ R
        S = np.zeros((p, p))
        b = np.zeros(p)
        for j in range(q):
            s = m[owner, j]
            S += weights[j] * np.outer(s, s) * G
            b += weights[j] * s * XtR[:, j]
        S += 1e-10 * max(np.trace(S) / p, 1e-300) * np.eye(p)
        t_vec = np.linalg.solve(S, b)
        for h in range(k):
            t_h = t_vec[owner == h]
            norm = np.linalg.norm(t_h)
            t[h] = t_h / norm if norm > 0 else t[h]

    n_params = int(sum(len(c) - 1 for c in clusters) + allowed.sum())
    # sign convention, as in extract_abstract_coefficients: largest |t| positive
    t = [t_h if t_h[np.argmax(np.abs(t_h))] >= 0 else -t_h for t_h in t]
    return rss, n_params, t


def collinearity_constraints(
    cnc_data: np.ndarray,
    tau: dict[int, np.ndarray],
    new_rels_list: list[int],
    intersection: set[int],
    relevants: list[list[int]],
    abs_graph: nx.DiGraph,
    max_collinearity_constraints: int | None = None,
    delta: float = 0.25,
    lambda_ridge: float = 1e-5,
    verbose: bool = False,
    blocks: list[list[int]] | None = None,
    k_selection: str = "rrr",
    fitted_tau_out: dict | None = None,
) -> tuple[list[list[list[int]]], np.ndarray]:
    """
    Estimate the composed abstraction matrix A_hat and the candidate
    partitions of the new relevant variables (maximal partitioning).

    k_selection chooses the number of clusters k*:
        - "rrr": BIC-based maximal partitioning (default, used in the main
          experiments). The k-cluster model is fitted by least squares
          (reduced-rank regression, see _fit_rank_one_clusters), and k*
          minimizes BIC(k) = n sum_j log sigma_j^2(k) + log(n) * n_params(k);
        - "plateau": rank-based partitioning (appendix). The k-cluster model
          uses the rank-one factors of the blocks of A_hat (SVD), and k* is
          the first k of the last run of consecutive |l(k+1) - l(k)| < delta.

    fitted_tau_out: if given and k_selection is "rrr", it is filled with the
    least-squares abstraction vectors of the selected partition, as a map
    from the tuple of the variables of each relevant set (in the order of
    the returned candidate) to its abstraction vector.
    """
    assert k_selection in ("plateau", "rrr"), k_selection
    # number of new relevant variables
    X = cnc_data[:, new_rels_list]
    n, p = X.shape

    # order V topologically in abs_graph, so that "earlier" columns of Y
    # coincide with actual ancestors, never descendants
    v_list = [y for y in nx.topological_sort(abs_graph) if y in intersection]
    q = len(v_list)
    Y = np.zeros((n, q))
    for idx, Yj in enumerate(v_list):
        X_Ry = cnc_data[:, relevants[Yj]]
        Y[:, idx] = X_Ry @ tau[Yj]

    # K: already-known abstract variables with a direct edge, in abs_graph,
    # into some Yj in V. They must be adjusted for when recovering A, M, to
    # avoid omitted-variable bias (e.g. a variable confounding two separate
    # pairs resolved at different recursion depths), even though their own
    # coefficients are discarded afterward, since K's structure is already
    # known and does not need to be rediscovered.
    known_parents = set().union(
        *(set(abs_graph.predecessors(yj)) for yj in intersection)
    ) - intersection
    k_list = sorted(known_parents)

    # abstract realizations of K
    Y_K = np.zeros((n, len(k_list)))
    for idx, Yk in enumerate(k_list):
        X_Rk = cnc_data[:, relevants[Yk]]
        Y_K[:, idx] = X_Rk @ tau[Yk]

    # W = K columns followed by V columns (in topological order): both are
    # available as regressors for each Yj in V, subject to parent_mask, which
    # restricts them to the *actual* parents of Yj in abs_graph, rather than
    # blanket-including every known variable or every causally-prior V member.
    W = np.column_stack([Y_K, Y])
    r = len(k_list) + q
    parent_mask = np.zeros((r, q), dtype=bool)
    for col, Yk in enumerate(v_list):
        parents_Yk = set(abs_graph.predecessors(Yk))
        for i, Ki in enumerate(k_list):
            if Ki in parents_Yk:
                parent_mask[i, col] = True
        for j, Yj in enumerate(v_list[:col]):
            if Yj in parents_Yk:
                parent_mask[len(k_list) + j, col] = True

    # Z^{(k)}: new relevant variables targeting Yk, i.e., in its block. The
    # others have zero coefficient and are excluded from the regression of
    # Yk, since they might be descendants of Yk and hence correlated with
    # its exogenous term.
    z_mask = None
    if blocks is not None:
        z_mask = np.zeros((p, q), dtype=bool)
        for col, Yk in enumerate(v_list):
            block_Yk = set(blocks[Yk])
            for l, xl in enumerate(new_rels_list):
                z_mask[l, col] = xl in block_Yk

    X_c = X - X.mean(axis=0)
    W_c = W - W.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)

    A, M = recover_anm_coefficients(
        X_c,
        W_c,
        Y_c,
        parent_mask,
        z_mask=z_mask,
        lambda_ridge=lambda_ridge,
        verbose=verbose,
    )

    likelihoods: list[float] = []
    candidates: list[list[list[int]]] = []
    differences: list[float] = []
    bics: list[float] = []
    rrr_taus: list[list[np.ndarray]] = []

    for k in range(1, p + 1):
        if verbose:
            print(f"Testing k={k} collinearity constraints...")
        # compose clusters with their indexes in the new_rels_list
        clusters_rel_idx: list[list[int]] = cluster_columns(A, k)

        # extract tau
        tau_hat, m_hat = extract_abstract_coefficients(A, clusters_rel_idx)
        tau_hat = {key + len(relevants): value for key, value in tau_hat.items()}
        tau_hat.update(tau.copy())

        # compute the residual (W_c @ M accounts for both K's and the
        # within-V causal structure in one unified term)
        residuals = Y_c - W_c @ M
        for h, cluster in enumerate(clusters_rel_idx):
            X_Rh = X_c[:, cluster]
            h_idx = h + len(relevants)
            Y_h = X_Rh @ tau_hat[h_idx]
            residuals -= m_hat[h] * Y_h[:, np.newaxis]

        # MLE variance per target: sigma_k^2 = ||eps_k||^2 / n
        sigma = (residuals**2).sum(axis=0) / n  # (q,)
        if k_selection == "rrr":
            # least-squares fit of the k-cluster model, started from the
            # rank-one factors of A_hat
            t_init = [tau_hat[h + len(relevants)] for h in range(len(clusters_rel_idx))]
            sigma, n_params, t_fit = _fit_rank_one_clusters(
                X_c, W_c, Y_c, parent_mask, z_mask, clusters_rel_idx, t_init
            )
            rrr_taus.append(t_fit)
            bics.append(
                float(
                    n * np.sum(np.log(np.maximum(sigma, 1e-300)))
                    + np.log(n) * n_params
                )
            )
        # -2 log-likelihood (dropping the constant nq(1 + log(2pi)))
        nll = -2 * np.sum(np.log(sigma))

        # compose clusters with their actual variable indices
        clusters = [
            [new_rels_list[xi] for xi in cluster] for cluster in clusters_rel_idx
        ]

        if verbose:
            print(f"k={k}, nll={nll:.2f}")
            with np.printoptions(precision=3, suppress=True):
                for y, ty in tau_hat.items():
                    print(f"tau_hat[{y}] = {ty}")
                for y, my in m_hat.items():
                    print(f"m_hat[{y}] = {my}")
                for y, ry in enumerate(clusters):
                    print(f"R{y} = {ry}")

        if len(likelihoods) > 0:
            differences.append(nll - likelihoods[-1])
        likelihoods.append(nll)
        candidates.append(clusters)

    if k_selection == "rrr" and len(bics) > 0:
        # rrr: minimum BIC over the candidate partitions
        k_star = int(np.argmin(bics)) + 1
        if verbose:
            print(f"Selected k*={k_star} by BIC: {np.round(bics, 2)}")
    else:
        # plateau: first k of the last run of consecutive |l(k+1) - l(k)| < delta
        k_star_arrays = []
        new_array = True
        for k, delta_nll in enumerate(differences):
            if abs(delta_nll) < delta:
                if new_array:
                    k_star_arrays.append([k + 1])
                    new_array = False
                else:
                    k_star_arrays[-1].append(k + 1)
            else:
                new_array = True
        if len(k_star_arrays) > 0:
            k_star = k_star_arrays[-1][0]
        else:
            k_star = len(differences) + 1
        if verbose:
            print(f"Selected k*={k_star} by the NLL plateau rule.")

    # least-squares abstraction vectors of the selected partition
    if fitted_tau_out is not None and k_selection == "rrr" and rrr_taus:
        for cluster, t_h in zip(candidates[k_star - 1], rrr_taus[k_star - 1]):
            fitted_tau_out[tuple(cluster)] = t_h

    candidates_next = candidates[k_star - 1:]
    candidates_prev = list(reversed(candidates[:k_star - 1]))
    candidates = candidates_next + candidates_prev
    if verbose:
        print(f"Candidates reordered: {candidates}")

    if max_collinearity_constraints is not None:
        candidates = candidates[:max_collinearity_constraints]

    return candidates, A


def data_RSE(
    cnc_graph: nx.DiGraph,
    cnc_data: np.ndarray,
    tau: dict[int, np.ndarray],
    relevants: list[list[int]],
    max_collinearity_constraints: int | None = None,
    delta: float = 0.25,
    lambda_ridge: float = 1e-5,
    verbose: bool = False,
    k_selection: str = "rrr",
) -> Generator[tuple[list[list[int]], dict[int, np.ndarray]], None, None]:
    """
    Data-driven RSE algorithm (dataRSE).

    k_selection is "rrr" for the BIC-based maximal partitioning (default)
    or "plateau" for the rank-based partitioning of the appendix.
    """
    # stop if not graphically consistent
    if not graphical_consistency(cnc_graph, relevants):
        return

    # compute block intersections
    blocks = blocks_from_relevants(cnc_graph, relevants)
    intersections = abstract_intersections(blocks)

    # found a sufficient solution
    if len(intersections) == 0:
        yield (relevants, tau)
        return

    # get intersected and non intersected vars
    all_vars = set(cnc_graph.nodes())
    all_intersection_vars = set().union(
        *(cnc_intersection(i, blocks) for i in intersections)
    )
    non_intersected_vars = all_vars - all_intersection_vars

    v_set: set[int] = set()
    z_set: set[int] = set()

    for intersection in intersections:
        # select low-level vars in the intersection
        intersected_vars = cnc_intersection(intersection, blocks)

        # select candidate relevants
        new_relevants = get_new_relevants(
            intersected_vars, non_intersected_vars, cnc_graph, verbose
        )

        v_set = v_set | intersection
        z_set = z_set | new_relevants

    z_list = sorted(z_set)

    # explicitly build the abstract graph on the currently known abstract
    # variables; computed for free from cnc_graph and relevants, no extra
    # data needed, and used to determine the actual (not blanket) parents
    # of each confounded variable, both within V and among the known
    # variables outside V (K).
    abs_graph = abstract_graph(cnc_graph, relevants)

    seen = set()

    # least-squares abstraction vectors of the selected partition (rrr only)
    fitted_tau: dict | None = {} if k_selection == "rrr" else None

    # iterate over BIC-sorted collinearity candidates
    constraints, A = collinearity_constraints(
        cnc_data,
        tau,
        z_list,  # new relevants
        v_set,  # abstract node indices
        relevants,
        abs_graph,
        max_collinearity_constraints,
        delta=delta,
        lambda_ridge=lambda_ridge,
        verbose=verbose,
        blocks=blocks,
        k_selection=k_selection,
        fitted_tau_out=fitted_tau,
    )
    for new_abs in constraints:
        # build the extended abstract graph with the new abstract variables
        ext_graph = abstract_graph(cnc_graph, relevants + new_abs)
        b = len(relevants)
        # keep only new abstract variables not feeding into other new ones
        new_abs = [
            r_h
            for h, r_h in enumerate(new_abs)
            if all(y < b for y in ext_graph.successors(b + h))
        ]
        # avoid empty new abstract variables
        if len(new_abs) == 0:
            continue

        # avoid duplicates
        partition_key = canonical(new_abs)
        if partition_key in seen:
            continue
        seen.add(partition_key)

        # extract tau for this specific partition from the shared A matrix
        new_abs_idx = [[z_list.index(var) for var in ry] for ry in new_abs]
        tau_hat, _ = extract_abstract_coefficients(A, new_abs_idx)
        tau_hat = {key + len(relevants): value for key, value in tau_hat.items()}
        # replace them with the least-squares ones, when available
        if fitted_tau:
            for h, ry in enumerate(new_abs):
                if tuple(ry) in fitted_tau:
                    tau_hat[h + len(relevants)] = fitted_tau[tuple(ry)]
        tau_hat.update(tau.copy())

        for sol in data_RSE(
            cnc_graph,
            cnc_data,
            tau_hat,
            relevants + new_abs,
            max_collinearity_constraints,
            delta,
            lambda_ridge,
            verbose,
            k_selection=k_selection,
        ):
            # stop after the first complete solution
            yield sol
            return
