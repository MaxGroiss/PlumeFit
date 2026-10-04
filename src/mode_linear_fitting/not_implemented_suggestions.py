# THIS FILE IS NOT PART OF THE IMPLEMENTATION AND IT IS CLEARLY STATED THAT IT ONLY CONTAINS CODE WRITTEN BY AI

# This file contains code suggestion coming from AI Reviews of the pipeline code, these suggestions were not implemented
# due to complexity, not trivial to understand, would not have come to that solution by hand. As code in this file
# goes beyond my understanding I cant explain it therefore it is kept outside the Thesis but provided for potential
# optimization outside the thesis scope.


#AI-Assisted: <Fable 5> ; (Implementation came from a Code Review of the pipeline refine_positions)
# This implementation of refine positions drops a significant amount of runtime, it uses complex bounds to reduce the
# number of fits through the grid walk. It was not implemented due to lack of understanding the mechanisms behind it.

# This optimization refines the filter approach already used in the pipeline code to further reduce bvls fits,
# on co2 refinements it barely makes a diffrence as co2 plumes generally have a significant plume that enables the filter to
# reduce the needed bvls checks on positions suspected to lower the sse significantly, as pollutant peaks are genereally
# less prominent and often near zero or noisier than co2 peaks the filter implemented in the pipeline can barely reduce the number
# of needed fits for dsse checking. This is caused by the ols lower bounds following noise and assigning negative values for parameter optimization.
# The refined filter has a tighter filter with the capability to reduce bvls checks on pollutant
# data significantly. A Runtime Example from a Dev Repo showed that the pipeline Filter needs 3,5 Seconds on a segment with 30
# candidates in co2 the refined needs 2,9 seconds (about half the number of fits) and on pollutant the pipeline refine
# takes 84 seconds so basically the full Gridwalk without a Filter (pollutants are not position refined in the pipeline) and with the refined filter 2,7 seconds.
# Both filter Methodes converge on the same position for the plumes.
def refine_positions(y: np.ndarray, templates: list[ShapeTemplate], positions: np.ndarray,
                     lo: np.ndarray, hi: np.ndarray, dt: float, baseline: str | None,
                     sigma2: float, sweeps: int = 10, min_dsse: float = 8.0) -> np.ndarray:
    """Refines the peak positions per pass by a coordinate-descent SSE grid search.


    :param y: Signal of the segment
    :param templates: One template per pass in the segment
    :param positions: rel peak positions in the segment (median placed)
    :param lo: Lowest allowed sample index for pass i
    :param hi: Highest allowed sample index for pass i
    :param dt: Sampling time
    :param baseline: Baseline Methode
    :param sigma2: Noise variance of the vehicle-free day residual (the acceptance threshold
                   must not come from the start-fit residual: a badly placed template inflates
                   that residual with model error and would block its own correction)
    :param sweeps: Maximum number of sweeps, stops early once a sweep moves nothing
    :param min_dsse: a move is accepted only if SSE drops by more than min_dsse * sigma2
    :return: Refined peak positions (same shape as positions)
    """
    # Peak Positions
    pos = np.array(positions, dtype=int)
    n = y.size
    valid = np.isfinite(y)
    yv = y[valid]
    yy = float(yv @ yv)
    thr = min_dsse * sigma2

    def fit(p: np.ndarray):
        try:
            return fit_mode_l(y, templates, p, dt, baseline=baseline, with_se=False)
        except ValueError:
            return None

    # Initial fit without refinement, used for deciding amplitude hierarchic
    first = fit_mode_l(y, templates, pos, dt, baseline=baseline, with_se=False)
    # Processing order from highest to lowest amplitude
    order = np.argsort(-first.amplitudes)
    # Current SSE that needs to be beaten to justify a move
    cur = first.sse
    # Design matrix (valid rows) and its normal equations, kept up to date with the moves;
    # column i is the template of pass i, the baseline columns come last
    A = np.column_stack([template_column(t, p, n, dt) for t, p in zip(templates, pos)]
                        + baseline_columns(n, baseline))[valid]
    G, b = A.T @ A, A.T @ yv
    m = A.shape[1]

    def multipliers(f) -> np.ndarray:
        # KKT multipliers of the amplitudes clamped at 0 by the bound (SSE gradient, >= 0 at the
        # BVLS optimum). Weak duality: for ANY lambda >= 0 the value y'y - (b+lambda)' G^-1 (b+lambda)
        # is a lower bound of the BVLS SSE. With lambda = 0 this is the plain OLS bound, which is far
        # too loose on noisy channels where most amplitudes are clamped (the search then verifies
        # nearly every candidate); with the current multipliers the bound is tight as long as the
        # active set does not change with the move
        x = np.r_[f.amplitudes, f.baseline_coeffs]
        lam = np.zeros(m)
        clamped = np.r_[f.amplitudes <= 0.0, np.zeros(m - len(templates), dtype=bool)]
        lam[clamped] = np.maximum((G @ x - b)[clamped], 0.0)
        return lam

    lam = multipliers(first)
    for _ in range(sweeps):
        moved = False
        for i in order:
            # Dual value of the problem without pass i (see multipliers): coefficients of the
            # remaining columns and the SSE lower bound they leave
            # ponytail: pinv per pass is O(m^3), fine up to a few hundred passes per segment
            idx = np.r_[0:i, i + 1:m]
            g_inv = np.linalg.pinv(G[np.ix_(idx, idx)])
            w = (b + lam)[idx]
            c = g_inv @ w
            sse_wo = yy - float(w @ c)
            # Candidate columns of pass i inside its band (start value excluded,
            # template too far outside the window -> candidate skipped)
            cands, cols = [], []
            for p in range(max(int(lo[i]), 0), min(int(hi[i]), n - 1) + 1):
                col = template_column(templates[i], p, n, dt) if p != pos[i] else None
                if col is not None:
                    cands.append(p)
                    cols.append(col[valid])
            if not cands:
                continue
            S = np.column_stack(cols)
            # Closed-form gain of adding column s: (s'Py)^2 / (s'Ps) with P the projector orthogonal
            # to the other columns (Schur complement of the dual value). A negative numerator means
            # the multiplier of the moved column itself is optimal at |numerator| -> no gain, a
            # column inside the span of the others (den ~ 0) -> no gain
            V = A[:, idx].T @ S
            ss = np.einsum("ij,ij->j", S, S)
            num = S.T @ yv - V.T @ c
            den = ss - np.einsum("ij,ij->j", V, g_inv @ V)
            gain = np.where((num > 0) & (den > 1e-9 * ss), num ** 2 / np.maximum(den, 1e-300), 0.0)
            # Upper bound of the BVLS improvement per candidate
            bound = cur - (sse_wo - gain)
            best_gain, best_p, best_fit = 0.0, pos[i], None
            for j in np.argsort(-bound):
                # Neither the acceptance threshold nor the best verified move can be beaten -> stop
                if bound[j] <= max(thr, best_gain):
                    break
                trial = pos.copy()
                trial[i] = cands[j]
                f = fit(trial)
                if f is not None and cur - f.sse > best_gain:
                    best_gain, best_p, best_fit = cur - f.sse, cands[j], f
            # Accept only improvements that noise alone could not explain
            if best_gain > thr:
                pos[i], cur, moved = best_p, cur - best_gain, True
                A[:, i] = template_column(templates[i], best_p, n, dt)[valid]
                G[i, :] = G[:, i] = A.T @ A[:, i]
                b[i] = A[:, i] @ yv
                lam = multipliers(best_fit)
        if not moved:
            break
    return pos