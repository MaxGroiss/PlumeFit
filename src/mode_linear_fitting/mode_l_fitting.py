"""Mode L fitting: plume area per vehicle pass by bounded least squares on template columns.

Flow of fit_day (one measurement segment, one channel):
    1. Faulty recordings -> NaN, background subtraction with the same rolling percentile
       as the extraction.
    2. Noise estimate σ̂ (MAD of the vehicle-free residual), centering on its median.
    3. Anchor runs: vehicle-free samples inside the noise band ±k₃σ̂, at least T_A long.
    4. Segmentation at the anchor runs into independent fit problems (segment_day).
    5. Per segment: baseline model, position refinement (refine_positions), merging of
       collinear template columns (collinearity_grouping, ShapeTemplate.composite),
       BVLS fit with standard errors (fit_mode_l), detection.

Refinement runs before merging on purpose: merging at the nominal positions would
irreversibly fuse passes whose true delays differ. The noise-calibrated acceptance
threshold κσ̂² keeps truly inseparable passes from being pulled apart by noise.

Thesis: chapter "The PlumeFit algorithm", section "Amplitude fitting", and the appendix
"Refinement of the template placement".

Citation:
Ludwig Fahrmeir, Thomas Kneib, Stefan Lang; Regression: Modelle, Methoden und Anwendungen.
2. Auflage. Berlin, Heidelberg: Springer; 2009. (Statistik und ihre Anwendungen)
https://doi.org/10.1007/978-3-642-01837-4

"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Assistance with Docstring writing, latex equation to Unicode symbol equations,
#  Variable Renaming Suggestions to reflect the used variables in the thesis)

import numpy as np
from scipy.optimize import lsq_linear

from src.shared_services.measurement_register import MeasurementRegister
from src.shared_services.noise_and_background import (compute_background_series, contiguous_runs,
                                                      estimate_noise, influence_mask, faulty_recording_mask)

from src.plume_template_extraction.extraction_config import ExtractionConfig
from src.mode_linear_fitting.fitting_config import ShapeTemplate, Segment, ModeLConfig
from src.mode_linear_fitting.fitting_result import PassAmplitude, ModeLFitResult, SegmentRecord, DayFitResult


MIN_AREA_FRACTION = 0.5
#Minimum share of the template area inside the segment for a template column.
#Below, the column is rejected; above, the truncated column is renormalized to unit
#area, -> the missing part is extrapolated with the template shape. -> Disable by setting it to 1.0

_N_BASELINE = {None: 0, "const": 1, "linear": 2}
# Number of baseline columns per baseline model -> linear needs a constant + linear collum (y = k*x + d)

def template_column(template: ShapeTemplate, position: int, n: int, dt: float) -> np.ndarray | None:
    """Design matrix column s_j(τ_j): the template with its peak at position, unit area.

    The column is renormalized to unit area inside the segment, so the fitted
    amplitude is the area of the whole plume even if the template is cut at the
    segment border. Extrapolation under the template-shape assumption / Only applies on segment bounds of the
    measurement data.

    Args:
        template: Template of the pass.
        position: Peak position relative to the segment start in samples.
        n: Segment length in samples.
        dt: Sampling interval in s.

    Returns:
        Unit-area column of length n, or None if at most MIN_AREA_FRACTION of the
        template area falls inside the segment.
    """
    s = template.sample(np.arange(n)-position)
    area = s.sum() * dt
    return None if area <= MIN_AREA_FRACTION else s/area

def baseline_columns(n: int, baseline: str | None)-> list[np.ndarray]:
    """Baseline columns B of the design matrix.

    The templates are background free, the segment signal still carries a local
    baseline offset. Without own columns it would be absorbed into the amplitudes.

    Args:
        n: Segment length [samples].
        baseline: None (no column), "const" (offset: ones) or "linear" (offset and a
            centered ramp (k − k̄) / n).

    Returns:
        List of 0, 1 or 2 columns of length n.
    """
    rel = np.arange(n)
    return [np.ones(n),(rel - rel.mean()) / n ][:_N_BASELINE[baseline]]


def fit_mode_l(segment: np.ndarray, templates: list[ShapeTemplate],
               peak_positions: np.ndarray, dt: float,
               baseline: str | None = "const",
               with_se: bool = True) -> ModeLFitResult:
    """Fit template amplitudes and baseline to one segment by BVLS.

    Design matrix A = [S | B]: one unit-area template column per pass (peak at its
    position) and the baseline columns. Amplitudes are bounded to a ≥ 0, baseline
    coefficients are free. Samples with NaN are left out.

    Args:
        segment: Background-subtracted, median-centered segment signal.
        templates: One template per column.
        peak_positions: Peak position per column relative to the segment start in samples.
        dt: Sampling interval in s.
        baseline: Baseline model, see baseline_columns. Defaults to "const".
        with_se: Compute the amplitude standard errors. Defaults to True.

    Returns:
        Amplitudes (= plume areas), baseline coefficients and diagnostics.

    Raises:
        ValueError: If a template column has too little area in the segment, or the
            segment has fewer valid samples than columns.

    Note:
        TODO: Farren et al. states that trough turbulences small negative values are valid, bvls bounds a >= 0 for stability
        TODO: Is there a way to respect the turbulences and still provide a stable fit
    """

    #Building the Design Matrix
    # Depending on the number of vehicles being considered to influence each other in a window the segment size varies.
    # Templates are the basis function for the fit, they are put in the design matrix aligning their peaks with the
    # suspected peak positions of the segment members. Numerical templates are only defined for a certain set of samples
    # outside of that set they are valued 0, meaning this car only influences the area covered by its template in the segment
    cols = [template_column(template,peak_position, segment.size,dt) for template, peak_position in zip(templates, peak_positions)]
    #AI-Assisted: <Opus 5> ; (Catch Potential Value Errors)
    if any(c is None for c in cols):
        raise ValueError("template column has too little area in window")
    S = np.column_stack(cols)

    # Templates are Baseline Free and the segment signal contains a baseline, to avoid overweighting the
    # fitted amplitudes a separate baseline collum is added. Fitting baselines works best with plume free spots in a
    # segment so the baseline can find a clear level to anchor to.
    n_base = _N_BASELINE[baseline]
    base_cols = baseline_columns(segment.size, baseline)
    # The complete design matrix with baseline colums
    A = np.column_stack([S] + base_cols) if base_cols else S

    # Guard against nan/inf values in the signal (these are possibly brought in trough QA)
    valid = np.isfinite(segment)
    if valid.sum() < A.shape[1]:
        raise ValueError("fewer valid samples than columns - segment not identifiable")
    # Calculate the condition number of the valid matrix
    condition_number = np.linalg.cond(A[valid])

    # Number of vehicles in segment
    J = S.shape[1]
    # Bounds for the least squares (amplitudes >= 0, baseline >= -inf)
    # amplitudes >= 0 censors turbulence-dilution effects at zero — deliberate, see docstring note

    bounds = (np.r_[np.zeros(J), np.full(n_base, -np.inf)], np.full(J + n_base, np.inf))
    res = lsq_linear(A[valid], segment[valid], bounds=bounds, method="bvls")
    A_v, y_v = A[valid], segment[valid]
    resid = A_v @ res.x - y_v
    sse = float(resid @ resid)
    dof = A_v.shape[0] - A_v.shape[1]
    sigma2_fit = sse / dof if dof > 0 else np.inf
    if with_se:
        try:
            # Catching because the inverse could fail due to singularities
            # Due to the fact that collinear columns are grouped this should usually not occur
            cov = sigma2_fit * np.linalg.inv(A_v.T @ A_v)
            amplitude_se = np.sqrt(np.diag(cov)[:J])
        except np.linalg.LinAlgError:
            amplitude_se = np.full(J, np.inf)
    else:
        amplitude_se = np.full(J, np.nan)

    # Residual RMS over the valid segment samples (reuses the residual from the SSE)
    residual_rms = float(np.sqrt(np.mean(resid ** 2)))

    return ModeLFitResult(res.x[:J], res.x[J:], residual_rms, condition_number,
                          amplitude_se, sse, dof, A @ res.x)


def collinearity_grouping(S: np.ndarray, vif_max: float) -> list[np.ndarray]:
    """Group neighboring template columns until every column's VIF is at most vif_max.

    Repeatedly merges the column with the largest VIF with its more similar direct
    neighbor (cosine similarity). A merged group is represented by the mean of its
    columns, matching ShapeTemplate.composite.

    Args:
        S: Template block of the design matrix, columns sorted by peak position,
            only valid rows.
        vif_max: VIF_max.

    Returns:
        One index array per group (indices into the columns of S), in column order.

    Citation:
       Fahrmeir et al., Regression (2009), pp. 101, 171.
    """
    k = S.shape[1]
    if k == 0:
        return []
    vif = None
    # Calculating Gram Matrix
    gramm_init = S.T @ S

    groups = [np.array([j]) for j in range(k)]

    #AI-Assisted: <Opus 5> ; (Usage of W Wight Matrix to reduce the need to Calculate full Gram Matrix more than once)
    # W maps original columns to groups: column g of W holds the weight of every original column in group g
    # (identity at the start, 1/size for the members after merging). The group columns are S W, their Gram
    # matrix (S W)ᵀ(S W) = Wᵀ G₀ W, so G₀ = SᵀS is computed only once
    W = np.eye(k)

    # Merging continues till there is only one group left or the VIF threshold is met
    while W.shape[1] > 1:
        gramm = W.T @ gramm_init @ W
        d = np.diag(gramm)

        # Due to the fact that np.linalg.inv can throw a LinAlgError for singularity the calculation is wrapt in a
        # try catch, the catch path only happens on exact collinearity, then the most similar neighbors are merged
        # before the calculation continues.
        try:
            vif = d * np.diag(np.linalg.inv(gramm))
            # np.linalg.inv can produce nonsense for near singular gramm, this is guarded by stating that a valid
            # vif is always >= 1 (R^2 in VIF = 1 / 1- R^2 lies between 0 and 1), the value 0,999 is chosen due to float
            # rounding behavior
            vif_valid = bool(np.all(vif >= 0.999))
        except np.linalg.LinAlgError:
            vif_valid = False

        # The vif already satisfies the threshold
        if vif_valid and vif.max() <= vif_max:
            break

        # The similarity between neighbors is calculated over the cosine (correlation)
        # the algorithm only merges direct neighbors.
        cos_nbr = np.diag(gramm,1)/np.sqrt(d[:-1] * d[1:])
        if vif_valid:
            # The collumn with the largest vif is merged with its more similar neighbor
            j = int(np.argmax(vif))
            # Check for segment edges (only one neighbor)
            if j == 0:
                lo = 0
            elif j == W.shape[1] - 1:
                lo = j - 1
            else:
                lo = j - 1 if cos_nbr[j - 1] >= cos_nbr[j] else j
        else:
            # Exact collinearity -> vif could not be calculated the most similar neighbor pair is merged
            lo = int(np.argmax(cos_nbr))

        # Merge groups lo and lo + 1
        merged = np.concatenate([groups[lo], groups[lo + 1]])
        groups[lo] = merged
        del groups[lo + 1]

        #Recalculate the weight matrix
        w = np.zeros(k)
        w[merged] = 1.0 / merged.size
        W[:, lo] = w
        W = np.delete(W, lo + 1, axis=1)

    return groups


def segment_day(n: int, pass_sample_idx: np.ndarray,
                starts: np.ndarray, stops: np.ndarray) -> list[Segment]:
    """Split a measurement segment into fit segments bounded by anchor runs.

    Every gap between two anchor runs that contains at least one expected peak becomes
    a segment, from the start of the left run to the end of the right run. Gaps
    without a pass produce no segment.

    Args:
        n: Signal length in samples.
        pass_sample_idx: Expected peak position per pass in samples.
        starts: Start indices of the anchor runs, see contiguous_runs.
        stops: Stop indices of the anchor runs (exclusive), parallel to starts.

    Returns:
        Segments in time order.
    """

    # Anchors are rolled out in day length boolean array for easy cutting
    anchor = np.zeros(n, dtype=bool)
    for a, b in zip(starts, stops):
        anchor[a:b] = True

    # stops equal the ends of the anchor runs timely sorted
    # in wich gap does the pass_sample_idx fit (by time of arrival)
    # if a peak arrives before the forst stop it adds 0 to gap it is part of segment 0,
    # right -> sample_idx hits stop exactly it is part of the next segment
    gap = np.searchsorted(stops, pass_sample_idx, side="right")
    segments = []
    # unique makes sure every gap leads to one segment (gaps can appear multiple times if they fit more than one peak)
    for g in np.unique(gap):
        # ids of all peaks that fit into this gap
        ids = np.flatnonzero(gap == g)
        # is the segment separated by left right anchor
        left, right = g >= 1, g < starts.size
        start = int(starts[g - 1]) if left else 0
        stop = int(stops[g]) if right else n
        # cuts the segment and adds it to the list
        segments.append(Segment(start, stop, ids, anchor[start:stop], left, right))
    return segments


def refine_positions(signal: np.ndarray, templates: list[ShapeTemplate],positions: np.ndarray,
                     lo: np.ndarray, hi: np.ndarray, dt: float, baseline: str | None, noise_var: float, sweeps: int = 10,
                     min_dsse: float = 8.0) -> np.ndarray:
    """Refine the template positions of one segment by minimizing the SSE.

    Coordinate descent: one pass at a time, in order of decreasing initial amplitude,
    is moved to the candidate position τ' with the lowest BVLS SSE. Candidates are
    pre-sorted by a cheap lower bound of their SSE (one least-squares solve for all
    candidates); BVLS only runs while the bound can still beat the best SSE. A move is
    accepted only if it lowers the SSE by more than min_dsse · noise_var. Sweeps repeat
    until nothing moves. The arrival order of the passes is preserved.

    Args:
        signal: Segment signal, as for fit_mode_l.
        templates: Template per pass.
        positions: Initial peak positions τ⁽⁰⁾ (trigger + median delay) relative to
            the segment start in samples.
        lo: Lowest allowed position per pass, relative to the segment start in samples.
        hi: Highest allowed position per pass, relative to the segment start in samples.
        dt: Sampling interval in s.
        baseline: Baseline model, see baseline_columns.
        noise_var: Noise variance σ̂² of the vehicle-free residual.
        sweeps: Maximum number of sweeps. Defaults to 10.
        min_dsse: κ of the acceptance threshold κ · σ̂². Defaults to 8.0.

    Returns:
        Refined positions τ̂ [samples].

    Note:
        Derivation of the lower bound: thesis appendix "Refinement of the template
        placement". Slow on pollutant channels !
    """

    # Parameter preparation
    tau = np.array(positions, dtype=int)
    n, valid = signal.size, np.isfinite(signal)
    valid_signal = signal[valid]
    threshold = min_dsse * noise_var

    # AI-Assisted: <Opus 5> ; (Review -> Extract SSE Calculation to catch Value Errors )
    def sse_bvls(p: np.ndarray) -> float:
        try:
            return fit_mode_l(signal, templates, p, dt, baseline=baseline, with_se=False).sse
        except ValueError:  # a template column with too little area in the segment
            return np.inf

    # Initial fit: SSE reference and processing order (largest amplitude first, so weak
    # plumes cannot settle inside a strong plume before the strong one is placed)
    initial = fit_mode_l(signal, templates, tau, dt, baseline=baseline, with_se=False)
    amp_order = np.argsort(-initial.amplitudes)
    curr_sse = initial.sse
    # Design matrix at the current positions (valid rows), column j = pass j, baseline last.
    # A move only replaces the column of the moved pass.
    A = np.column_stack([template_column(t, p, n, dt) for t, p in zip(templates, tau)]
                        + baseline_columns(n, baseline))[valid]
    # The Position Refinement keeps the order of occurrence dictated from the Licht Barrier Passes intact
    # meaning vehicle A passes LB before vehicle B after refine_positions vehicle B cannot be in front of vehicle A
    rank = np.argsort(positions, kind="stable")
    prev_pass = np.full(tau.size, -1)
    next_pass = np.full(tau.size, -1)
    prev_pass[rank[1:]] = rank[:-1]
    next_pass[rank[:-1]] = rank[1:]
    # The refine positions loop
    for _ in range(sweeps):
        moved = False
        # Highest Amplitudes get shifted first trying to prevent low emitters hiding in high emitter plumes instead of
        # their own

        for j in amp_order:
            # Search band C_j = [τ_min, τ_max], clipped to the segment and to the neighbors
            # The shift is guraed by p_lo and p_hi, bounds to prevent a template to travel away from its plume
            # It is also prevented that candidates sitting near segment edge travel out of it (0, n-1) bounds.
            # For the current j pass all possible positions are collected and the template centeres are shifted
            p_lo = max(int(lo[j]), 0)
            p_hi = min(int(hi[j]), n - 1)
            if prev_pass[j] >= 0:
                p_lo = max(p_lo, int(tau[prev_pass[j]]))
            if next_pass[j] >= 0:
                p_hi = min(p_hi, int(tau[next_pass[j]]))
            candidates, cols = [], []
            for p in range(p_lo, p_hi + 1):
                col = template_column(templates[j], p, n, dt) if p != tau[j] else None
                if col is not None:
                    candidates.append(p)
                    cols.append(col[valid])
            if not candidates:
                continue

            # Lower bound of the SSE per candidate: project signal and candidate columns onto the
            # orthogonal complement of the other columns (one lstsq for all), then the best
            # non-negative amplitude of each candidate on the remaining residual
            other_passes = np.delete(A, j, axis=1)
            targets = np.column_stack([valid_signal] + cols)
            residual = targets - other_passes @ np.linalg.lstsq(other_passes, targets, rcond=None)[0]
            y_perp, s_perp = residual[:, 0], residual[:, 1:]
            sse_without = float(y_perp @ y_perp)
            num = s_perp.T @ y_perp
            denom = (s_perp ** 2).sum(axis=0)
            # A negative optimal amplitude is clipped to 0 by BVLS -> no gain
            gain = np.divide(num ** 2, denom, out=np.zeros_like(num), where=num > 0)
            lower = sse_without - gain
            best_sse, best_position = curr_sse - threshold, tau[j]

            # Verify candidates by BVLS in order of their bound; stop once the bound
            # cannot beat the best SSE found so far
            for c in np.argsort(lower):
                if lower[c] >= best_sse:
                    break
                trial = tau.copy()
                trial[j] = candidates[c]
                sse = sse_bvls(trial)
                if sse < best_sse:
                    best_sse, best_position = sse, candidates[c]
            if best_position != tau[j]:
                tau[j], curr_sse, moved = best_position, best_sse, True
                A[:, j] = template_column(templates[j], best_position, n, dt)[valid]
        if not moved:
            break
    return tau


def fit_day(register: MeasurementRegister, channel: str, template: ShapeTemplate | list[ShapeTemplate],
            peak_positions: np.ndarray, cfg: ModeLConfig = ModeLConfig(),
            pos_lo: np.ndarray | None = None, pos_hi: np.ndarray | None = None,
            keep_signals: bool = False) -> DayFitResult:
    """Fit the plume area of every pass of one measurement segment and channel.

    See the module docstring for the steps. A merged group is fitted as one composite
    column; its amplitude is the total area of the group.

    Args:
        register: Measurement data of the segment.
        channel: Channel to fit.
        template: One template for all passes, or one template per pass (e.g. chosen
            by vehicle class beforehand).
        peak_positions: Expected peak position τ⁽⁰⁾ per pass [samples], e.g. trigger
            index + median trigger delay. Should be sorted ascending.
        cfg: Fit parameters. Defaults to ModeLConfig().
        pos_lo: Lowest allowed peak position per pass [samples]. Setting pos_lo and
            pos_hi enables refine_positions. Defaults to None.
        pos_hi: Highest allowed peak position per pass [samples]. Defaults to None.
        keep_signals: Keep the residual signal and the per-segment models for plots.
            Defaults to False.

    Returns:
        One SegmentRecord per segment.

    Raises:
        ValueError: If the number of templates does not match the passes, or no
            vehicle-free sample exists for the noise estimate.
    """

    peak_positions = np.asarray(peak_positions, dtype=int)
    # One template per pass: a single template is expanded, a list is taken as is
    if isinstance(template, ShapeTemplate):
        templates = [template] * peak_positions.size
    else:
        templates = list(template)
        if len(templates) != peak_positions.size:
            raise ValueError(f"{len(templates)} templates for {peak_positions.size} passes")
    data = register.get_channel_data_by_name(channel)
    # Faulty recordings (dropouts, instrument off) become NaN, fit_mode_l skips those rows
    dt = register.dt
    phys_run = ExtractionConfig.as_samples(cfg.min_physical_run,dt)
    data = np.where(faulty_recording_mask(data, cfg.min_physical_value, phys_run), np.nan, data)

    # Calculate the background subtracted signal
    # The baseline is subtracted using the same method as in plume_extraction.py
    # the remaining background correction happens in the fitting itself
    residual = data - compute_background_series(
        data, cfg.bg_percentile, ExtractionConfig.as_samples(cfg.bg_rolling_window, dt),
        cfg.min_physical_value, phys_run)

    # Unit conversion
    n_influence_before = ExtractionConfig.as_samples(cfg.influence_before, dt)
    n_influence_after = ExtractionConfig.as_samples(cfg.influence_after, dt)
    n_min_anchor_run = ExtractionConfig.as_samples(cfg.min_anchor_run, dt)
    t_min_linear_s = cfg.t_min_linear / np.timedelta64(1, "s")

    # Influence windows around the expected peaks approximate T₀ noise estimate on the rest.
    # Centering on median(r_F) removes the low bias of the percentile background.
    influenced = influence_mask(residual.size, peak_positions, n_influence_before, n_influence_after)
    free_median, free_sigma = estimate_noise(residual, influenced)
    residual = residual - free_median
    noise_band = cfg.noise_band_sigma * free_sigma

    # Anchor points: vehicle free, finite and inside the noise band (excludes quiet-looking (vehicle free) samples
    # disturbed by unregistered sources). The runs are shared by segmentation and fixed baseline.
    quiet = ~influenced & np.isfinite(residual) & (np.abs(residual) < noise_band)
    run_starts, run_stops = contiguous_runs(quiet, n_min_anchor_run)

    # Split the day into segments
    segments = segment_day(residual.size, peak_positions, run_starts, run_stops)

    # Fixed baseline: polyline through the median level of every anchor run of the whole day
    centers = (run_starts + run_stops) / 2
    levels = np.array([np.median(residual[a:b]) for a, b in zip(run_starts, run_stops)])


    records = []
    for seg in segments:
        # Signal in the segment
        y = residual[seg.start:seg.stop]
        # Rel. Peak positions in the segment
        tau0 = peak_positions[seg.pass_ids] - seg.start

        # Count of available anchors in the segment
        n_anchor = int(seg.anchor_mask.sum())
        # Length of the segment in seconds
        span_s = (seg.stop - seg.start) * dt
        # This logic is used to restrict the degrees of freedom of the background fit
        # anchored_fixed: No free parameters, the baseline is not fitted but best effort subtracted befor the fit
        # also applies when there are to little anchor pints.
        # fitted_linear: There are available anchor points on both sides of the segment (measurement array bounds)
        # the segment is long enough to rectify a linear baseline and has enough anchors.
        # -> linear so there are two degrees of freedom for the fit (slope and intercept)
        # fitted_const: one degree of freedom for the fit (intercept)
        if cfg.force_anchored_fixed or n_anchor < cfg.n_anchor_min:
            mode = "anchored_fixed"
        elif (seg.anchored_left and seg.anchored_right
              and span_s > t_min_linear_s and n_anchor >= 2 * cfg.n_anchor_min):
            mode = "fitted_linear"
        else:
            mode = "fitted_const"

        base = None
        if mode == "anchored_fixed":
            # Calculates the polyline through the anchor points, subtracts it from the signal
            # baseline_arg is set to None meaning the fitter has no baseline collum to work with
            base = (np.interp(np.arange(seg.start, seg.stop), centers, levels)
                    if centers.size else np.zeros(seg.stop - seg.start))
            y = y - base
            baseline_arg = None
        else:
            baseline_arg = "const" if mode == "fitted_const" else "linear"


        # Templates of the passes in this segment (aligned with nominal / local)
        seg_templates = [templates[i] for i in seg.pass_ids]

        try:
            tau_hat = tau0
            if pos_lo is not None:
                tau_hat = refine_positions(signal=y, templates=seg_templates, positions=tau0,
                                           lo=pos_lo[seg.pass_ids] - seg.start,
                                           hi=pos_hi[seg.pass_ids] - seg.start,
                                           dt=dt, baseline=baseline_arg,
                                           noise_var=free_sigma ** 2, sweeps=cfg.refine_sweeps,
                                           min_dsse=cfg.refine_min_dsse)
            # Merge peaks that are considered non-separable
            order = np.argsort(tau_hat, kind="stable")
            cols = [template_column(seg_templates[i], tau_hat[i], y.size, dt) for i in order]
            if any(c is None for c in cols):
                raise ValueError("template column has too little area in window")
            S = np.column_stack(cols)[np.isfinite(y)]
            groups = [order[g] for g in collinearity_grouping(S, cfg.merge_vif_max)]
            # One column per group at the (rounded) mean member position; merged groups get a
            # composite template of the members at their own positions
            col_pos = np.array([int(round(tau_hat[g].mean())) for g in groups])
            col_templates = [seg_templates[g[0]] if g.size == 1
                             else ShapeTemplate.composite([seg_templates[i] for i in g], tau_hat[g] - cp)
                             for g, cp in zip(groups, col_pos)]
            # Performing the fit, every group gets one column in the design matrix
            fit = fit_mode_l(y, col_templates, col_pos, dt, baseline=baseline_arg)

        except ValueError:
            records.append(SegmentRecord(seg, "failed", (), n_anchor, np.nan, np.nan, []))
            continue

        # Back from columns to passes; members of a group share amplitude, SE and detection
        shifts = tau_hat - tau0
        passes = []
        for gi, g in enumerate(groups):
            group_ids = tuple(int(seg.pass_ids[i]) for i in g)
            tail = (not seg.anchored_right) and gi == len(groups) - 1
            # Standard error and detection flag are per column -> shared by all members of a merged group
            se = float(fit.amplitudes_se[gi])
            detected = bool(fit.amplitudes[gi] >= cfg.detect_sigma * se)
            for i in g:
                passes.append(PassAmplitude(int(seg.pass_ids[i]), float(fit.amplitudes[gi]),
                                            group_ids, tail, int(shifts[i]), se, detected))

        records.append(SegmentRecord(seg, mode, tuple(fit.baseline_coeffs), n_anchor,
                                     fit.condition_number, fit.residual_rms, passes,
                                     model=(fit.model + (base if base is not None else 0.0)) if keep_signals else None))

    return DayFitResult(channel, records, residual if keep_signals else None)