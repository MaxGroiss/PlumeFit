"""Mode L fitting: plume-template amplitudes per vehicle pass via bounded least squares.

Day pipeline (fit_day)
    1. Background subtraction (rolling percentile, same method as the template extraction)
    2. Anchor detection: vehicle-free samples inside a noise band (sigma via MAD of the
       vehicle-free residual)
    3. Segmentation at anchor runs -> independent fit problems (segment_day)
    4. Per segment, in this order:
       a. position refinement inside the causal delay band (refine_positions,
          coordinate-descent SSE grid search, sweeps repeat until nothing moves)
       b. merging of non-separable passes into one composite column (_merge_groups,
          ShapeTemplate.composite)
       c. BVLS fit of template columns + baseline column(s) (fit_mode_l)

The order refine -> merge -> fit is deliberate: merging on nominal positions would
irreversibly fuse pairs whose true delays differ, refinement first lets such pairs
separate; the noise-calibrated acceptance threshold (refine_min_dsse) keeps genuinely
inseparable blobs from being split apart.
"""
# This file contains code created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)

import numpy as np
from scipy.optimize import lsq_linear

from src.shared_services.measurement_register import MeasurementRegister
from src.shared_services.noise_and_background import (compute_background_series, contiguous_runs,
                                                      estimate_noise, influence_mask, faulty_recording_mask)

from src.plume_template_extraction.extraction_config import ExtractionConfig
from src.mode_linear_fitting.fitting_config import ShapeTemplate, Segment, ModeLConfig
from src.mode_linear_fitting.fitting_result import PassAmplitude, ModeLFitResult, SegmentRecord, DayFitResult


MIN_AREA_FRACTION = 0.5 # More than -,- of the Plume template have to fit in the segment for extrapolation
_N_BASELINE = {None: 0, "const": 1, "linear": 2}


def template_column(template: ShapeTemplate, position: int, n: int, dt: float) -> np.ndarray | None:
    """Represent one design matrix collum with the peak of the template at position, sampled over n rows.

    The Template Area is re normalized to 1, this catches faulty normalized templates and opens up
    a form of extrapolation. If a template sits at the border of the section a part of it is cut of renormalizing it
    still calculates the Area of the whole plume under the assumption that the plume follows the template shape.
    This only appears at dataset boundaries, all segments inside a measurement day are guarded with anker points.


    :param template: Template of the pass
    :param position: Peak position relative to the segment start
    :param n: Segment Length
    :param dt: Sampling interval in seconds
    :return: Unit area collum, None if less than MIN_AREA_FRACTION of the template falls in the segment.
    """
    s = template.sample(np.arange(n)-position)
    area = s.sum() * dt
    return None if area <= MIN_AREA_FRACTION else s/area

def baseline_columns(n: int, baseline: str | None)-> list[np.ndarray]:
    """Baseline columns of the design matrix

    A segment-signal is overlaid by a baseline, the templates are baseline free
    to not systematically overweight the fitted amplitudes a separate baseline columns are added
    as for now there are two baseline assumption that can be chosen by the baseline parameter
    "const" for a constant baseline
    "linear" for a linear baseline in the segment

    :param n: Segment Length
    :param baseline: None, "const" or "linear
    :return: List of baseline columns
    """
    rel = np.arange(n)
    return [np.ones(n),(rel - rel.mean()) / n ][:_N_BASELINE[baseline]]




def fit_mode_l(segment: np.ndarray, templates: list[ShapeTemplate],
               peak_positions: np.ndarray, dt: float,
               baseline: str | None = "const",
               with_se: bool = True) -> ModeLFitResult:
    """Fits plume templates plus baseline to a segment via bounded least squares (BVLS).

    Builds a design matrix with one re-normalized template column per pass (template peak
    aligned with peak_positions) plus optional baseline columns, then solves with the
    amplitudes bound to >= 0.
    TODO: Farren et al. states that trough turbulences small negative values are valid

    :param segment: Signal of the segment (background subtracted)
    :param templates: One ShapeTemplate per pass in the segment
    :param peak_positions: Peak position per pass, relative to the segment start
    :param dt: Sampling interval in seconds
    :param baseline: Baseline model: None (no baseline column), "const" or "linear"
    :param with_se: If True the amplitude standard errors are calculated
    :return: ModeLFitResult with amplitudes, baseline coefficients and diagnostics
    :raises ValueError: If a template column has too little area in the window or the
                        segment has fewer valid samples than columns
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
    k = S.shape[1]
    # Bounds for the least squares (amplitudes >= 0, baseline >= -inf)
    # amplitudes >= 0 censors turbulence-dilution effects at zero — deliberate, see docstring note

    bounds = (np.r_[np.zeros(k), np.full(n_base, -np.inf)], np.full(k + n_base, np.inf))
    # Performs the fit within the bounds
    res = lsq_linear(A[valid], segment[valid], bounds=bounds, method="bvls")
    A_v, y_v = A[valid], segment[valid]
    resid = A_v @ res.x - y_v
    sse = float(resid @ resid)
    dof = A_v.shape[0] - A_v.shape[1]
    sigma2 = sse/ dof if dof > 0 else np.inf
    if with_se:
        try:
            cov = sigma2 * np.linalg.inv(A_v.T @ A_v)
            amplitude_se = np.sqrt(np.diag(cov)[:k])
        except np.linalg.LinAlgError:
            amplitude_se = np.full(k, np.inf)
    else:
        amplitude_se = np.full(k, np.nan)


    # Residual RMS over the valid segment samples (reuses the residual from the SSE)
    residual_rms = float(np.sqrt(np.mean(resid ** 2)))

    return ModeLFitResult(res.x[:k], res.x[k:], residual_rms, condition_number,
                          amplitude_se, sse, dof, A @ res.x)




def _separability_shift(template: ShapeTemplate, threshold: float) -> int:
    """Minimum shift between templates in the design matrix to be considered separable.

    Was done because the threshold means the same for sharp edge templates
    and slow rise templates.

    :param template: Shape template of the channel
    :param threshold: Autocorrelation threshold below which the overlap counts as separable
    :return: Minimum shift in samples
    """
    v = template.values
    denom = float(v @ v)
    for s in range(1, v.size):
        # Calculate the auto-correlation of the template
        if float(v[s:] @ v[:-s]) / denom < threshold:
            return s
    return v.size


def _merge_groups(local_pos: np.ndarray, min_sep: int) -> list[np.ndarray]:
    """Groups vehicle passes that don't pass min_sep (templates are too close).

    If vehicle A and B are too close and B and C are too close -> A, B, C are in one group.

    :param local_pos: Peak positions relative to the segment start
    :param min_sep: Minimum separable shift in samples
    :return: List of index arrays, one array per group
    """
    # AI Assisted-by: Claude Fable (5) (comments and code written by hand)
    order = np.argsort(local_pos)
    groups = [[order[0]]]
    for i in order[1:]:
        if local_pos[i] - local_pos[groups[-1][-1]] < min_sep:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [np.asarray(g) for g in groups]


def segment_day(n: int, pass_sample_idx: np.ndarray,
                starts: np.ndarray, stops: np.ndarray) -> list[Segment]:
    """Splits a measurement day into segments bounded by anchor runs.

    :param n: Length of the day signal in samples
    :param pass_sample_idx: peak positions of the passes in samples
    :param starts: Start indices of the anchor runs (see :func:`contiguous_runs` on the quiet mask)
    :param stops: Stop indices of the anchor runs (parallel to starts)
    :return: segments of the day
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
                     lo: np.ndarray, hi: np.ndarray, dt: float, baseline: str | None, sigma2: float, sweeps: int = 10,
                     min_dsse: float = 8.0) -> np.ndarray:
    """Refines the roughly offset based Template positioning




    :param signal:
    :param templates:
    :param positions:
    :param lo:
    :param hi:
    :param dt:
    :param baseline:
    :param sigma2:
    :param sweeps:
    :param min_dsse:
    :return:
    """

    # Parameter preparation
    pos = np.array(positions,dtype = int)
    n, valid = signal.size, np.isfinite(signal)
    valid_signal = signal[valid]
    threshold = min_dsse * sigma2

    # AI-Assisted: <Opus 5> ; (Review -> Extract SSE Calculation to catch Value Errors )
    def sse_bvls(p: np.ndarray) -> float:
        try:
            return fit_mode_l(signal, templates, p, dt, baseline=baseline, with_se=False).sse
        except ValueError:  # a template column with too little area in the segment
            return np.inf

    # Performs the inital fit to get amplitude descending order for grid walk
    initial = fit_mode_l(signal, templates, pos, dt, baseline=baseline, with_se=False)
    amp_order = np.argsort(-initial.amplitudes)
    # The SSE to compare shifted fits to each other
    curr_sse = initial.sse
    # Now the Design Matrix is build for the current position with column i = template of pass i the
    # baseline column is at the end. A moved pass replaces its collumn.
    A = np.column_stack([template_column(t, p, n, dt) for t, p in zip(templates, pos)]
                        + baseline_columns(n, baseline))[valid]
    # The refine positions loop
    for _ in range(sweeps):
        moved = False
        # Highest Amplitudes get shifted first trying to prevent low emitters hiding in high emitter plumes instead of
        # their own
        for i in amp_order:
            candidates, cols = [],[]
            # The shift is guraed by lo and hi, bounds to prevent a template to travel away from its plume
            # It is also prevented that candidates sitting near segment edge travel out of it (0, n-1) bounds.
            # For the current i pass all possible positions are collected and the template centeres are shifted
            for p in range(max(int(lo[i]), 0), min(int(hi[i]), n - 1) + 1):
                col = template_column(templates[i], p, n, dt) if p != pos[i] else None
                if col is not None:
                    candidates.append(p)
                    cols.append(col[valid])
            # The Plume cannot be shiftet further
            if not candidates:
                continue

            # The Idea is to let the other plume templates try to fit the signal without moving as best as they can,
            # this can be done quickly using a regression on the other templates and subtracting the result from the
            # target.
            other_passes = np.delete(A, i, axis=1)
            targets = np.column_stack([valid_signal] + cols)
            residual = targets - other_passes @ np.linalg.lstsq(other_passes, targets, rcond=None)[0]
            # The Resiudal can now be
            y_perp, s_perp = residual[:, 0], residual[:, 1:]
            sse_without = float(y_perp @ y_perp)
            num = s_perp.T @ y_perp
            den = (s_perp ** 2).sum(axis=1)
            gain = np.divide(num ** 2, den, out = np.zeros_like(num), where = num > 0)
            lower = sse_without - gain

            best_sse, best_position = curr_sse - threshold, pos[i]
            for j in np.argsort(lower):
                if lower[j] >= best_sse:
                    break
                trial = pos.copy()
                trial[i] = candidates[j]
                sse = sse_bvls(trial)
                if sse < best_sse:
                    best_sse, best_position = sse, candidates[j]
            if best_position != pos[i]:
                pos[i], curr_sse, moved = best_position, best_sse, True
                A[:,i]= template_column(templates[i], best_position,n,dt)[valid]
        if not moved:
            break
    return pos


def fit_day(register: MeasurementRegister, channel: str, template: ShapeTemplate | list[ShapeTemplate],
            peak_positions: np.ndarray, cfg: ModeLConfig = ModeLConfig(),
            pos_lo: np.ndarray | None = None, pos_hi: np.ndarray | None = None,
            keep_signals: bool = False) -> DayFitResult:
    """Performs the Mode L fitting for one day and one channel.

    Background subtraction -> anchor detection -> segmentation -> per segment:
    optional position refinement, merging of non-separable passes, fitting.
    A non-separable group is fitted as one Farren-like composite column (mean of the
    member templates at their own positions), its amplitude is the total group area.

    :param register: MeasurementRegister of the measurement day
    :param channel: Name of the channel to fit
    :param template: One shape template used for all passes, or a list with one template
                     per pass (resolved beforehand, e.g. from a vehicle-id mapping)
    :param peak_positions: Expected global peak position per pass in samples
    :param cfg: Mode L configuration
    :param pos_lo: Lowest allowed sample index per pass, enables :func:`refine_positions`
    :param pos_hi: Highest allowed sample index per pass
    :param keep_signals: If True the day residual and the per-segment models are kept (for plotting)
    :return: DayFitResult with one SegmentRecord per segment
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
    # Non-physical values (dropouts, instrument off) become NaN so fit_mode_l drops those rows
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
    inf_before = ExtractionConfig.as_samples(cfg.influence_before, dt)
    inf_after = ExtractionConfig.as_samples(cfg.influence_after, dt)
    min_run = ExtractionConfig.as_samples(cfg.min_anchor_run, dt)
    t_min_linear_s = cfg.t_min_linear / np.timedelta64(1, "s")

    # For later influence free area detection a noise band is needed to detect areas that appear vehicle free
    # but are influenced by other sources there for resulting in unclean anchoring points
    influenced = influence_mask(residual.size, peak_positions, inf_before, inf_after)
    # The noise sigma is estimated via the MAD of the vehicle-free residual (see estimate_noise),
    # the median centers the residual on the quiet signal level
    free_median, free_sigma = estimate_noise(residual, influenced)
    residual = residual - free_median
    noise_band = cfg.noise_band_sigma * free_sigma

    # Quiet = vehicle free, a meaningful measurement and inside the noise band; the anchor runs
    # derived from it are computed ONCE and shared by segmentation and baseline anchoring
    quiet = ~influenced & np.isfinite(residual) & (np.abs(residual) < noise_band)
    run_starts, run_stops = contiguous_runs(quiet, min_run)

    # Split the day into segments
    segments = segment_day(residual.size, peak_positions, run_starts, run_stops)

    # Anchor run centers and levels for the anchored_fixed baseline
    centers = (run_starts + run_stops) / 2
    # Median of the residual in the quiet area, one area median per center
    levels = np.array([np.median(residual[a:b]) for a, b in zip(run_starts, run_stops)])
    # Minimum necessary shift between templates in design matrix to be considered separable
    # With per-pass templates the conservative maximum over the distinct templates is used
    distinct_templates = {id(t): t for t in templates}.values()
    min_sep = max((_separability_shift(t, cfg.merge_corr_threshold) for t in distinct_templates), default=1)
    records = []
    for seg in segments:
        # Signal in the segment
        y = residual[seg.start:seg.stop]
        # Rel. Peak positions in the segment
        nominal = peak_positions[seg.pass_ids] - seg.start

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
            local = nominal
            if pos_lo is not None:
                local = refine_positions(signal=y, templates= seg_templates,positions= nominal,
                                         lo = pos_lo[seg.pass_ids] - seg.start,
                                         hi = pos_hi[seg.pass_ids] - seg.start,
                                         dt = dt, baseline = baseline_arg,
                                         sigma2 = free_sigma ** 2, sweeps = cfg.refine_sweeps,
                                         min_dsse = cfg.refine_min_dsse)
            # Merge peaks that are considered non-separable
            groups = _merge_groups(local, min_sep)
            # A group only gets one collum placed at the mean of the members
            col_pos = np.array([int(round(local[g].mean())) for g in groups])
            # Merged groups get a Farren-like composite column: the member templates are
            # superimposed at their own positions, single-pass groups keep their template
            col_templates = [seg_templates[g[0]] if g.size == 1
                             else ShapeTemplate.composite([seg_templates[i] for i in g], local[g] - cp)
                             for g, cp in zip(groups, col_pos)]
            # Performing the fit, every group gets one column in the design matrix
            fit = fit_mode_l(y, col_templates, col_pos, dt, baseline=baseline_arg)

        except ValueError:
            records.append(SegmentRecord(seg, "failed", (), n_anchor, np.nan, np.nan, []))
            continue

        # Groups get split back up in individual passes (keeping the information that they were part of the group)
        shifts = local - nominal
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