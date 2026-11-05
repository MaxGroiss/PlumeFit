"""Mode L fitting of one channel and day: plume area per vehicle pass by BVLS on template columns.

Flow of fit_channel_day (one channel of a PairSetup, one measurement segment = one MeasurementRegister):
    1. prepare_signal: faulty recordings -> NaN, background subtraction with the same rolling percentile
       as the extraction.
    2. noise_level: noise estimate σ̂ (MAD of the vehicle-free residual), centering on its median.
    3. anchor_runs: vehicle-free samples inside the noise band ±k₃σ̂, at least T_A long.
    4. segment_day: segmentation at the anchor runs into independent fit problems.
    5. fit_segment, per segment: choose_baseline → identifiable passes → position refinement (strategy of
       the ChannelSetup, see position_refinement) → merging of collinear template columns
       (collinearity_grouping, ShapeTemplate.composite) → BVLS fit with standard errors (fit_mode_l) →
       detection → one PassFit per pass → QA checks and non-detect policy (fit_checks).
Every pass handed to the fit appears in the result with a status (see fitting_result.STATUS_*).

Refinement runs before merging on purpose: merging at the nominal positions would
irreversibly fuse passes whose true delays differ. The noise-calibrated acceptance
threshold κσ̂² keeps truly inseparable passes from being pulled apart by noise.

The coupling of CO₂ and pollutant channels (positions of the pollutant from the refined CO₂ peaks) is in
pair_fitting.

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
# AI-Assisted: <Opus 5.5> ; (Assistance with optimizing the code structure and reducing unneeded runtime intensive calls)


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from src.shared_services.measurement_register import MeasurementRegister, as_samples
from src.shared_services.noise_and_background import (compute_background_series, contiguous_runs,
                                                      estimate_noise, influence_mask, faulty_recording_mask)

from src.mode_linear_fitting.fitting_config import ModeLConfig, ChannelSetup, PairSetup
from src.mode_linear_fitting.fitting_inputs import ShapeTemplate
from src.mode_linear_fitting.fit_checks import SegmentContext, run_checks
from src.mode_linear_fitting.fitting_result import (ModeLFitResult, PassFit, SegmentDiagnostics, ChannelDayResult,
                                                    STATUS_FITTED, STATUS_NOT_IDENTIFIABLE, STATUS_SEGMENT_FAILED,
                                                    STATUS_OUTSIDE_SIGNAL)


@dataclass(frozen=True)
class ChannelDay:
    """Day-level inputs of one channel, everything that is not pass_specific

    Mainly done to reduce input parameters for fitting functions

    Attributes:
        pair: The pair: fit parameters (pair.fit), QA checks (pair.qa_checks), non-detect policy (pair.non_detect).
        setup: The fitted channel of the pair: refinement strategy (setup.refinement).
        residual: Centered residual of the whole day.
        fixed_level: (centers, levels) of all anchor runs of the day, the fixed baseline polyline.
        sigma: Noise level σ̂.
        dt: Sampling interval in s.
        reference: CO₂ PassFit per pass_index (pollutant fits), handed to the checks; empty for CO₂.
        keep_diagnostics: Return SegmentDiagnostics. Defaults to False.
    """

    pair: PairSetup
    setup: ChannelSetup
    residual: np.ndarray
    fixed_level: tuple[np.ndarray, np.ndarray]
    sigma: float
    dt: float
    reference: dict[int, PassFit]
    keep_diagnostics: bool = False


class SegmentNotIdentifiable(ValueError):
    """A segment cannot be fitted: a template column has too little area in the segment,
       or the segment has fewer valid samples than columns. fit_segment reports its passes as segment_failed."""


def design_matrix(n: int, templates: list[ShapeTemplate],positions: np.ndarray,
                  dt: float, baseline: str | None) -> np.ndarray:
    """Design matrix A = [S | B] of one segment.

    Templates are the basis function of the fit: one unit-area column per pass (or merged group) with the peak at its
    position, then the baseline columns. Outside its support a template is 0, so a pass only influences the samples
    covered by its template.

    Args:
        n: Segment length in samples.
        templates: One template per column.
        positions: Peak position per column relative to the segment start in samples.
        dt: Sampling interval in s.
        baseline: Baseline model, see baseline_columns.

    Returns:
        A with the shape (n, J + number of baseline columns).
    Raises:
        SegmentNotIdentifiable: If a template has too littel area in the segment.
    """
    cols = [template_column(template, position, n, dt) for template, position in zip(templates, positions)]
    if any(c is None for c in cols):
        raise SegmentNotIdentifiable("template column has too little area in window")
    # Templates are baseline free and the segment signal contains a baseline, to avoid overweighting the
    # fitted amplitudes a separate baseline column is added.
    return np.column_stack([np.column_stack(cols)] + baseline_columns(n,baseline))

def bvls(A: np.ndarray, y: np.ndarray, n_amplitudes: int) -> tuple[np.ndarray, float, np.ndarray]:
    """Bounded least squares min ‖y − A x‖² with x ≥ 0 for the amplitudes, baseline coefficients free of restriction

    Samples with NaN in y are left out.

    Args:
        A: Design matrix [ S | B ] ober whole segment.
        y: Segment signal (may contain NaN).
        n_amplitudes: Number J of template columns (Number of passes in a segment)
    Returns:
        (x, sse, valid): coefficients [a; β], sum of squared residuals over the valid samples, bool mask of the valid samples.
    Raises:
        SegmentNotIdentifiable: If the segment has fewer valid samples than columns.
    """

    # Guard agains nan/inf values in the signal
    valid = np.isfinite(y)
    if valid.sum() < A.shape[1]:
        raise SegmentNotIdentifiable("Segments contain fewer valid samples than columns -> not identifiable")
    # Number of baseline columns that need to be unrestricted
    n_base = A.shape[1] - n_amplitudes
    # Bounds for the least squares (amplitudes >= 0, baseline >= -inf)
    # amplitudes >= 0 censors turbulence-dilution effects at zero — deliberate
    # np_r_ -> first row of bounds matrix the lower bound just concats the values in its call
    # np.full... bvls needs an upper bound (second row) the fit is not in this direction so one row of infs
    bounds = (np.r_[np.zeros(n_amplitudes), np.full(n_base, -np.inf)], np.full(A.shape[1], np.inf))
    res = lsq_linear(A[valid], y[valid], bounds=bounds, method="bvls")
    # AI-Assisted: <Opus 5.5> ; (Bug with negative values in output while bvls should only produce >= 0)
    # BVLS returns amplitudes at the bound 0 with float noise (e.g. -1e-16) -> clip to exactly 0
    res.x[:n_amplitudes] = np.maximum(res.x[:n_amplitudes], 0.0)
    resid = A[valid] @ res.x - y[valid]
    return res.x, float(resid @ resid), valid

def segment_sse(segment: np.ndarray, templates: list[ShapeTemplate], peak_positions: np.ndarray, dt: float,
                baseline: str | None) -> float:
    """SSE(τ) of the BVLS fit at the given positions np.inf if the segment is not identifiable there.

    Used by the position refinement, which only compares SSE values.
    """

    try:
        A = design_matrix(segment.size, templates, peak_positions, dt, baseline)
        return bvls(A, segment, len(templates))[1]
    except SegmentNotIdentifiable: # a template column with too little area in the segment
        return np.inf


def condition_number(segment: np.ndarray, templates: list[ShapeTemplate], peak_positions: np.ndarray, dt: float,
                     baseline: str | None) -> float:
    """Condition number of the design matrix on the valid samples (collinearity diagnostics)."""
    A = design_matrix(segment.size, templates, peak_positions, dt, baseline)
    return float(np.linalg.cond(A[np.isfinite(segment)]))


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


# AI-Assisted: <Opus 5.5> ; (Bugfix: template columns without valid samples)
def valid_area_fraction(template: ShapeTemplate, position: int, valid: np.ndarray, dt: float) -> float:
    """Share of the template area on the valid (finite) samples of the segment.

    template_column only checks the area inside the segment. If the template support
    lies on faulty (NaN) samples, the column is (almost) zero on all valid rows: the pass
    is not identifiable and a zero column breaks the collinearity grouping (cosine 0/0).

    Args:
        template: Template of the pass (unit area).
        position: Peak position relative to the segment start in samples.
        valid: Bool per segment sample, True where the signal is finite.
        dt: Sampling interval in s.

    Returns:
        Template area on valid samples inside the segment (1 = complete template).
    """
    return float(template.sample(np.arange(valid.size) - position)[valid].sum() * dt)

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
               baseline: str | None = "const") -> ModeLFitResult:
    """Final fit of one segment: amplitudes, baseline, standard errors and model.

    Design matrix A = [S | B]: one unit-area template column per pass (peak at its
    position) and the baseline columns. Amplitudes are bounded to a ≥ 0, baseline
    coefficients are free. Samples with NaN are left out.
    Standard errors: Cov = σ̂²_fit (AᵀA)⁻¹ with σ̂²_fit = SSE / (n − p) (Gl. 3.25), se_j = √Cov_jj (Gl. 3.26).

    Args:
        segment: Background-subtracted, median-centered segment signal.
        templates: One template per column.
        peak_positions: Peak position per column relative to the segment start in samples.
        dt: Sampling interval in s.
        baseline: Baseline model, see baseline_columns. Defaults to "const".

    Returns:
        Amplitudes (= plume areas), baseline coefficients, SE, SSE and the model.

    Raises:
        SegmentNotIdentifiable: If a template column has too little area in the segment, or the
            segment has fewer valid samples than columns.

    Note:
        TODO: Farren et al. states that trough turbulences small negative values are valid, bvls bounds a >= 0 for stability
        TODO: Is there a way to respect the turbulences and still provide a stable fit
    """
    # Number of vehicles in segment
    J = len(templates)
    A = design_matrix(segment.size, templates, peak_positions, dt, baseline)
    x, sse, valid = bvls(A, segment, J)
    A_v = A[valid]
    dof = A_v.shape[0] - A_v.shape[1]
    sigma2_fit = sse / dof if dof > 0 else np.inf
    try:
        # Catching because the inverse could fail due to singularities
        # Due to the fact that collinear columns are grouped this should usually not occur
        cov = sigma2_fit * np.linalg.inv(A_v.T @ A_v)
        var = np.diag(cov)[:J]
        # Guard implemented because fitting trough errors caused by 2 vehicles sharing the exact same timestamp and
        # therefore exactly collinear
        amplitude_se = np.sqrt(np.where(var >= 0, var, np.inf))   # exactly collinear columns: SE undefined, as LinAlgError
    except np.linalg.LinAlgError:
        amplitude_se = np.full(J, np.inf)
    return ModeLFitResult(x[:J], x[J:], amplitude_se, sse, A @ x)


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
        # AI-Assisted: <Opus 5.5> ; (Bugfix: guard against zero columns -> NaN, np.argmax would return the first NaN)
        # A zero column has no defined similarity, it gets 0 instead of NaN. Zero columns are excluded in fit_day
        # _fit_segment_passes (valid_area_fraction), this is only a safeguard.
        denom = np.sqrt(d[:-1] * d[1:])
        cos_nbr = np.divide(np.diag(gramm, 1), denom, out=np.zeros_like(denom), where=denom > 0)
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


@dataclass(frozen=True)
class Segment:
    """Part of a measurement segment between two anchor runs, one independent fit problem.

    It contains all passes that may influence each other. An anchor point is a sample
    outside every influence window whose residual lies inside the noise band; an anchor
    run is a run of at least min_anchor_run of them. The bounding anchor runs are part of
    the segment, neighboring segments share them.

    Attributes:
        start: First sample of the segment (start of the left anchor run, or 0).
        stop: Sample after the segment (end of the right anchor run, or n), exclusive.
        pass_ids: Indices into the expected peak positions passed to segment_day of the passes in
            this segment.
        anchor_mask: Bool per segment sample, True on anchor runs.
        anchored_left: True if a left anchor run bounds the segment (False at the start
            of the measurement segment).
        anchored_right: True if a right anchor run bounds the segment (False at the end).
    """

    start: int
    stop: int
    pass_ids: np.ndarray
    anchor_mask: np.ndarray
    anchored_left: bool
    anchored_right: bool


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

# ----------------------------------------------------------------------------- day fit, named steps

def prepare_signal(data: np.ndarray, min_physical_value: float, cfg: ModeLConfig, dt: float) -> np.ndarray:
    """ Step 1: faulty recordings -> NaN and background subtraction r(t) = x(t) - b̂(t).

    The baseline is subtracted using the same method as in the extraction (rolling percentile), the remaining
    background correction happens in the fit itself (baseline columns or fixed baseline)

    Args:
        data: Raw channel signal of the day
        min_physical_value: See ChannelSetup.min_physical_value
        cfg: Fit parameters (min_physical_run, bg_percentile, bg_rolling_window)
        dt: Sampling interval in s.
    Returns:
        Background-subtracted signal, NaN on faulty recordings (fit_mode_l skips those rows).
    """

    phys_run = as_samples(cfg.min_physical_run,dt)
    # Test all values that are faulty to none in data
    data = np.where(faulty_recording_mask(data, min_physical_value, phys_run), np.nan, data)

    return data - compute_background_series(data, cfg.bg_percentile, as_samples(cfg.bg_rolling_window, dt),
                                            min_physical_value, phys_run)


# AI-Assisted: <Opus 5.5> ; (Review finding: σ̂ = 0 on coarsely quantized channels, floor from the sensor resolution)
# Check for an error that is otherwise hard to find
def signal_resolution(data: np.ndarray) -> float:
    """Resolution q of a channel: smallest positive step between distinct finite values (0 if constant)."""
    steps = np.diff(np.unique(data[np.isfinite(data)]))
    steps = steps[steps > 0]
    return float(steps.min()) if steps.size else 0.0

def noise_level(residual: np.ndarray, influenced: np.ndarray, resolution: float = 0.0) -> tuple[float, float]:
    """Step 2: median(r_F) and σ̂ of the vehicle-free residual, see estimate_noise.

    Influence windows around the expected peaks approximate T_0; centering on median(r_F) removes the low bias of the
    percentile background.
    If more than half of the vehicle-free values are identical (coarse quantization), the MAD and σ̂ are 0 and the
    acceptance threshold κσ̂² of the refinement would accept every SSE decrease. σ̂ is therefore bounded below by the
    standard deviation of the quantization error.

    Args:
        residual: Background-subtracted signal.
        influenced: Vehicle-influenced samples, see influence_mask.
        resolution: Sensor resolution q of the raw channel, see signal_resolution. Defaults to 0 (no floor)
    """

    median, sigma = estimate_noise(residual, influenced)
    return median, max(sigma, resolution / np.sqrt(12.0))


def anchor_runs(residual: np.ndarray, influenced: np.ndarray, sigma: float, cfg: ModeLConfig,
                dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Step 3: anchor runs: vehicle free, finite, |r| < k₃σ̂, at least T_A long.

    The band excludes quiet-looking (vehicle free) samples disturbed by unregistered sources. The runs are
    shared by the segmentation and the fixed baseline.

    Returns:
        (starts, stops) of the anchor runs, see contiguous_runs.
    """
    quiet = ~influenced & np.isfinite(residual) & (np.abs(residual) < cfg.noise_band_sigma * sigma)
    return contiguous_runs(quiet, as_samples(cfg.min_anchor_run, dt))


def choose_baseline(seg: Segment, cfg: ModeLConfig, dt: float) -> str:
    """Baseline mode of segment, restricts the degrees of freedom of the background fit.

    anchored_fixed: no free parameter, the polyline through the anchor levels is subtracted before the fit
    fitted_linear: anchor runs on both sides, the segment is long enough for a linear baseline and has enough
                   anchors -> two degrees of freedom (intercept and slope).
    fitted_const: one degree of freedom (intercept)
    """
    anchor_s = int(seg.anchor_mask.sum()) * dt
    span_s = (seg.stop - seg.start) * dt
    min_anchor_s = cfg.min_anchor_time / np.timedelta64(1, "s")
    if cfg.force_anchored_fixed or anchor_s < min_anchor_s:
        return "anchored_fixed"

    long_enough = bool(span_s > cfg.t_min_linear / np.timedelta64(1, "s"))
    enough_anchor_points = anchor_s >= 2 * min_anchor_s
    if seg.anchored_left and seg.anchored_right and long_enough and enough_anchor_points:
        return "fitted_linear"
    return "fitted_const"

def merge_collinear(y: np.ndarray, templates: list[ShapeTemplate], tau_hat: np.ndarray, cfg: ModeLConfig,
                    dt: float) -> tuple[list[np.ndarray], list[ShapeTemplate], np.ndarray]:
    """Group non-separable passes (VIF on S) and build one column per group.

    Args:
        y: Segment signal
        templates: Template per pass.
        tau_hat: (Refined) peak position per pass relative to the segment start in samples
        cfg: Fit parameters (merge_vif_max).
        dt: Sampling interval in s.

    Returns:
        (groups, column templates, column positions):: index arrays into the passes, in column order; merged
        groups get a composite template of the members at their own positions, placed at the rounded mean member
        position.

    Raises:
        SegmentNotIdentifiable: If a template column has too little area in the segment.
    """

    order = np.argsort(tau_hat, kind="stable")
    cols = [template_column(templates[i], tau_hat[i], y.size, dt) for i in order]
    if any(c is None for c in cols):
        raise SegmentNotIdentifiable("template column has too little area in window")
    S = np.column_stack(cols)[np.isfinite(y)]
    groups = [order[g] for g in collinearity_grouping(S, cfg.merge_vif_max)]
    col_pos = np.array([int(round(tau_hat[g].mean())) for g in groups])
    col_templates = [templates[g[0]] if g.size == 1
                     else ShapeTemplate.composite([templates[i] for i in g], tau_hat[g] - cp)
                     for g, cp in zip(groups, col_pos)]
    return groups, col_templates, col_pos



def criteria_of_detection(area: float | np.ndarray, se: float | np.ndarray, cfg: ModeLConfig) -> bool | np.ndarray:
    """Detection criterion a ≥ k_det · se, for one column or element-wise for all columns of a fit.

    Used for PassFit.detected (pass_fits) and as significance gate of position_refinement.refine_pollutant_positions,
    so both always agree. Change the detection mechanism here.
    """
    return area >= cfg.detect_sigma * se



def pass_fits(fit: ModeLFitResult, groups: list[np.ndarray], pass_index: np.ndarray, tau0: np.ndarray,
              tau_hat: np.ndarray, seg: Segment, mode: str, templates: list[ShapeTemplate],
              cfg: ModeLConfig) -> list[PassFit]:
    """Back from columns to passes: one PassFit per pass of the fitted segment.

    Members of a merged group share SE and detection; the group area is stored once (group_amplitude of the
    first member), see fitting_result.
    """

    passes = []
    for gi, g in enumerate(groups):
        members = pass_index[g]
        # The first member of a group defines the group id
        group_id = int(members.min())
        se = float(fit.amplitudes_se[gi])
        area = float(fit.amplitudes[gi])
        detected = criteria_of_detection(area,se,cfg)
        # AI-Assisted: <Opus 5.5> ; (Review finding: edge flag on both sides)
        # First column without left anchor run / last column without right anchor run: the template may be cut at
        # the signal border and its area is extrapolated (template_column renormalizes the truncated column)
        edge = ((not seg.anchored_left) and gi == 0) or ((not seg.anchored_right) and gi == len(groups) - 1)
        for i, pi in zip(g, members):
            passes.append(PassFit(pass_index=int(pi), status=STATUS_FITTED, segment_status=mode,
                                  amplitude=area if g.size == 1 else np.nan, se=se, detected=detected,
                                  group_id=group_id, group_size=int(g.size),
                                  group_amplitude=area if pi == group_id else np.nan,
                                  position=float(tau_hat[i] + seg.start), shift=int(tau_hat[i] - tau0[i]),
                                  edge_extrapolated=edge, template=templates[i].provenance))
    return passes


def fit_segment(day: ChannelDay, seg: Segment, pass_index: np.ndarray, templates: list[ShapeTemplate],
                tau0: np.ndarray, band: tuple[np.ndarray, np.ndarray]|None, priority: np.ndarray | None
                ) -> tuple[list[PassFit], SegmentDiagnostics|None]:
    """Step 5: Fit one segment

    Args:
        day: Day-level inputs of the channel
        seg: The segment
        pass_index: Light barrier pass index per pass of the segment
        templates: Template per pass of the segment
        tau0: Initial positions relative to the segment start in samples
        band: (lo, hi) search band C_j per pass relative to the segment start, None without refinement.
        priority: Optional refinement priority per pass (e.g. CO2 amplitudes for pollutant refinement order),
                  handled by refinement strategy

    Returns:
        (PassFit per pass of the segment, diagnostics or None).
    """

    cfg= day.pair.fit
    passes, model, contributions, diagnostics = _fit_segment_passes(day, seg, pass_index, templates, tau0, band, priority)
    # QA checks and non-detected policy per pass (extension points, see fit_checks)
    y = day.residual[seg.start:seg.stop] - (0.0 if model is None else model[1])
    bands = {} if band is None else {int(i): (int(lo) + seg.start, int(hi) + seg.start)
                                     for i, lo, hi in zip(pass_index, band[0], band[1])}
    ctx = SegmentContext(y=y, model=None if model is None else model[0], start=seg.start, sigma=day.sigma, cfg=cfg,
                         templates={int(i): t for i, t in zip(pass_index, templates)}, contributions=contributions,
                         bands=bands, reference=day.reference)
    passes = [day.pair.non_detect(run_checks(p, ctx, day.pair.qa_checks), cfg) for p in passes]
    return passes, diagnostics



def _fit_segment_passes(day: ChannelDay, seg: Segment, pass_index: np.ndarray, templates: list[ShapeTemplate],
                        tau0: np.ndarray, band: tuple[np.ndarray, np.ndarray] | None, priority: np.ndarray | None
                        ) -> tuple[list[PassFit], tuple[np.ndarray, np.ndarray] | None, dict[int, np.ndarray],SegmentDiagnostics | None]:
    """Fit part of fit_segment (baseline mode, identifiable passes, refinement, merging, BVLS, PassFits)

    Args:
         See fit_segment.
    Returns:
         (PassFit per pass, (model, fixed baseline) ober the segment or None if the segment failed, column
         contribution a · s(τ̂) per group_id, diagnostics or None)
    """
    cfg = day.pair.fit
    dt = day.dt
    refinement = day.setup.refinement
    keep_diagnostics = day.keep_diagnostics
    # Signal to be fitted -> residual because it is background corrected
    y = day.residual[seg.start:seg.stop]
    mode = choose_baseline(seg,cfg,dt)
    base = np.zeros(y.size)
    if mode == "anchored_fixed":
        #Polyline through the anchor levels is subtracted, the fitter has no baseline column to work with
        centers, levels = day.fixed_level
        base = np.interp(np.arange(seg.start, seg.stop), centers, levels) if centers.size else base
        y = y - base
        baseline = None
    else:
        baseline = "const" if mode == "fitted_const" else "linear"
    # AI-Assisted: <Opus 5.5> ; (Bugfix: passes whose template lies on faulty samples)
    # Passes whose template lies (almost) entirely on faulty (NaN) samples are not identifiable: their column
    # is zero on all valid rows. They are left out of the segment fit and reported as not_identifiable.
    # The refinement never moves a template onto such a position (a zero column cannot lower the SSE).
    # -> introduced the ok masking
    # Needed because prepare_signal sets invalid signal parts to NaN
    valid_y = np.isfinite(y)
    # Pre Fit prep
    # 1. The template has enough area in the segment > MIN_AREA_FRACTION
    valid_area = np.array([valid_area_fraction(t, p, valid_y, dt) > MIN_AREA_FRACTION for t, p in zip(templates, tau0)])
    # 2. Assign all passes that did not pass the criteria the failed mark
    # If every pass in a segment is invalid the whole segment is failed (can happen for example at data edges with
    # the pass at the edge beeing in its own segment) -> if stops before fitting
    segment_status = mode if valid_area.any() else "failed" # -> The baseline mode
    unfitted = []
    for index, start_pos, template, identifiable in zip(pass_index, tau0, templates, valid_area):
        if not identifiable:
            # Add the invalid pass to unfitted
            unfitted.append(PassFit(int(index), STATUS_NOT_IDENTIFIABLE, segment_status=segment_status,
                                    position=float(start_pos + seg.start), template=template.provenance))
    if not valid_area.any():
        return unfitted, None, {}, _failed_diagnostics(seg,pass_index) if keep_diagnostics else None

    # 3. Get the information of valid passes in the segment (where a fit/refinement should be performed)
    # idx and tau0 of valid passes
    idx, valid_tau0 = pass_index[valid_area], tau0[valid_area]
    # templates of the valid passes
    tmpl = [t for t, valid in zip(templates, valid_area) if valid]

    # Performing the position refinement and fitting (also only pass information for valid passes)
    try:
        #1. Position Refinement
        tau_hat = valid_tau0
        if refinement is not None and band is not None:
            tau_hat = refinement(y,tmpl,valid_tau0, band[0][valid_area], band[1][valid_area], dt, baseline,
                                 day.sigma ** 2, cfg, None if priority is None else priority[valid_area])
        #2. Grouping
        groups, col_templates, col_pos = merge_collinear(y, tmpl, tau_hat, cfg, dt)
        #3. Perform fit
        fit = fit_mode_l(y, col_templates, col_pos, dt, baseline=baseline)
    except SegmentNotIdentifiable:
        failed = [PassFit(int(pi), STATUS_SEGMENT_FAILED, segment_status="failed", position=float(p + seg.start),
                          template=t.provenance) for pi, p, t in zip(idx, valid_tau0, tmpl)]
        return failed + unfitted, None, {}, _failed_diagnostics(seg, pass_index) if keep_diagnostics else None
    passes = pass_fits(fit, groups, idx, valid_tau0, tau_hat, seg, mode, tmpl, cfg) + unfitted
    # Contribution a · s(τ̂) of every column, keyed by the group_id (first member) of its group
    contributions = {int(idx[g].min()): a * template_column(t, p, y.size, dt)
                     for g, a, t, p in zip(groups, fit.amplitudes, col_templates, col_pos)}
    diagnostics = None
    if keep_diagnostics:
        valid = np.isfinite(y)
        columns = tuple((tuple(int(i) for i in idx[g]), contributions[int(idx[g].min())]) for g in groups)
        diagnostics = SegmentDiagnostics(seg.start, seg.stop, mode, tuple(int(i) for i in pass_index),
                                         tuple(fit.baseline_coeffs),
                                         condition_number(y,col_templates, col_pos, dt, baseline),
                                         residual_rms=float(np.sqrt(fit.sse / valid.sum())),
                                         model = fit.model + base,
                                         columns=columns
                                         )

    return passes, (fit.model, base), contributions, diagnostics



def _failed_diagnostics(seg: Segment, pass_index: np.ndarray) -> SegmentDiagnostics:
    return SegmentDiagnostics(seg.start, seg.stop, "failed", tuple(int(i) for i in pass_index))


def fit_channel_day(register: MeasurementRegister, pair: PairSetup, setup: ChannelSetup, positions: np.ndarray,
                    band: tuple[np.ndarray, np.ndarray] |None, priority: np.ndarray | None = None,
                    reference: ChannelDayResult |None = None, keep_diagnostics: bool = False) -> ChannelDayResult:
    """Fit the plume area of every light barrier pass of one channel of a pair and measurement segment.

    Called by pair_fitting (fit_co2_day, fit_pollutant_day), which derives positions and band from the peak windows

    Args:
        register: Measurement data of the segment (one day)
        pair: The pair (CO2 + poll couple): fit parameters, QA checks and non-detect policy
        setup: The fitted channel of the pairs setup (pair.co2 or pair.pollutant) (channel specific config)
        positions: Initial peak positions tau_0 per light barrier pass in samples
        band: (lo, hi) search band C_j per light barrier pass
        priority: Optional refinement priority per light barrier pass
        reference: CO2 result of the same register (pollutant fits)
        keep_diagnostics: Keep the day residual and SegmentDiagnostics
    Returns:
        One PassFit per light barrier pass.
    Raises:
        ValueError: If positions or band do not habe one entry per pass, or no vehicle-free sample exists for noise
                    estimation.
    """
    n_pass = register.vehicle_pass_times.size
    cfg = pair.fit
    positions = np.asarray(positions, dtype=float)

    no_band_entry_per_pass = bool(band is not None and (len(band[0]) != n_pass or len(band[1]) != n_pass))
    if positions.size != n_pass or no_band_entry_per_pass:
        raise ValueError(f"positions and band need one entry per light barrier pass ({n_pass})")

    # Value Preparation
    dt = register.dt
    templates = [setup.templates(register.source_day,i) for i in range(n_pass)]
    residual = prepare_signal(register.get_channel_data_by_name(setup.channel), setup.min_physical_value, cfg, dt)
    n = residual.size

    # AI-Assisted: <Opus 5.5> ; (Bugfix from a review)
    # Passes behind the end of the signal (e.g. data cut off after a faulty period) would put a template column
    # with too little area into the last segment and fail it, so they get no segment (outside_signal)
    inside = np.flatnonzero(np.isfinite(positions) & (positions >= 0) & (positions < n))
    tau0 = positions[inside].astype(int)
    # Get vehicle free signal mask
    influenced = influence_mask(n,tau0, as_samples(cfg.influence_before,dt), as_samples(cfg.influence_after,dt))
    resolution = signal_resolution(register.get_channel_data_by_name(setup.channel))
    median, sigma = noise_level(residual, influenced, resolution)
    # Shift the background signal by the median of vehicle free samples to "fix" baseline bias
    residual = residual - median
    # Get anchor runs
    run_starts, run_stops = anchor_runs(residual, influenced, sigma, cfg, dt)
    # Split in segments
    segments = segment_day(n,tau0,run_starts,run_stops)
    # Needed for fixed baseline: Median Level of each anchor run for Polyline Baseline, Point is in the middle of the run
    fixed_level = ((run_starts + run_stops)/2, np.array([np.median(residual[a:b]) for a, b in zip(run_starts, run_stops)]))

    #Prepare Results dict with already faulty passes
    # np.setdiff1d(a,b) -> all passes in a that are not in b
    results = {int(i): PassFit(int(i), STATUS_OUTSIDE_SIGNAL, position=float(positions[i]),
                               template = templates[i].provenance) for i in np.setdiff1d(np.arange(n_pass), inside)}
    diagnostics = []
    # Pass all the day constant parameters as a class / for pollutant runs the reference passes of the co2 run
    day = ChannelDay(pair, setup, residual, fixed_level, sigma, dt,
                     {} if reference is None else {p.pass_index: p for p in reference.passes}, keep_diagnostics)
    # Loop through all the segments and perform the fitting
    for seg in segments:
        idx = inside[seg.pass_ids]
        # -seg.start conversion from global channel idx to segment idx
        # Get the movement bands for all passes in this segment
        seg_band = None if band is None else (np.asarray(band[0])[idx].astype(int) - seg.start,
                                              np.asarray(band[1])[idx].astype(int) - seg.start,)
        passes, diag = fit_segment(day = day,seg = seg, pass_index=idx,
                                   templates=[templates[i] for i in idx],
                                   tau0 = tau0[seg.pass_ids]-seg.start,
                                   band = seg_band,
                                   priority=None if priority is None else np.asarray(priority)[idx])
        results.update({p.pass_index: p for p in passes})
        if diag is not None:
            diagnostics.append(diag)

    return ChannelDayResult(channel=setup.channel,source_day=register.source_day,dt=dt,
                            passes=[results[i] for i in range(n_pass)],
                            sigma = sigma,
                            diagnostics=diagnostics if keep_diagnostics else None,
                            residual=residual if keep_diagnostics else None)

