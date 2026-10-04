"""QA checks of the template extraction.

All functions work on windows already cut by template_extraction.py, one plume per row.

CO₂ channel:
    assess_vectorized_lb_centered: Window around the light barrier trigger
        (faulty recording, peak height, peak position).
    assess_iterative_peak_centered: Window around the CO₂ peak (multiple peaks, tail anomaly).

Pollutant channel:
    assess_pollutant_peak_centered: Window around the CO₂ peak (faulty recording,
        peak search on the smoothed signal, multiple peaks, band around the CO₂ peak).

Noise-derived thresholds (h_min, ρ_min, R_krit) are resolved once per segment by
resolve_qa_thresholds. Thesis: chapter "The PlumeFit algorithm", section "Quality checks".
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Assistance with Docstring writing)

from dataclasses import replace

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from src.plume_template_extraction.extraction_config import ChannelQAConfig
from src.plume_template_extraction.plume_status import PlumeStatus
from src.shared_services.noise_and_background import estimate_noise, contiguous_runs, faulty_recording_mask

# Minimum number of vehicle-free reference windows for the tail threshold calibration.
MIN_TAIL_WINDOWS = 30

def null_data_mask(windows: np.ndarray, min_physical_value: float, min_physical_run: int) -> np.ndarray:
    """Flag windows that contain NaN or a faulty recording (NULL_DATA criterion).

        Args:
            windows: Plume windows, one per row.
            min_physical_value: See faulty_recording_mask.
            min_physical_run: See faulty_recording_mask [samples].

        Returns:
            Bool per window, True if the window must be rejected.
        """
    nan = np.isnan(windows).any(axis=1)
    faulty_recording = faulty_recording_mask(windows, min_physical_value, min_physical_run).any(axis=1)

    return nan | faulty_recording


def _tail_smooth_window(n_tail: int) -> int:
    """Rolling mean width L = max(2, ⌊n_tail / 4⌋) for the tail statistic.

        Shared by calibration and check, both must smooth identically.
        """
    return max(2, n_tail // 4)

def tail_rise_statistic(tail: np.ndarray, tail_smooth_window: int) -> float:
    """Cumulative re-rise R of a signal tail.

    Smooths the tail with a rolling mean and sums all positive increments. A clean
    exponential decay gives R ≈ 0, a second plume or a background step gives large R.


    Args:
        tail: Signal after the main peak, 1-D.
        tail_smooth_window: Rolling mean width L [samples].

    Returns:
        R = Σ max(0, x[k+1] − x[k]) in the channel unit, 0.0 if the tail is too short
        for smoothing.
    """

    # AI-Assisted: <Fable 5> ; (Review of Rolling-Mean implementation)

    if len(tail) < tail_smooth_window + 1:
        return 0.0

    # Implements the rolling mean for tail smoothing (similar to pandas rolling mean)
    kernel = np.ones(tail_smooth_window) / tail_smooth_window
    # Every element of smoothed is the mean of tail_smooth_window elements of tail
    smoothed = np.convolve(tail, kernel, mode='valid')
    # Inside Out: diff->rise/fall->fall=0:only rising is interesting->sum up all positive increments
    return float(np.sum(np.maximum(np.diff(smoothed), 0)))


def _tail_threshold(residual: np.ndarray, influenced: np.ndarray,
                    tail_len: int, percentile: float) -> float:
    """Calibrate the absolute tail threshold R_krit from vehicle-free windows.

    The vehicle-free samples are split into contiguous runs, every run is cut into
    non-overlapping windows of tail_len samples. R of all windows
    is the distribution of the statistic without a vehicle, its percentile is R_krit:
    a tail is flagged if it re-rises more than noise alone does in percentile % of the
    vehicle-free windows.

    Args:
        residual: Background-subtracted signal of the segment.
        influenced: Vehicle-influenced samples, see influence_mask.
        tail_len: Tail length = window_after_peak in samples.
        percentile: Percentile q_an of the distribution.

    Returns:
        R_krit in the channel unit.

    Raises:
        ValueError: If fewer than MIN_TAIL_WINDOWS vehicle-free windows exist.
    """
    smooth_window = _tail_smooth_window(tail_len)
    starts, stops = contiguous_runs(~influenced & np.isfinite(residual), tail_len)
    stats = [tail_rise_statistic(residual[k:k + tail_len], smooth_window)
             for a, b in zip(starts, stops) for k in range(a, b - tail_len + 1, tail_len)]
    if len(stats) < MIN_TAIL_WINDOWS:
        raise ValueError(f"Only {len(stats)} vehicle-free windows available "
                         f"for tail threshold calibration")
    return float(np.percentile(stats, percentile))


def resolve_qa_thresholds(qa_config: ChannelQAConfig, residual: np.ndarray,
                          influenced: np.ndarray, tail_len: int | None = None) -> ChannelQAConfig:
    """Fill the noise-derived thresholds of a QA config for one segment.

    h_min = median(r_F) + k₁ · σ, ρ_min = k₂ · σ and R_krit (see _tail_threshold) are
    only computed where the config holds None. Thresholds given by the user are kept.

    Args:
        qa_config: QA config, thresholds possibly None.
        residual: Background-subtracted signal of the segment.
        influenced: Vehicle-influenced samples, see influence_mask.
        tail_len: Tail length in samples for the R_krit calibration. None skips it
            (pollutant channels, which have no tail check).

    Returns:
        Copy of qa_config with resolved thresholds.
    """
    need_tail = (tail_len is not None and qa_config.tail_rise_ratio is None
                 and qa_config.tail_rise_abs is None)
    h_min = qa_config.min_peak_above_bg
    rho_min = qa_config.min_prominence_floor
    if h_min is not None and rho_min is not None and not need_tail:
        return qa_config
    if h_min is None or rho_min is None:
        med, sigma = estimate_noise(residual, influenced)
        # h_min is compared to the height above b, which is biased low -> includes the median offset.
        # ρ_min is a height difference between peak and surroundings -> no offset
        if h_min is None:
            h_min = med + qa_config.peak_above_bg_sigma * sigma
        if rho_min is None:
            rho_min = qa_config.prominence_floor_sigma * sigma
    tail_abs = qa_config.tail_rise_abs
    if need_tail:
        tail_abs = _tail_threshold(residual, influenced, tail_len, qa_config.tail_percentile)
    return replace(qa_config, min_peak_above_bg=h_min, min_prominence_floor=rho_min,
                   tail_rise_abs=tail_abs)


def assess_vectorized_lb_centered(plumes: np.ndarray, samples_before: int, peak_search_window: int,
                                  backgrounds: np.ndarray, qa_config: ChannelQAConfig,
                                  min_physical_run: int) -> tuple[np.ndarray, np.ndarray]:
    """First CO₂ QA stage on windows around the light barrier trigger, vectorized.

    Checks for faulty recordings, finds the CO₂ peak as the maximum after the
    trigger and checks its height and position.

    Args:
        plumes: CO₂ windows around the trigger, one per row.
        samples_before: Samples before the trigger in each window -> the column
            of the trigger (window_before in samples).
        peak_search_window: Samples after the trigger searched for the peak
            (peak_search_after in samples).
        backgrounds: b(t_LB) per window.
        qa_config: Resolved CO₂ QA config.
        min_physical_run: See faulty_recording_mask in samples.

    Returns:
        Tuple (statuses, rel_peak_idx):
            statuses: PlumeStatus per window (VALID, NO_PEAK or NULL_DATA).
            rel_peak_idx: Peak position relative to the trigger in samples, equals
                the trigger-to-peak delay.
    """

    null_mask = null_data_mask(plumes, qa_config.min_physical_value, min_physical_run)

    # Peak search in the window of interest (search window after the LB pass), high SNR -> plain argmax
    window = plumes[:, samples_before:samples_before + peak_search_window]
    # Peak index relative to samples_before and peak height above background, for every plume
    rel_peak_idx = np.argmax(window, axis=1)
    peak_height = window.max(axis=1) - backgrounds
    # Initialize all plumes as valid
    statuses = np.full(plumes.shape[0], PlumeStatus.VALID, dtype=object)

    # Check for NO_PEAK
    statuses[peak_height < qa_config.min_peak_above_bg] = PlumeStatus.NO_PEAK
    # A maximum on the first or last sample of the search window is no peak but a
    # rising or falling signal reaching beyond the window
    at_edge = (rel_peak_idx == 0) | (rel_peak_idx == peak_search_window - 1)
    statuses[at_edge] = PlumeStatus.NO_PEAK

    # NULL_DATA overrides all other statuses (prominence cheks on NaN windows is meaningless)
    statuses[null_mask] = PlumeStatus.NULL_DATA

    return statuses, rel_peak_idx


def check_tail_anomaly(plume: np.ndarray,
                       peak_idx: int,
                       threshold: float,
                       tail_smooth_window: int) -> bool:
    """Check whether the tail after the peak re-rises more than the threshold.

    Args:
        plume: One peak-centered plume.
        peak_idx: Column of the peak.
        threshold: R_krit in the channel unit.
        tail_smooth_window: Rolling mean width in samples.

    Returns:
        True if R_tail > R_krit.
    """

    return tail_rise_statistic(plume[peak_idx:], tail_smooth_window) > threshold


def check_multiple_peaks(plume: np.ndarray, peak_idx: int, background: float,
                         min_prominence_ratio: float,
                         min_prominence_floor: float) -> bool:
    """Check whether a plume has more than one prominent peak.

    A peak counts if its prominence exceeds max(ρ_min, ρ_rel · h), with h the height
    of the main peak above the background.

    Args:
        plume: One peak-centered plume.
        peak_idx: Column of the main peak.
        background: b(t_LB) of the plume.
        min_prominence_ratio: ρ_rel.
        min_prominence_floor: ρ_min in the channel unit.

    Returns:
        True if more than one prominent peak is found.
    """

    peak_height = plume[peak_idx] - background
    # Defines how high a peak must be to be considered prominent
    effective_prominence = max(min_prominence_floor, peak_height * min_prominence_ratio)
    # Find all peaks with the defined prominence
    peaks, _ = find_peaks(plume, prominence=effective_prominence)
    # Returns True if more than one prominent peak is detected
    return len(peaks) > 1


def assess_iterative_peak_centered(plumes: np.ndarray, backgrounds: np.ndarray, peak_index: int,
                                   window_after: int, qa_config: ChannelQAConfig, min_physical_run: int = 1) -> np.ndarray:
    """Second CO₂ QA stage on windows around the CO₂ peak, one plume at a time.

    Order: faulty recording, multiple peaks, tail anomaly. The first failing check
    sets the status.

    Args:
        plumes: Peak-centered CO₂ plumes, one per row.
        backgrounds: b(t_LB) per plume.
        peak_index: Column of the peak (window_before_peak in samples).
        window_after: Samples after the peak (window_after_peak in samples), sets the
            tail smoothing width.
        qa_config: Resolved CO₂ QA config.
        min_physical_run: See faulty_recording_mask in samples.

    Returns:
        PlumeStatus per plume (VALID, NULL_DATA, MULTIPLE_PEAKS or TAIL_ANOMALY).
    """
    statuses = []

    tail_smooth_window = _tail_smooth_window(window_after)

    # Checks every plume on nan and below physical value -> NULL_DATA
    null_mask = null_data_mask(plumes, qa_config.min_physical_value, min_physical_run)
    # Checks every plume for multiple peaks and tail anomaly
    for plume, bg, is_null in zip(plumes, backgrounds, null_mask):

        if is_null:
            statuses.append(PlumeStatus.NULL_DATA)
            continue

        if check_multiple_peaks(plume, peak_index, bg, qa_config.min_prominence_ratio,
                                qa_config.min_prominence_floor):
            statuses.append(PlumeStatus.MULTIPLE_PEAKS)
            continue
        # Calibrated absolute threshold if available, else relative to this plume's peak heigh
        tail_threshold = (qa_config.tail_rise_abs if qa_config.tail_rise_abs is not None
                          else qa_config.tail_rise_ratio * (plume[peak_index] - bg))
        if check_tail_anomaly(plume, peak_index, tail_threshold, tail_smooth_window):
            statuses.append(PlumeStatus.TAIL_ANOMALY)
            continue
        # All QA Passed -> VALID
        statuses.append(PlumeStatus.VALID)
    return np.array(statuses, dtype=object)


def assess_pollutant_peak_centered(windows: np.ndarray, backgrounds: np.ndarray,
                                   qa_config: ChannelQAConfig,
                                   band_before: int, band_after: int,
                                   center: int, smooth_window: int,
                                   min_physical_run: int) -> tuple[np.ndarray, np.ndarray]:
    """QA and peak search for pollutant windows cut around the CO₂ peak.

    The peak search runs on a Savitzky-Golay smoothed copy (lower SNR than CO₂), the
    window itself stays unchanged. Candidates must exceed ρ_min in prominence,
    a peak counts if its prominence ≥ ρ_rel · height of the highest candidate
    and its height ≥ h_min. Exactly one peak must remain, and it must lie in the band
    [center − band_before, center + band_after].

    Args:
        windows: Pollutant windows centered on the CO₂ peak, one per row.
        backgrounds: Pollutant b(t_LB) per window.
        qa_config: Resolved pollutant QA config.
        band_before: Accepted peak position before the CO₂ peak in samples.
        band_after: Accepted peak position after the CO₂ peak in samples.
        center: Column of the CO₂ peak in the window.
        smooth_window: Savitzky-Golay window in samples, odd and > smooth_polyorder.
        min_physical_run: See faulty_recording_mask in samples.

    Returns:
        Tuple (statuses, offsets):
            statuses: PlumeStatus per window (VALID, NULL_DATA, NO_PEAK or MULTIPLE_PEAKS).
            offsets: Pollutant peak position relative to the CO₂ peak in samples,
                0 where no valid peak was found.
    """
    number_of_plumes = windows.shape[0]
    statuses = np.full(number_of_plumes, PlumeStatus.VALID, dtype=object)
    offsets = np.zeros(number_of_plumes, dtype=int)

    # Guard against below physical value and nan
    statuses[null_data_mask(windows, qa_config.min_physical_value, min_physical_run)] = PlumeStatus.NULL_DATA
    valid = statuses == PlumeStatus.VALID

    # AI-Assisted: <Opus 5> ; (Review Filter Implementation)

    # The Pollutant Channel SNR is lower than the CO2 Channel SNR therefor it is smoothened
    # This Filter was chosen because of its capability of removing noise while preserving features (in bounds)
    # of the signal -> Peak-Position
    # The smoothed signal is only used to find the peaks, it does not replace the original data
    smooth = np.full_like(windows, np.nan)
    smooth[valid] = savgol_filter(windows[valid], window_length=smooth_window,
                                  polyorder=qa_config.smooth_polyorder, axis=1)

    # The pollutant search band needs to be relativ to the co2 peak center
    band_lo = center - band_before
    band_hi = center + band_after

    for plume_idx in np.flatnonzero(valid):
        # Scipy find_peaks is performed on the smoothed signal window
        candidates, props = find_peaks(smooth[plume_idx], prominence=qa_config.min_prominence_floor)
        # Regrading prominence a floor is given so peaks underneath a certain threshold (noise peaks)
        # are not even counted as peaks
        if candidates.size == 0:
            statuses[plume_idx] = PlumeStatus.NO_PEAK
            continue

        # Prominence is judged on the whole window, not only inside the band:
        # a second plume next to the band still makes the window unusable
        peak_heights = smooth[plume_idx, candidates] - backgrounds[plume_idx]
        main_peak_height = peak_heights.max()
        prom_valid = props["prominences"] >= qa_config.min_prominence_ratio * main_peak_height
        height_valid = peak_heights >= qa_config.min_peak_above_bg
        prominent = candidates[prom_valid & height_valid]

        if prominent.size == 0:
            # This plume has no valid peaks
            statuses[plume_idx] = PlumeStatus.NO_PEAK
        elif prominent.size > 1:
            # This plume has multiple valid peaks -> potential peak overlap
            statuses[plume_idx] = PlumeStatus.MULTIPLE_PEAKS
        else:
            # Valid Peak found -> Pollutant peak found
            # Peak is only valid if it sits in the tight search window around the co2 peak
            peak = prominent[0]
            if band_lo <= peak <= band_hi:
                offsets[plume_idx] = peak - center
            else:
                statuses[plume_idx] = PlumeStatus.NO_PEAK

    return statuses, offsets
