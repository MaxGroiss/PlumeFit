""" This module contains all QA checks done throughout the extraction pipeline

CO2 channel:
    assess_vectorized_lb_centered (light-barrier centered window)
    assess_iterative_peak_centered (peak centered window)

Pollutant channel:
    assess_pollutant_peak_centered (window centered on the co2 peak)

All the functions in this module operate in already cut windows provided by template_extraction.py
Noise Relate parameter estimations are justified in src/docs/Herleitungen_WIP.pdf
TODO: Replace WIP pdf with thesis reference once ready.
"""
# This file contains code created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)


from dataclasses import replace

import numpy as np
from scipy.signal import find_peaks, savgol_filter


from src.plume_template_extraction.extraction_config import ChannelQAConfig
from src.plume_template_extraction.plume_status import PlumeStatus
from src.shared_services.noise_and_background import estimate_noise, contiguous_runs, faulty_recording_mask



def _null_mask(windows: np.ndarray, min_physical_value: float, min_physical_run: int) -> np.ndarray:
    # NULL_DATA criterion: any nan or below-physical value inside the window
    return (np.isnan(windows).any(axis=1)
            | faulty_recording_mask(windows, min_physical_value, min_physical_run).any(axis=1))


# Tail Anomaly QA

def _tail_smooth_window(window_after: int) -> int:
    # Calibration and QA check must smooth the tail identically
    return max(2, window_after // 4)

def tail_rise_statistic(tail: np.ndarray, tail_smooth_window: int) -> float:
    """Cumulative re-rise of a signal tail.

    Smooths the tail with a rolling mean and sums all positive increments.

    :param tail: (np.ndarray) 1-D signal tail (window after the main peak).
    :param tail_smooth_window: (int) Rolling mean width for tail smoothing.
    :return: (float) Maximum cumulative re-rise, 0.0 if the tail is too short for smoothing.
    """

    # AI-Assisted: <Fable 5> ; (Review Rolling Mean Implementation)

    # Checks if the tail is long enough for smoothing
    if len(tail) < tail_smooth_window + 1:
        return 0.0

    # Implements the rolling mean for tail smoothing (similar to pandas rolling mean)
    kernel = np.ones(tail_smooth_window) / tail_smooth_window
    # Every element of smoothed is the mean of tail_smooth_window elements of tail
    smoothed = np.convolve(tail, kernel, mode='valid')
    # Inside Out: diff->rise/fall->fall=0:only rising is interesting->sum up all positive increments
    cumulative_rise = np.cumsum(np.maximum(np.diff(smoothed), 0))
    return float(np.max(cumulative_rise))


def _tail_threshold(residual: np.ndarray, influenced: np.ndarray,
                    tail_len: int, percentile: float) -> float:
    """Empirical calibration of the absolute tail-rise threshold.

    Evaluates the tail statistic on all consecutive vehicle-free windows of tail_len
    samples — the distribution of the statistic under "no vehicle present" — and
    returns the requested percentile. A tail is flagged if its cumulative re-rise exceeds
    the value that noise alone stays below in tail_percentile % of vehicle-free windows.

    :param residual: (np.ndarray) Background subtracted channel signal of the day
    :param influenced: (np.ndarray) Bool mask of vehicle-influenced samples
    :param tail_len: (int) Tail length in samples (window after the peak)
    :param percentile: (float) Percentile of the distribution used as threshold
    :return: (float) Absolute tail-rise threshold in signal units
    :raises ValueError: Too few vehicle-free windows
    """
    smooth_window = _tail_smooth_window(tail_len)
    # Split the free samples into contiguous runs, chop every run into tail_len windows
    starts, stops = contiguous_runs(~influenced & np.isfinite(residual), tail_len)
    stats = [tail_rise_statistic(residual[k:k + tail_len], smooth_window)
             for a, b in zip(starts, stops) for k in range(a, b - tail_len + 1, tail_len)]
    if len(stats) < 30:
        raise ValueError(f"Only {len(stats)} vehicle-free windows available "
                         f"for tail threshold calibration")
    return float(np.percentile(stats, percentile))


def resolve_qa_thresholds(qa_config: ChannelQAConfig, residual: np.ndarray,
                          influenced: np.ndarray, tail_len: int | None = None) -> ChannelQAConfig:
    """Resolves the noise-derived QA thresholds of a channel for one day.

    :param qa_config: (ChannelQAConfig) QA configuration, thresholds possibly None
    :param residual: (np.ndarray) Background subtracted channel signal of the day
    :param influenced: (np.ndarray) Bool mask of vehicle-influenced samples
    :param tail_len: (int | None) Tail length for the calibration, None skips it
                     (channels whose QA never runs the tail check)
    :return: (ChannelQAConfig) Thresholds
    """
    need_tail = (tail_len is not None and qa_config.tail_rise_ratio is None
                 and qa_config.tail_rise_abs is None)
    peak = qa_config.min_peak_above_bg
    floor = qa_config.min_prominence_floor
    if peak is not None and floor is not None and not need_tail:
        return qa_config
    if peak is None or floor is None:
        med, sigma = estimate_noise(residual, influenced)
        if peak is None:
            peak = med + qa_config.peak_above_bg_sigma * sigma
        if floor is None:
            floor = qa_config.prominence_floor_sigma * sigma
    tail_abs = qa_config.tail_rise_abs
    if need_tail:
        tail_abs = _tail_threshold(residual, influenced, tail_len, qa_config.tail_percentile)
    return replace(qa_config, min_peak_above_bg=peak, min_prominence_floor=floor,
                   tail_rise_abs=tail_abs)


def assess_vectorized_lb_centered(plumes: np.ndarray, samples_before: int, peak_search_window: int,
                                  backgrounds: np.ndarray, qa_config: ChannelQAConfig,
                                  min_physical_run: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized quality assessment for LB centered plumes.

    :param plumes: (np.ndarray) Isolated plumes
    :param samples_before: (int)  Window constraints left
    :param peak_search_window: (int) Window constraints right
    :param backgrounds: (np.ndarray) Background values
    :param qa_config: (ChannelQAConfig) QA configuration
    :param min_physical_run: Minimum consecutive below-threshold samples to count as faulty
    :return: Tuple (np.ndarray, np.ndarray) of plume statuses and relative peak indices
    """

    # Checks every plume on nan and below physical value -> NULL_DATA
    null_mask = _null_mask(plumes, qa_config.min_physical_value, min_physical_run)

    # Peak search in the window of interest (search window after the LB pass), high SNR -> plain argmax
    window = plumes[:, samples_before:samples_before + peak_search_window]
    # Peak index relative to samples_before and peak height above background, for every plume
    rel_peak_idx = np.argmax(window, axis=1)
    peak_height = window.max(axis=1) - backgrounds
    # Initialize all plumes as valid
    statuses = np.full(plumes.shape[0], PlumeStatus.VALID, dtype=object)

    # Check for NO_PEAK
    statuses[peak_height < qa_config.min_peak_above_bg] = PlumeStatus.NO_PEAK

    # Assign Null Mask overrides all other statuses
    statuses[null_mask] = PlumeStatus.NULL_DATA

    return statuses, rel_peak_idx


def check_tail_anomaly(plume: np.ndarray,
                       peak_idx: int,
                       threshold: float,
                       tail_smooth_window: int) -> bool:
    """Detect sustained re-rises in the plume tail.

        :param plume: (np.ndarray) Isolated plumes
        :param peak_idx: (int) Index of the main peak.
        :param threshold: (float) Absolute threshold for the cumulative re-rise in signal units.
        :param tail_smooth_window: (int) Rolling mean width for tail smoothing.
        :returns: (bool) True if the tail shows an anomalous re-rise.
        """

    # Defines the window after the peak as the tail
    return tail_rise_statistic(plume[peak_idx:], tail_smooth_window) > threshold


def check_multiple_peaks(plume: np.ndarray, peak_idx: int, background: float,
                         min_prominence_ratio: float,
                         min_prominence_floor: float) -> bool:
    """Check for more than one prominent peak.

        :param plume: (np.ndarray) Isolated plumes
        :param peak_idx: (int) Index of the peak.
        :param background: (float) Background value.
        :param min_prominence_ratio: (float) Prominence threshold as fraction of peak height.
        :param min_prominence_floor: (float) Absolute minimum prominence.
        :returns: (bool) True if multiple peaks are detected.
        """
    peak_height = plume[peak_idx] - background
    # Defines how high a peak must be to be considered prominent
    effective_prominence = max(min_prominence_floor, peak_height * min_prominence_ratio)
    # Find all peaks with the defined prominence
    peaks, _ = find_peaks(plume, prominence=effective_prominence)
    # Returns True if more than one prominent peak is detected
    return len(peaks) > 1


def assess_iterative_peak_centered(plumes: np.ndarray, backgrounds: np.ndarray, peak_index: int,
                                   window_after: int, qa_config: ChannelQAConfig) -> np.ndarray:
    """
    Iterative assessment of plume quality based on peak centered plumes.

    :param plumes: (np.ndarray) Peak Centered Plumes
    :param backgrounds: (np.ndarray) Background values
    :param peak_index: (int) Index of the Main Peak of the Plume (Window constraints left)
    :param window_after: (int) Window constraints right
    :param qa_config: (ChannelQAConfig) QA configuration
    :return: (np.ndarray) List of plume statuses
    """
    statuses = []

    tail_smooth_window = _tail_smooth_window(window_after)

    # Checks every plume for multiple peaks and tail anomaly
    for plume, bg in zip(plumes, backgrounds):

        if check_multiple_peaks(plume, peak_index, bg, qa_config.min_prominence_ratio,
                                qa_config.min_prominence_floor):
            statuses.append(PlumeStatus.MULTIPLE_PEAKS)
            continue
        # Tail threshold: noise-referenced absolute value if derived (H0 calibration),
        # else relative to the peak height of this plume
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
    """Quality assessment and peak search for pollutant windows centered on the co2 peak.

    :param windows: (np.ndarray) Pollutant windows cut around the co2 peak
    :param backgrounds: (np.ndarray) Background value per window
    :param qa_config: (ChannelQAConfig) QA configuration of the pollutant channel
    :param band_before: (int) Samples before the co2 peak where the pollutant peak is accepted
    :param band_after: (int) Samples after the co2 peak where the pollutant peak is accepted
    :param center: (int) Index of the co2 peak inside the window
    :param smooth_window: (int) Savitzky-golay window in samples, odd (converted in find_pollutant_peak)
    :param min_physical_run: Minimum consecutive below-threshold samples to count as faulty
    :return: Tuple (np.ndarray, np.ndarray) of plume statuses and pollutant peak offsets relative to the co2 peak
             (offset = 0 if no valid pollutant peak)
    """
    number_of_plumes = windows.shape[0]
    statuses = np.full(number_of_plumes, PlumeStatus.VALID, dtype=object)
    offsets = np.zeros(number_of_plumes, dtype=int)

    # Guard against below physical value and nan
    statuses[_null_mask(windows, qa_config.min_physical_value, min_physical_run)] = PlumeStatus.NULL_DATA
    valid = statuses == PlumeStatus.VALID

    # AI-Assisted: <Opus 5> ; (Review, Filter Implementation)

    # The Pollutant Channel SNR is lower than the CO2 Channel SNR therefor it is smoothened
    # This Filter was chosen because of its capability of removing noise while preserving features (in bounds)
    # of the signal -> Peak-Position
    # The smoothed signal is only used to find the peaks, it does not replace the original data
    smooth = np.full_like(windows, np.nan)
    smooth[valid] = savgol_filter(windows[valid], window_length=smooth_window,
                                  polyorder=qa_config.smooth_polyorder, axis=1)

    # The pollutant search band needs to be relativ to the co2 peak center
    rel_band_lop = center - band_before
    rel_band_rop = center + band_after

    for plume_idx in np.flatnonzero(valid):
        # Scipy find_peaks is performed on the smoothed signal window
        rel_peak_candidates, props = find_peaks(smooth[plume_idx], prominence=qa_config.min_prominence_floor)
        # Regrading prominence a floor is given so peaks underneath a certain threshold (noise peaks)
        # are not even counted as peaks
        if rel_peak_candidates.size == 0:
            statuses[plume_idx] = PlumeStatus.NO_PEAK
            continue

        # Now it is checked if there is one or more clearly prominent peaks (Has to happen on whole Window not
        # just the small search frame)
        peak_heights = smooth[plume_idx, rel_peak_candidates] - backgrounds[plume_idx]
        main_peak_height = peak_heights.max()
        prom_valid = props["prominences"] >= qa_config.min_prominence_ratio * main_peak_height
        height_valid = peak_heights >= qa_config.min_peak_above_bg
        prominent_peak_candidates = rel_peak_candidates[prom_valid & height_valid]

        if prominent_peak_candidates.size == 0:
            # This plume has no valid peaks
            statuses[plume_idx] = PlumeStatus.NO_PEAK
        elif prominent_peak_candidates.size > 1:
            # This plume has multiple valid peaks -> potential peak overlap
            statuses[plume_idx] = PlumeStatus.MULTIPLE_PEAKS
        else:
            # Valid Peak found -> Pollutant peak found
            # Peak is only valid if it sits in the tight search window around the co2 peak
            peak = prominent_peak_candidates[0]
            if rel_band_lop <= peak <= rel_band_rop:
                offsets[plume_idx] = peak - center
            else:
                statuses[plume_idx] = PlumeStatus.NO_PEAK

    return statuses, offsets
