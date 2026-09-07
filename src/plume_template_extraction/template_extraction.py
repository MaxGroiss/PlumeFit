"""This module contains the main functionality of the plume template extraction pipeline:
Isolation of vehicle passes -> window cutting -> calls into the QA -> Normalization -> ExtractionResults

It also contains the batch wrapper for template extraction run_batch

"""
# This file contains code created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Review, Simplification)

from __future__ import annotations

from collections import Counter
import numpy as np

from src.shared_services.measurement_register import MeasurementRegister
from src.plume_template_extraction.extraction_config import ChannelQAConfig, ExtractionConfig
from src.plume_template_extraction.extraction_result import ExtractionResult, BatchResult
from src.plume_template_extraction.plume_status import PlumeStatus

from src.shared_services.noise_and_background import compute_background_series, influence_mask

from src.plume_template_extraction.extraction_quality import (assess_vectorized_lb_centered,
     assess_iterative_peak_centered, assess_pollutant_peak_centered, resolve_qa_thresholds)


def _qa_counts(statuses: np.ndarray) -> dict[PlumeStatus, int]:
    """Counts the plumes per QA status, zero-filled so every status appears.

    :param statuses: (np.ndarray) Array of PlumeStatus per plume
    :return: dict(PlumeStatus, int) with key = status and value = number of plumes
    """
    counts = Counter(statuses.tolist())
    return {s: counts.get(s, 0) for s in PlumeStatus}


def find_isolated_passes(vehicle_pass_times: np.ndarray, min_gap: np.timedelta64) -> np.ndarray:
    """Finds isolated vehicle passes based on minimum gap between passes.

    :param vehicle_pass_times: (np.ndarray) Detected vehicle passes by a light barrier
    :param min_gap: (np.timedelta64) Timedelta between passes to be considered isolated
    :return: (np.ndarray) Indices of isolated passes
    """

    # Returns the time difference between consecutive passes
    diffs = np.diff(vehicle_pass_times)

    # Creates two Bool Arrays: Is the left neighbor min_gap away ? | Is the right neighbor min_gap away ?
    gap_before = np.concatenate([[True], diffs >= min_gap]) # Before Condition for first pass is assumed True
    gap_after = np.concatenate([diffs >= min_gap, [True]]) # After Condition for last pass is assumed True

    # Only True if both neighbors are min_gap away
    mask = gap_before & gap_after

    # Returns the indices of True values -> Indices of isolated passes
    return np.flatnonzero(mask)

def zero_baseline_start(centered_normalized_matrix: np.ndarray, n_start: int = 3) -> np.ndarray:
    """Shift each plume so its leading background sits at zero.

    Subtracts a per-plume offset (median of the first n_start samples)
    from the whole plume.

    :param centered_normalized_matrix: (np.ndarray) Normalized plumes.
    :param n_start: (int) Samples for offset calculation should be a small value.
    :returns: (np.ndarray) Shifted plume matrix
    """
    offset = np.nanmedian(centered_normalized_matrix[:, :n_start], axis=1)
    return centered_normalized_matrix - offset[:, None]


def normalize_area(centered_matrix: np.ndarray, dt: float) -> np.ndarray:
    """Scale each plume so its integral (sum × dt) equals 1.

    :param centered_matrix: (np.ndarray) Centered plumes.
    :param dt: (float) Sampling interval in seconds.
    :returns: (np.ndarray) Area-normalized plume matrix.
    :raises ValueError: If any plume has zero area.
    """
    areas = (np.sum(centered_matrix, axis=1, keepdims=True) * dt)
    if np.any(areas <= 0):
        raise ValueError("Zero Emission area detected")

    return centered_matrix / areas


def cut_around_peak(channel:np.ndarray, peak_global:np.ndarray,
                      samples_before:int, samples_after:int) -> tuple[np.ndarray, np.ndarray]:
    """Cuts a window with samples_before and samples_after around the peak.

    :param channel: (np.ndarray) Data of the Channel in interest
    :param peak_global: (np.ndarray) Peaks of the previously isolated and  s1 qa checked  plumes
    :param samples_before: (int) Window constraints left
    :param samples_after: (int) Window constraints right
    :return: Tuple (np.ndarray, np.ndarray) of peak centered plumes, cutout mask
    """

    # Cutout window around indices
    offsets = np.arange(-samples_before,samples_after)
    # Broadcasting: (n,1) + (window,) -> (n, window)
    # Each row contains the absolute sample indices for one plume window
    final_idx = peak_global[:,None] + offsets
    # Boundary check: window must not exceed channel array bounds
    mask = (peak_global-samples_before >= 0) & (peak_global+samples_after <= len(channel))

    return channel[final_idx[mask]],mask



def cutout_isolated_plumes(timestamps: np.ndarray, channel: np.ndarray, vehicle_pass_times: np.ndarray,
                           isolated_passes: np.ndarray, samples_before: int, samples_after: int
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cuts a window around the isolated passes.

    :param timestamps: (np.ndarray) Timeseries
    :param channel: (np.ndarray) Data of the Channel in interest
    :param vehicle_pass_times: (np.ndarray) Light barrier pass times
    :param isolated_passes: (np.ndarray) Indices that survived the isolation process
    :param samples_before: (int) Window constraints left
    :param samples_after: (int) Window constraints right
    :return: Tuple (np.ndarray, np.ndarray, np.ndarray) of cutout plumes, valid_passes, ts_valid_indices
    """

    # Finds the indices of the isolated passes in the timestamps array: LB to measurement timeseries
    timestamp_indices = np.searchsorted(timestamps, vehicle_pass_times[isolated_passes])
    # Bounds check and window cutting are shared with the peak centered cutout
    plumes, valid_mask = cut_around_peak(channel, timestamp_indices, samples_before, samples_after)

    return plumes, isolated_passes[valid_mask], timestamp_indices[valid_mask]


def find_pollutant_peak(poll_data:np.ndarray, co2_peak_global_idx:np.ndarray, poll_bg:np.ndarray,
                        config:ExtractionConfig, qa:ChannelQAConfig,
                        dt:float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cuts pollutant windows around the co2 peaks and runs the pollutant QA on them.

    All QA criteria live in plume_quality.assess_pollutant_peak_centered — this function
    only cuts the windows and maps the results back onto the co2 index dimensions.

    :param poll_data: Measurement data of the pollutant channel
    :param co2_peak_global_idx: Global sample index of the co2 peak per plume
    :param poll_bg: Background value per plume (all plumes that passed the LB centered qa)
    :param config: Extraction configuration (window parameters are used)
    :param qa: Resolved pollutant QA configuration (see plume_quality.resolve_qa_thresholds)
    :param dt: Sampling interval in seconds
    :return: Tuple of global pollutant peak indices, offsets to the co2 peak, statuses
    """

    # AI-Assisted: <Opus 5> ; (Review of indexing)

    # Convert Time Windows in Sample Windows
    band_before_co2_peak = config.as_samples(qa.band_before_co2_peak, dt)
    band_after_co2_peak = config.as_samples(qa.band_after_co2_peak, dt)

    window_before_peak = config.as_samples(config.window_before_peak, dt)
    window_after_peak = config.as_samples(config.window_after_peak, dt)

    min_physical_run = config.as_samples(qa.min_physical_run, dt)

    # Savgol window: time -> samples; scipy requires an odd window larger than the polyorder
    smooth_window = max(config.as_samples(qa.smooth_window, dt), qa.smooth_polyorder + 1)
    if smooth_window % 2 == 0:
        smooth_window += 1

    # To perform operations on the relevant pollution windows they are cut
    # At this point all windows should be in bound (Corresponding Co2 Windows are) the guard is still applied
    windows, in_bounds = cut_around_peak(poll_data, co2_peak_global_idx, window_before_peak, window_after_peak)

    # QA and peak search on the cut windows, offsets are relative to the co2 peak (0 = no pollutant peak found)
    statuses_ib, offsets_ib = assess_pollutant_peak_centered(
        windows, poll_bg[in_bounds], qa,
        band_before_co2_peak, band_after_co2_peak, window_before_peak, smooth_window, min_physical_run)

    # As in_bounds may have cut plumes we remap to the dimensions of the co2 plume index
    statuses = np.full(co2_peak_global_idx.shape[0], PlumeStatus.WINDOW_EDGE, dtype=object)
    # The statuses that were in bounds during calculation get overridden
    statuses[in_bounds] = statuses_ib
    # The offsets of the pollutant peak from the co2 peak for every plume
    offsets = np.zeros(co2_peak_global_idx.shape[0], dtype=int)
    offsets[in_bounds] = offsets_ib
    peak_global_idx = co2_peak_global_idx + offsets

    return peak_global_idx, offsets, statuses


def extract_plumes(register: MeasurementRegister,
                   config: ExtractionConfig) -> list[ExtractionResult]:
    """Main plume extraction pipeline for one day and one channel (pair).

    Isolation -> LB centered cutout -> vectorized QA -> peak centered cutout -> iterative QA
    -> optional pollutant peak search + QA around the co2 peak -> normalization.

    :param register: MeasurementRegister of one measurement day
    :param config: Extraction configuration (channels, windows, background, QA)
    :return: List of ExtractionResults: [co2] or [co2, pollutant]
    """

    # AI-Assisted: <Opus 5> ; (Review of indexing)

    #Input Validation
    if config.co2_channel not in register.channel_names:
        raise ValueError(f"CO2 channel '{config.co2_channel}' not in register")
    # Every Pollutant Channel needs its corresponding Co2 Channel for extraction
    if config.poll_channel is not None and config.poll_channel not in register.channel_names:
        raise ValueError(f"Pollutant channel '{config.poll_channel}' not in register")

    # Data Preparation
    co2_data = register.get_channel_data_by_name(config.co2_channel)
    dt = register.dt
    source_day = register.source_day

    # Cutout Parameters
    window_before_lb = config.as_samples(config.window_before, dt)
    window_after_lb = config.as_samples(config.window_after, dt)

    window_before_peak = config.as_samples(config.window_before_peak, dt)
    window_after_peak = config.as_samples(config.window_after_peak, dt)

    peak_search_window = config.as_samples(config.peak_search_after, dt)
    bg_rolling_window = config.as_samples(config.bg_rolling_window, dt)

    co2_phys_run = config.as_samples(config.co2_qa.min_physical_run, dt)

    # Finds isolated vehicle passes: Returns the indices of passes that are isolated by a minimum gap (int array)
    isolated_passes = find_isolated_passes(register.vehicle_pass_times, config.min_gap)

    # Cuts a window around the isolated passes (Center: Light Barrier Trigger)
    co2_sw_plumes, valid_passes, ts_valid_idx = cutout_isolated_plumes(
        timestamps=register.timestamps, channel=co2_data,
        vehicle_pass_times=register.vehicle_pass_times,
        isolated_passes=isolated_passes,
        samples_before=window_before_lb, samples_after=window_after_lb)

    # Computes a background series using pandas.rolling_quantile
    co2_bg_series = compute_background_series(
        channel=co2_data, percentile=config.bg_percentile,
        rolling_window=bg_rolling_window, min_physical_value=config.co2_qa.min_physical_value,
        min_physical_run=co2_phys_run)
    # The extraction only uses the background value at the moment of the light barrier trigger
    # It is assumed that rising background in a plume window is caused by the passing vehicle itself
    # TODO:This could be optimized by finding the actual start of the plume (the real background probably doesn't drift much between those timestamps)
    co2_bg = co2_bg_series[ts_valid_idx]

    # Resolve the noise-derived QA thresholds (min_peak_above_bg / min_prominence_floor = None)
    # The influence mask covers ALL vehicle passes, not only the isolated ones
    all_pass_idx = np.searchsorted(register.timestamps, register.vehicle_pass_times)
    influenced = influence_mask(len(co2_data), all_pass_idx, window_before_lb, window_after_lb)
    # tail_len = tail of the peak centered window, needed for the H0 tail calibration
    co2_qa = resolve_qa_thresholds(config.co2_qa, co2_data - co2_bg_series, influenced,
                                   tail_len=window_after_peak)

    # Vectorized assessment of plume quality (QA-Checks that can be applied on LB Centered Window and don't need a loop)
    # Returns the statuses of the plumes and the peak index relative to samples before the light barrier trigger
    statuses_lbc_v, relative_peak_idx = assess_vectorized_lb_centered(
        plumes=co2_sw_plumes,
        samples_before=window_before_lb,
        peak_search_window=peak_search_window,
        backgrounds=co2_bg,
        qa_config=co2_qa,
        min_physical_run=co2_phys_run)

    # Returns the indices of the valid plumes
    valid_lbc_plumes_idx = np.flatnonzero(statuses_lbc_v == PlumeStatus.VALID)
    # Peak index is still relativ, needs to be global for cutout
    # Gets plume idx LB in global and then moves relativ to peak
    peak_global_idx = ts_valid_idx[valid_lbc_plumes_idx] + relative_peak_idx[valid_lbc_plumes_idx]

    # Cuts a window around the isolated passes (Center: Peak of isolated Plume)
    # It is important to know that co2_plumes as the return value only holds the in bound plumes
    # that is the reason why co2_in_bounds is needed for further indexing
    co2_plumes, co2_in_bounds = cut_around_peak(co2_data, peak_global_idx, window_before_peak, window_after_peak)

    # Bringing the background array to the same size and correct indexes
    # co2 background -> valid lbc plumes that the co2 peak cutout did not drop with its mask
    co2_bg_f = co2_bg[valid_lbc_plumes_idx[co2_in_bounds]]

    # Iterative assessment of plume quality (QA-Checks that are applied on Peak Centered Window)
    statuses_pc_i =assess_iterative_peak_centered(plumes=co2_plumes, backgrounds=co2_bg_f,
                                                  peak_index=window_before_peak,
                                                  window_after=window_after_peak,
                                                  qa_config=co2_qa)

    # Peak Centered Cutout Boundaries may have dropped plumes that are still valid in valid_lbc_plumes_idx
    # This only applies if the peak centered cutout is greater than the initial light barrier cutout window
    # These dropped plumes are marked and then the two qa-checks are merged
    # All Valid Plumes outside the cut mask are marked as WINDOW_EDGE
    statuses_final = statuses_lbc_v.copy()
    statuses_final[valid_lbc_plumes_idx[~co2_in_bounds]] = PlumeStatus.WINDOW_EDGE
    # Statuses from both QA-checks are merged
    statuses_final[valid_lbc_plumes_idx[co2_in_bounds]] = statuses_pc_i


    #-------------------Pollutant-------------------

    poll_result = None
    co2_valid = statuses_final[valid_lbc_plumes_idx] == PlumeStatus.VALID

    # poll_channel is None => co2 only run
    if config.poll_channel is None:
        # This is only defined so the output can work with one variable for both cases
        co2_out = co2_valid
    else:
        # Data Preparation
        poll_data = register.get_channel_data_by_name(config.poll_channel)
        # The co2->pollutant offset was assumed constant per sensor coupling, so centering the pollutant
        # window on the co2 peak should keep plumes aligned. THIS PROVED WRONG: overlapping those plumes
        # distorts the mean shape -> the pollutant peak is searched in a small band around the co2 peak.

        # Only the background for pollutant plumes with a valid co2 plume is needed
        poll_bg_series = compute_background_series(
            channel=poll_data, percentile=config.bg_percentile,
            rolling_window=bg_rolling_window,
            min_physical_value=config.pollutant_qa.min_physical_value,
            min_physical_run=config.as_samples(config.pollutant_qa.min_physical_run,dt)
            )
        # To avoid complicated indexing the background for all plumes that passed the light barrier centered
        # qa are passed
        poll_bg = poll_bg_series[ts_valid_idx][valid_lbc_plumes_idx]

        # Resolve the noise-derived QA thresholds for the pollutant channel (same influence mask)
        poll_qa = resolve_qa_thresholds(config.pollutant_qa, poll_data - poll_bg_series, influenced)

        # The pollutant peak is found in a small band around the co2 peak
        # to avoid complicated indexing the fact that find_pollutant_peak runs on potentially
        # already invalid co2 pollutant pairs is ignored (co2 did not pass iterative qa)
        poll_peak_idx, poll_offsets, poll_loc_status = find_pollutant_peak(
            poll_data, peak_global_idx, poll_bg, config, poll_qa, dt)

        # Pollutant Plumes are cut around their own peak
        poll_plumes, poll_in_bounds = cut_around_peak(
            poll_data, poll_peak_idx, window_before_peak, window_after_peak)

        # For a pollutant plume to be finally valid it has to have a valid co2 plume, be in bounds (poll_in_bounds)
        # and counted as valid in find_pollutant_peak -> poll_loc_status
        combined_poll_mask = co2_valid & (poll_loc_status == PlumeStatus.VALID) & poll_in_bounds

        # poll_in_bounds is needed because cut_around_peak only returns the in bound plumes
        valid_poll_plumes = poll_plumes[combined_poll_mask[poll_in_bounds]] - poll_bg[combined_poll_mask][:, None]

        # Building Pollutant Output
        poll_result = ExtractionResult(
            channel=config.poll_channel, config=config,
            normalized_matrix=zero_baseline_start(normalize_area(valid_poll_plumes, dt)), # -> Normalize
            centered_matrix=valid_poll_plumes,
            peak_index=window_before_peak, dt=dt,
            pass_indices=valid_passes[valid_lbc_plumes_idx[combined_poll_mask]],
            source_day=source_day, n_isolated=len(isolated_passes), qa_counts=_qa_counts(poll_loc_status),
            pollutant_offsets=poll_offsets[combined_poll_mask])

        # This step decides if only co2 plumes are kept that have a valid pollutant plume assigned to them
        co2_out = combined_poll_mask if config.drop_co2_invalid_poll else co2_valid

    # Building CO2 Output
    # in bounds still needed to not run into out of bounds errors
    valid_co2_plumes = co2_plumes[co2_out[co2_in_bounds]] - co2_bg_f[co2_out[co2_in_bounds]][:, None]

    co2_result = ExtractionResult(
        channel=config.co2_channel, config=config,
        normalized_matrix=zero_baseline_start(normalize_area(valid_co2_plumes, dt)),
        centered_matrix=valid_co2_plumes,
        peak_index=window_before_peak, dt=dt,
        pass_indices=valid_passes[valid_lbc_plumes_idx[co2_out]],
        source_day=source_day, n_isolated=len(isolated_passes), qa_counts=_qa_counts(statuses_final),
        pollutant_offsets=None,
        trigger_delays=relative_peak_idx[valid_lbc_plumes_idx[co2_out]])

    return [co2_result] if poll_result is None else [co2_result, poll_result]


def run_batch(jobs: list[tuple[MeasurementRegister, list[ExtractionConfig]]]) -> BatchResult:
    """Run plume extraction for a list of jobs.
    A job is a tuple of a MeasurementRegister and a list of ExtractionConfig.
    Meaning a MeasurementRegister per day and an ExtractionConfig per channel.

    :param jobs: list[tuple[MeasurementRegister, list[ExtractionConfig]]]
    :return: (BatchResult) Batch Result DataClass containing the extraction results
    """

    results = []

    # Runs extract_plumes for every day and every channel per day
    # (channel validation happens inside extract_plumes)
    for register, configs in jobs:
        for config in configs:
            results.extend(extract_plumes(register, config))

    return BatchResult(results)

