"""Template extraction pipeline: isolated plumes of one segment and one channel (pair).

Flow of extract_plumes:
    1. Isolation: passes whose neighbors are at least min_gap away.
    2. Light barrier window: cutout around the trigger, first CO₂ QA, CO₂ peak search.
    3. Peak window: cutout around the CO₂ peak, second CO₂ QA (multiple peaks, tail).
    4. Area plausibility of the zeroed CO₂ plumes.
    5. Optional pollutant: peak search in a band around the CO₂ peak, cutout around
       the own pollutant peak, area plausibility.
    6. Background subtraction b(t_LB), zeroing and unit-area normalization.

run_batch repeats this for a list of segments and configs.
Thesis: chapter "The PlumeFit algorithm", section "Template extraction".
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Review, Simplification)
# AI-Assisted: <Opus 5> ; (Assistance with Docstring writing / Variable Renaming Suggestions to make the indexing space
#                          more clear in extract_plumes)

from __future__ import annotations

from collections import Counter
import numpy as np

from src.plume_template_extraction.normalization import  normalize_plumes, area_plausibility_check
from src.shared_services.measurement_register import MeasurementRegister
from src.plume_template_extraction.extraction_config import ChannelQAConfig, ExtractionConfig
from src.plume_template_extraction.extraction_result import ExtractionResult, BatchResult
from src.plume_template_extraction.plume_status import PlumeStatus

from src.shared_services.noise_and_background import compute_background_series, influence_mask

from src.plume_template_extraction.extraction_quality import (assess_vectorized_lb_centered,
                                                              assess_iterative_peak_centered,
                                                              assess_pollutant_peak_centered, resolve_qa_thresholds,
                                                              null_data_mask)


def _qa_counts(statuses: np.ndarray) -> dict[PlumeStatus, int]:
    """Count the plumes per status, every PlumeStatus appears (zero if unused).

    Args:
        statuses: PlumeStatus per plume.

    Returns:
        Number of plumes per status.
    """
    counts = Counter(statuses.tolist())
    return {s: counts.get(s, 0) for s in PlumeStatus}


def find_isolated_passes(vehicle_pass_times: np.ndarray, min_gap: np.timedelta64) -> np.ndarray:
    """Find the passes whose previous and next pass are both at least min_gap away.

    The first and last pass of the segment only need one free side.

    Args:
        vehicle_pass_times: Light barrier trigger times, sorted ascending.
        min_gap: Minimum distance to both neighbors.

    Returns:
        Indices into vehicle_pass_times of the isolated passes.
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




def cut_around_peak(channel:np.ndarray, centers:np.ndarray,
                      samples_before:int, samples_after:int) -> tuple[np.ndarray, np.ndarray]:
    """Cut a window [center − samples_before, center + samples_after) around every center.

    Used for light barrier windows (center = trigger) and peak windows (center = peak).
    Windows reaching beyond the signal are skipped.

    Args:
        channel: Channel signal of the segment.
        centers: Sample index of every window center.
        samples_before: Window length before the center in samples.
        samples_after: Window length after the center in samples.

    Returns:
        Tuple (windows, in_bounds):
            windows: One window per row, only for the centers inside the bounds.
            in_bounds: Bool per center, True if its window was cut. Needed to map the
                rows of windows back to centers.
    """

    # Cutout window around indices
    offsets = np.arange(-samples_before,samples_after)
    # Broadcasting: (n,1) + (window,) -> (n, window)
    # Each row contains the absolute sample indices for one plume window
    final_idx = centers[:,None] + offsets
    # Boundary check: window must not exceed channel array bounds
    mask = (centers-samples_before >= 0) & (centers+samples_after <= len(channel))

    return channel[final_idx[mask]],mask



def cutout_isolated_plumes(timestamps: np.ndarray, channel: np.ndarray, vehicle_pass_times: np.ndarray,
                           isolated_passes: np.ndarray, samples_before: int, samples_after: int
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut light barrier windows around the triggers of the isolated passes.

    Args:
        timestamps: Sample times of the segment.
        channel: Channel signal of the segment.
        vehicle_pass_times: Light barrier trigger times of all passes.
        isolated_passes: Indices of the isolated passes, see find_isolated_passes.
        samples_before: Window length before the trigger in samples.
        samples_after: Window length after the trigger in samples.

    Returns:
        Tuple (windows, passes, trigger_idx), row aligned:
            windows: Light barrier window per isolated pass inside the bounds.
            passes: Pass index of every window.
            trigger_idx: Sample index of the trigger of every window.
    """

    # Finds the indices of the isolated passes in the timestamps array: LB to measurement timeseries
    timestamp_indices = np.searchsorted(timestamps, vehicle_pass_times[isolated_passes])
    # Bounds check and window cutting are shared with the peak centered cutout
    plumes, valid_mask = cut_around_peak(channel, timestamp_indices, samples_before, samples_after)

    return plumes, isolated_passes[valid_mask], timestamp_indices[valid_mask]


def find_pollutant_peak(poll_data:np.ndarray, co2_peak_idx:np.ndarray, poll_bg:np.ndarray,
                        config:ExtractionConfig, qa:ChannelQAConfig,
                        dt:float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the pollutant peak belonging to every CO₂ peak.

    Cuts pollutant windows around the CO₂ peaks, runs assess_pollutant_peak_centered
    on them and maps the results back to the order of co2_peak_idx.

    Args:
        poll_data: Pollutant signal of the segment.
        co2_peak_idx: Sample index of the CO₂ peak per plume.
        poll_bg: Pollutant b̂(t_LB) per plume, aligned with co2_peak_idx.
        config: Extraction config (peak window lengths).
        qa: Resolved pollutant QA config.
        dt: Sampling interval [s].

    Returns:
        Tuple (poll_peak_idx, offsets, statuses), aligned with co2_peak_idx:
            poll_peak_idx: Sample index of the pollutant peak (= CO₂ peak if none found).
            offsets: Pollutant peak relative to the CO₂ peak in samples, 0 if none found.
            statuses: PlumeStatus per plume, WINDOW_EDGE if the window was out of bounds.
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
    windows, in_bounds = cut_around_peak(poll_data, co2_peak_idx, window_before_peak, window_after_peak)

    # QA and peak search on the cut windows, offsets are relative to the co2 peak (0 = no pollutant peak found)
    statuses_ib, offsets_ib = assess_pollutant_peak_centered(
        windows, poll_bg[in_bounds], qa,
        band_before_co2_peak, band_after_co2_peak, window_before_peak, smooth_window, min_physical_run)

    # As in_bounds may have cut plumes we remap to the dimensions of the co2 plume index
    statuses = np.full(co2_peak_idx.shape[0], PlumeStatus.WINDOW_EDGE, dtype=object)
    # The statuses that were in bounds during calculation get overridden
    statuses[in_bounds] = statuses_ib
    # The offsets of the pollutant peak from the co2 peak for every plume
    offsets = np.zeros(co2_peak_idx.shape[0], dtype=int)
    offsets[in_bounds] = offsets_ib
    peak_global_idx = co2_peak_idx + offsets

    return peak_global_idx, offsets, statuses


def extract_plumes(register: MeasurementRegister,
                   config: ExtractionConfig) -> list[ExtractionResult]:
    """Extract, check and normalize the isolated plumes of one segment and channel (pair).

    See the module docstring for the order of the steps.

    Args:
        register: Measurement data of one segment.
        config: Channels, windows, background and QA parameters.

    Returns:
        [co2_result] for a CO₂-only config, [co2_result, pollutant_result] otherwise.

    Raises:
        ValueError: If a configured channel is missing in the register, or from the
            threshold calibration (too few vehicle-free samples).
    """

    # AI-Assisted: <Opus 5> ; (Review of indexing)

    # Index spaces used below:
    #   isolated passes  -> rows of co2_lb_windows / passes / trigger_idx
    #   lb_valid         -> indices into those rows that passed the first QA stage
    #   co2_in_bounds    -> bool over lb_valid, True if the peak window could be cut;
    #                       co2_peak_windows only holds the rows where it is True

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

    # Config durations -> samples
    n_before_lb = config.as_samples(config.window_before, dt)
    n_after_lb = config.as_samples(config.window_after, dt)
    n_before_peak = config.as_samples(config.window_before_peak, dt)
    n_after_peak = config.as_samples(config.window_after_peak, dt)
    n_peak_search = config.as_samples(config.peak_search_after, dt)
    n_bg_window = config.as_samples(config.bg_rolling_window, dt)
    n_co2_phys_run = config.as_samples(config.co2_qa.min_physical_run, dt)
    n_anchor = config.as_samples(config.baseline_anchor, dt)


    # 1. Isolation and Light Barrier Windows ---------------------------------------------------------------------------
    # Finds isolated vehicle passes: Returns the indices of passes that are isolated by a minimum gap (int array)
    isolated_passes = find_isolated_passes(register.vehicle_pass_times, config.min_gap)

    # Cuts a window around the isolated passes (Center: Light Barrier Trigger)
    co2_lb_windows, passes, trigger_idx = cutout_isolated_plumes(
        timestamps=register.timestamps, channel=co2_data,
        vehicle_pass_times=register.vehicle_pass_times,
        isolated_passes=isolated_passes,
        samples_before=n_before_lb, samples_after=n_after_lb)

    # Computes a background series using pandas.rolling_quantile
    co2_bg_series = compute_background_series(
        channel=co2_data, percentile=config.bg_percentile,
        rolling_window=n_bg_window, min_physical_value=config.co2_qa.min_physical_value,
        min_physical_run=n_co2_phys_run)
    # The extraction only uses the background value at the moment of the light barrier trigger wich may vary from the
    # background directly before the plume starts to rise Todo: Maybe find the start of the plume -> gradient
    co2_bg = co2_bg_series[trigger_idx]

    # Resolve the noise-derived QA thresholds (min_peak_above_bg / min_prominence_floor = None)
    # The influence mask covers ALL vehicle passes, not only the isolated ones
    all_trigger_idx = np.searchsorted(register.timestamps, register.vehicle_pass_times)
    influenced = influence_mask(len(co2_data), all_trigger_idx,
                                max(n_before_lb, n_before_peak),
                                n_peak_search + n_after_peak)
    co2_qa = resolve_qa_thresholds(config.co2_qa, co2_data - co2_bg_series, influenced,
                                   tail_len=n_after_peak)

    # 2. First CO₂ QA stage and peak search on the light barrier windows -----------------------------------------------
    lb_statuses, trigger_delay = assess_vectorized_lb_centered(
        plumes=co2_lb_windows,
        samples_before=n_before_lb,
        peak_search_window=n_peak_search,
        backgrounds=co2_bg,
        qa_config=co2_qa,
        min_physical_run=n_co2_phys_run)

    # Returns the indices of the valid plumes
    lb_valid = np.flatnonzero(lb_statuses == PlumeStatus.VALID)
    # Trigger index + trigger-to-peak delay = sample index of the CO₂ peak
    co2_peak_idx = trigger_idx[lb_valid] + trigger_delay[lb_valid]

    # 3. Peak windows and second CO₂ QA stage --------------------------------------------------------------------------
    co2_peak_windows, co2_in_bounds = cut_around_peak(co2_data, co2_peak_idx, n_before_peak, n_after_peak)

    # Background aligned with the rows of co2_peak_windows
    co2_bg_peak = co2_bg[lb_valid[co2_in_bounds]]

    # Iterative assessment of plume quality (QA-Checks that are applied on Peak Centered Window)
    peak_statuses = assess_iterative_peak_centered(plumes=co2_peak_windows, backgrounds=co2_bg_peak,
                                                   peak_index=n_before_peak,
                                                   window_after=n_after_peak,
                                                   qa_config=co2_qa,
                                                   min_physical_run=n_co2_phys_run)

    # Merge both QA stages into one status per isolated pass. A peak window can only leave the
    # bounds if it is wider than the light barrier window.
    statuses_final = lb_statuses.copy()
    statuses_final[lb_valid[~co2_in_bounds]] = PlumeStatus.WINDOW_EDGE
    # Statuses from both QA-checks are merged
    statuses_final[lb_valid[co2_in_bounds]] = peak_statuses

    # 4. Area plausibility of the CO₂ plumes (bool over lb_valid) ------------------------------------------------------
    co2_valid = statuses_final[lb_valid] == PlumeStatus.VALID
    area_ok = np.ones_like(co2_valid)  # out-of-bounds plumes are not VALID anyway
    co2_min_width_s = config.co2_qa.min_effective_width / np.timedelta64(1, "s")
    area_ok[co2_in_bounds] = area_plausibility_check(co2_peak_windows - co2_bg_peak[:, None], dt, n_anchor,
                                                     n_before_peak, co2_min_width_s)
    statuses_final[lb_valid[co2_valid & ~area_ok]] = PlumeStatus.NON_PLAUSIBLE_AREA
    co2_valid = co2_valid & area_ok


    # 5. Pollutant channel ---------------------------------------------------------------------------------------------
    poll_result = None
    # poll_channel is None => co2 only run
    if config.poll_channel is None:
        # This is only defined so the output can work with one variable for both cases
        co2_out = co2_valid
    else:
        poll_data = register.get_channel_data_by_name(config.poll_channel)
        # Centering the pollutant window on the CO₂ peak assumed a constant sensor delay. It is not
        # constant, the misaligned plumes distorted the mean shape -> the pollutant peak is searched
        # in a band around the CO₂ peak and the window is cut around the pollutant peak itself.

        poll_bg_series = compute_background_series(
            channel=poll_data, percentile=config.bg_percentile,
            rolling_window=n_bg_window,
            min_physical_value=config.pollutant_qa.min_physical_value,
            min_physical_run=config.as_samples(config.pollutant_qa.min_physical_run, dt)
        )

        # Aligned with lb_valid (keeps the indexing simple, CO₂-invalid rows are masked later)
        poll_bg = poll_bg_series[trigger_idx][lb_valid]

        # Same influence mask as CO₂; no tail calibration, the pollutant QA has no tail check
        poll_qa = resolve_qa_thresholds(config.pollutant_qa, poll_data - poll_bg_series, influenced)

        # The pollutant peak is found in a small band around the co2 peak
        # to avoid complicated indexing the fact that find_pollutant_peak runs on potentially
        # already invalid co2 pollutant pairs is ignored (co2 did not pass iterative qa)
        poll_peak_idx, poll_offsets, poll_statuses = find_pollutant_peak(
            poll_data, co2_peak_idx, poll_bg, config, poll_qa, dt)

        # Pollutant Plumes are cut around their own peak
        poll_peak_windows, poll_in_bounds = cut_around_peak(
            poll_data, poll_peak_idx, n_before_peak, n_after_peak)

        poll_null = np.zeros(poll_statuses.shape[0], dtype=bool)
        poll_null[poll_in_bounds] = null_data_mask(
            poll_peak_windows, config.pollutant_qa.min_physical_value,
            config.as_samples(config.pollutant_qa.min_physical_run, dt))
        # For a pollutant plume to be finally valid it has to have a valid co2 plume, be in bounds (poll_in_bounds)
        # and counted as valid in find_pollutant_peak -> poll_loc_status
        poll_valid = co2_valid & (poll_statuses == PlumeStatus.VALID) & poll_in_bounds

        # poll_peak_windows only holds in-bounds rows, hence poll_valid[poll_in_bounds]
        valid_poll_plumes = poll_peak_windows[poll_valid[poll_in_bounds]] - poll_bg[poll_valid][:, None]
        poll_min_width_s = config.pollutant_qa.min_effective_width / np.timedelta64(1, "s")
        poll_area_ok = area_plausibility_check(valid_poll_plumes, dt, n_anchor, n_before_peak, poll_min_width_s)
        implausible = np.flatnonzero(poll_valid)[~poll_area_ok]
        poll_statuses[implausible] = PlumeStatus.NON_PLAUSIBLE_AREA
        poll_valid[implausible] = False
        valid_poll_plumes = valid_poll_plumes[poll_area_ok]
        # Building Pollutant Output
        norm, areas = normalize_plumes(valid_poll_plumes, dt, n_anchor, channel_name=config.poll_channel, day=source_day)
        poll_result = ExtractionResult(
            channel=config.poll_channel, config=config,
            areas=areas,
            normalized_matrix=norm,
            centered_matrix=valid_poll_plumes,
            peak_index=n_before_peak, dt=dt,
            pass_indices=passes[lb_valid[poll_valid]],
            source_day=source_day, n_isolated=len(isolated_passes), qa_counts=_qa_counts(poll_statuses),
            pollutant_offsets=poll_offsets[poll_valid])

        co2_out = poll_valid if config.drop_co2_invalid_poll else co2_valid

    # 6. CO₂ output (co2_out is over lb_valid, co2_peak_windows only holds in-bounds rows) -----------------------------
    valid_co2_plumes = co2_peak_windows[co2_out[co2_in_bounds]] - co2_bg_peak[co2_out[co2_in_bounds]][:, None]
    norm, areas = normalize_plumes(valid_co2_plumes, dt, n_anchor, channel_name=config.co2_channel, day=source_day)
    co2_result = ExtractionResult(
        channel=config.co2_channel, config=config,
        areas=areas,
        normalized_matrix=norm,
        centered_matrix=valid_co2_plumes,
        peak_index=n_before_peak, dt=dt,
        pass_indices=passes[lb_valid[co2_out]],
        source_day=source_day, n_isolated=len(isolated_passes), qa_counts=_qa_counts(statuses_final),
        pollutant_offsets=None,
        trigger_delays=trigger_delay[lb_valid[co2_out]])

    return [co2_result] if poll_result is None else [co2_result, poll_result]


def run_batch(jobs: list[tuple[MeasurementRegister, list[ExtractionConfig]]]) -> BatchResult:
    """Run extract_plumes for several segments and configs.

    Args:
        jobs: One (register, configs) tuple per segment, typically one register per
            measurement day and one config per channel (pair).

    Returns:
        All ExtractionResults of the run.
    """

    results = []

    # Runs extract_plumes for every day and every channel per day
    # (channel validation happens inside extract_plumes)
    for register, configs in jobs:
        for config in configs:
            results.extend(extract_plumes(register, config))

    return BatchResult(results)

