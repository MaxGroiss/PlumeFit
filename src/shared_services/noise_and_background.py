"""Functions for background and noise estimation

Shared between plume_template_extraction and mode_linear_fitting
"""
# This file contains code created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose/extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Fable 5> ; (Review and Simplification / Catch potential value errors and guard them with named raises)

import numpy as np
import pandas as pd

# Constants
MAD_TO_SIGMA = 1.4826


def faulty_recording_mask(data: np.ndarray, min_physical_value: float, min_run: int = 1) -> np.ndarray:
    """Marks samples that persistently undercut min_physical_value

    The consecutive check prevents noise fluctuating around the min_physical_value line from being flagged as
    a faulty signal.

    :param data: Measurement data
    :param min_physical_value: Values below are counted as potentially faulty samples
    :param min_run: Minimum consecutive below-threshold samples to count as faulty
    :return: Bool mask, True where the data is considered a faulty recording
    """

    below = data < min_physical_value
    # Single Sample Check
    if min_run <= 1:
        return below

    mask = np.zeros_like(below)
    # AI-Flag <Fable 5>: Loop could be avoided by using scipy.ndimage -> binary_opening
    for row_below, row_mask in zip(np.atleast_2d(below), np.atleast_2d(mask)):
        for start, stop in zip(*contiguous_runs(row_below, min_run)):
            row_mask[start:stop] = True
    return mask


def compute_background_series(channel: np.ndarray, percentile: float,
                              rolling_window: int, min_physical_value: float = 0.0, min_physical_run: int = 1) -> np.ndarray:
    """Computes a channels background series using pandas rolling quantiles.

    :param channel: Channel data
    :param percentile: Percentile for the rolling quantile (low value rejects peak influence)
    :param rolling_window: Rolling window size in samples, centered around the current value
    :param min_physical_value: Values below this are counted as faulty and ignored (set to nan)
    :param min_physical_run: Minimum consecutive below-threshold samples to count as faulty
    :return: Background value per sample
    """

    # The Data sometimes has non-physical values like flat zero to avoid these values contaminating the background
    # we set them to nan, pandas will ignore them in the rolling quantile calculation
    masked = channel.astype(float).copy()
    masked[faulty_recording_mask(masked, min_physical_value, min_physical_run)] = np.nan

    # Compute the background series using pandas rolling quantile (for details check pandas documentation)
    # Set Null and Non-Physical values to nan -> Rolling Window with Length rolling_window centered around the
    # currently processed value (important so the bg value belongs to the correct index), min_periods=1 guarantees
    # a bg value even if the window is not full -> quantil returns the value > percentil % of all values
    bg_series = pd.Series(masked).rolling(min_periods=1, center=True,
                                          window=rolling_window).quantile(percentile / 100).to_numpy()

    return bg_series


def contiguous_runs(mask: np.ndarray, min_run: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Finds start/stop indices of contiguous True runs that are at least min_run long.

    :param mask: Bool array
    :param min_run: Minimum run length in samples
    :return: Tuple of start and stop indices (parallel arrays, sorted by pairs, stop exclusive)
    """

    # Convert bool to int8 (true 1, false 0), every index is subtracted with its left value
    # this results in a clear pattern: +1 -> run starts, 0 stays the same, -1 -> run ends
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    # Filter out runs that are too short
    keep = (stops - starts) >= min_run
    return starts[keep], stops[keep]


def influence_mask(n: int, positions: np.ndarray, before: int, after: int) -> np.ndarray:
    """Marks the samples where the signal is considered influenced by a vehicle.

    :param n: Length of the day signal in samples
    :param positions: Sample indices of the vehicle passes
    :param before: Influence window before a pass in samples
    :param after: Influence window after a pass in samples
    :return: Bool mask, True where a vehicle influences the signal
    """
    mask = np.zeros(n, dtype=bool)
    # a window around every pass is set to True -> this sample is influenced by a vehicle
    for p in positions:
        mask[max(0, p - before):min(n, p + after)] = True
    return mask


def estimate_noise(residual: np.ndarray, influenced: np.ndarray) -> tuple[float, float]:
    """Robust noise estimate of the vehicle-free residual signal.

    The standard deviation is estimated via the MAD (median absolute deviation)
    because of its robustness against outliers, MAD_TO_SIGMA is the conversion constant.

    MAD = median(|X_i - median(X)|)

    :param residual: Background subtracted channel signal
    :param influenced: Bool mask of vehicle-influenced samples
    :return: Tuple of (median, sigma) of the vehicle-free residual
    :raises ValueError: If no vehicle-free finite samples are available
    """
    free = residual[~influenced & np.isfinite(residual)]
    if free.size == 0:
        raise ValueError("No vehicle-free samples available for noise estimation")
    med = float(np.median(free))
    sigma = float(MAD_TO_SIGMA * np.median(np.abs(free - med)))
    return med, sigma
