"""Background and noise estimation shared by template extraction and Mode L fitting.

Both pipeline stages describe the measured concentration signal of one channel as

    x(t) = b(t) + ε(t) + Σ_j p_j(t)

with the slowly drifting background b(t), the measurement noise ε(t) and the plume
p_j(t) of every vehicle pass j. This module estimates b(t) with a rolling low
percentile, estimates the noise level σ of ε(t) robustly via the MAD of the
vehicle-free residual, and provides the sample masks both estimates rely on.

Functions:
    faulty_recording_mask: Samples that persistently undercut a physical minimum (sensor dropouts).
    compute_background_series: Rolling-percentile background estimate b̂(t).
    contiguous_runs: Start/stop indices of True runs in a boolean mask.
    influence_mask: Vehicle-influenced samples, the complement approximates the vehicle-free set T₀.
    estimate_noise: Median and robust standard deviation σ̂ of the vehicle-free residual r_F.

Background: thesis chapter "The PlumeFit algorithm", sections "Rolling percentile
background estimation" and "Robust noise estimation via the MAD".
"""

# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose/extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Assistance with Docstring writing -> mainly latex equation to Unicode symbol equations and
#                          formatting)
# AI-Assisted: <Fable 5> ; (Review and Simplification / Catch potential value errors and guard them with named raises)

import numpy as np
import pandas as pd
from scipy.ndimage import binary_opening

MAD_TO_SIGMA = 1.4826
#Consistency constant c_MAD = 1 / Φ⁻¹(3/4) ≈ 1.4826.
#Turns the MAD into an estimation of the standard deviation (σ̂ = c_MAD · MAD) under the assumption of normally
#distributed noise.


def faulty_recording_mask(data: np.ndarray, min_physical_value: float, min_physical_run: int = 1) -> np.ndarray:
    """Marks samples that stay below a physical minimum for a minimum duration

    Sensor dropouts cause flat runs below the usual background / measurement minimum. Min physical run is required to
    keep samples that drop below the chosen minimum value due to noise.

    Args:
        data : Channel signal. A 2-D input checked row by row (vectorized with binary opening) -> One plume window per row.
        min_physical_value : Lowest physically plausible value of the channel including background.
            Channel dependent, set slightly below the lowes expected ambient level.
        min_physical_run : Minimum number of consecutive below-threshold samples to count as faulty

    Returns:
        Bool mask, same shape as data. True where the sample belongs to a faulty recording
    """

    below = data < min_physical_value
    # Single Sample Check
    if min_physical_run <= 1:
        return below
    # AI-Assisted: <Opus 5> ; (vectorized fia binary opening)
    # Below flags all indexes with samples below min_physical value True, structure tells binary opening to only
    # keep True runs that are at least min_physical_run long -> 2D Array structure -> [[True,True,True]] -> 1D [True,True,True]
    structure = np.ones((1,) * (below.ndim - 1) + (min_physical_run,), dtype=bool)
    return binary_opening(below, structure=structure)


def compute_background_series(channel: np.ndarray, percentile: float,
                              rolling_window: int, min_physical_value: float = 0.0, min_physical_run: int = 1) -> np.ndarray:
    """Estimate the background b(t) of a channel with a centered rolling percentile.

    A low percentile of a window that is much longer than a plume follows the slowly drifting background but ignores
    plumes, as long as a small fraction of every window is vehicle free.

    Args:
        channel: Channel signal of one measurement segment.
        percentile: Percentile q of the rolling window. The estimate stays plume free while
            at least q % of every window is vehicle free.
        rolling_window: Rolling  window length, centered on the current sample. Needs to be much longer than a plume
            window and shorter than the background drift.
        min_physical_value: Threshold of faulty_recording_mask.
        min_physical_run: Minimum run of faulty_recording_mask.

    Returns:
        Background estimate b(t). NaN only where the whole window consists of faulty or NaN samples.
    """

    # Faulty recordings would drag low percentile down -> set to nan because pandas rolling skips those values.
    masked = channel.astype(float).copy()
    masked[faulty_recording_mask(masked, min_physical_value, min_physical_run)] = np.nan

    # center = True assigns the window result to its middle sample this is needed so the estimation of b is not shifted
    # in time.
    bg_series = pd.Series(masked).rolling(min_periods=1, center=True,
                                          window=rolling_window).quantile(percentile / 100).to_numpy()

    return bg_series


def contiguous_runs(mask: np.ndarray, min_run: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Find contiguous True runs of a boolean mask that are at least min_run long.

    Args:
        mask: 1-D bool mask.
        min_run: Minimum run length, shorter runs are dropped.

    Returns:
        Tuple (starts, stops) of parallel int arrays.
    """
    # As type int -> bool to 1/0 -> true falls only provides that the signal has changed
    # with int -> +1 Rise from False to True / -1 Fall from True to Falls -> Edges
    # np_r ads a 0 to the start and end of mask -> mask = 110 without the padding 0 at the start there is no start
    # -> rise from 0 to 1 same for the end so it is basically guaranteed that every run has a start and an end
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0]) # Array with +1 and - 1 (and zeros) indicates where a change
    # occurred and if it was a rise (+1) or a fall (-1) (0 no change)
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) # Returns the indices where a rise
    # (starts) and a fall (stops) occurred
    # The Diff of start and stop returns the length of every run and keeps is a boolean mask for all runs >= min_run
    keep = (stops - starts) >= min_run
    return starts[keep], stops[keep]


def influence_mask(n: int, positions: np.ndarray, before: int, after: int) -> np.ndarray:
    """Marks the samples where the signal is considered influenced by a vehicle pass.

    Args:
        n: Length of the segment signal.
        positions: Sample index per vehicle pass. Extraction -> LB-Trigger, Fitting -> Expected Peak Position
        before: Influence window before each position.
        after: Influence window after each position.

    Returns:
        Bool mask. True inside [p − before, p + after] of any position
        p, clipped to the segment.
    """

    mask = np.zeros(n, dtype=bool)
    for p in positions:
        mask[max(0, p - before):min(n, p + after)] = True
    return mask


def estimate_noise(residual: np.ndarray, influenced: np.ndarray) -> tuple[float, float]:
    """Robust median and standard deviation of the vehicle-free residual r_F.

    The standard deviation is derived from the MAD instead of the sample standard
    deviation, because the vehicle-free residual still contains outliers (plume
    tails, unregistered sources, see assumption on the noise definition).

    Args:
        residual: Background-subtracted signal r(t) = x(t) − b(t).
            NaN samples are ignored.
        influenced: Vehicle-influenced samples, see influence_mask.

    Returns:
        Tuple (median, sigma):
            median: median(r_F). Level of the quiet signal.
            sigma: Robust noise standard deviation σ̂ = c_MAD · MAD(r_F).

    Raises:
        ValueError: If no finite vehicle-free sample is available.

    Note:
        MAD = median(|r_F − median(r_F)|), σ̂ = 1.4826 · MAD (see MAD_TO_SIGMA).
    """
    free = residual[~influenced & np.isfinite(residual)]
    if free.size == 0:
        raise ValueError("No vehicle-free samples available for noise estimation")
    med = float(np.median(free))
    sigma = float(MAD_TO_SIGMA * np.median(np.abs(free - med)))
    return med, sigma
