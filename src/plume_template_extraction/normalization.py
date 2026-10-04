"""Zeroing, area normalization and pooled mean of extracted plumes.

All functions take plume matrices with one plume per row, cut around the peak and
already background-subtracted with b(t_LB) (see template_extraction.extract_plumes).

Thesis: chapter "The PlumeFit algorithm", section "Normalization and averaging".
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Assistance with Docstring writing)

import warnings

import numpy as np


def zero_baseline_start(plumes: np.ndarray, n_anchor: int) -> np.ndarray:
    """Shift every plume so that the mean of its first n_anchor samples is zero.

        Removes the remaining offset after the background subtraction, including the
        systematic underestimation of the low-percentile background.

        Args:
            plumes: Background-subtracted plumes, one per row.
            n_anchor: Number of samples at the window start used as zero level
                (ExtractionConfig.baseline_anchor in samples).

        Returns:
            Zeroed plumes z_i, same shape as plumes.
        """
    offset = np.nanmean(plumes[:, :n_anchor], axis=1)
    return plumes - offset[:, None]

def area_plausibility_check(plumes: np.ndarray, dt: float, n_anchor: int,
                            peak_index: int, min_width_s: float) -> np.ndarray:
    """Check that every zeroed plume has a positive peak and a plausible area.

    If the zeroing samples sit on the decaying tail of a neighbor, the zero level is
    too high and the own tail is pushed below zero. The net area A then approaches
    zero and the unit-area normalization (z / A) blows the plume up. Requiring a
    minimum effective width A / h catches these plumes before normalization.

    Args:
        plumes: Background-subtracted plumes, one per row.
        dt: Sampling interval [s].
        n_anchor: Zeroing samples, see zero_baseline_start.
        peak_index: Column of the peak.
        min_width_s: Minimum effective width T_w,min [s].

    Returns:
        Bool per plume, True if h > 0 and A ≥ T_w,min · h.
    """
    z = zero_baseline_start(plumes, n_anchor)
    area = np.sum(z, axis=1)*dt
    height = z[:, peak_index]
    return (height > 0) & (area >= min_width_s * height)

def normalize_plumes(plumes: np.ndarray, dt: float, n_anchor: int,
                     channel_name: str = "", day: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Zero every plume and normalize it to unit area.

    Args:
        plumes: Background-subtracted plumes, one per row.
        dt: Sampling interval.
        n_anchor: Zeroing samples, see zero_baseline_start.
        channel_name: Only used in the error message.
        day: Only used in the error message.

    Returns:
        Tuple (normalized, areas):
            normalized: s_i = z_i / A_i, every row with Δt · Σ s_i = 1.
            areas: Plume area A_i = Δt · Σ z_i in channel unit · s, one per row.

    Raises:
        ValueError: If any area is ≤ 0. Cannot happen after area_plausibility_check.
    """
    z = zero_baseline_start(plumes, n_anchor)
    areas = np.sum(z, axis=1)*dt
    if np.any(areas <= 0):
        raise ValueError(f"{channel_name} on {day}: non-positive emission area")
    return z / areas[:, None], areas

def pooled_mean_shape(normalized_matrix: np.ndarray, areas: np.ndarray, dt: float) -> np.ndarray:
    """Area-weighted (pooled) mean of normalized plumes, the template shape.

    Every plume contributes in proportion to its area. The weighting was implemented because small plumes with
    relatively high noise can skew the mean when they are weighted equally.

    Args:
        normalized_matrix: Unit-area plumes s_i, one per row.
        areas: Area A_i per plume, used as weight.
        dt: Sampling interval.

    Returns:
        Template shape with unit area. The weighted mean already has unit area, the
        final renormalization only absorbs rounding errors.
    """
    mean = np.average(normalized_matrix, axis=0, weights=areas)
    return mean / (np.sum(mean) * dt)

def pooled_mean_se(normalized_matrix: np.ndarray, areas: np.ndarray, dt: float,
                   ) -> np.ndarray:
    """Standard error of the pooled mean per sample (Gatz & Smith, 1995).

    Args:
        normalized_matrix: Unit-area plumes s_i, one per row.
        areas: Area A_i per plume, used as weight.
        dt: Sampling interval.

    Returns:
        Standard error per sample of the template, same length as a plume row.
        NaN everywhere for fewer than two plumes.

    Note:
        D. F. Gatz, L. Smith, "The standard error of a weighted mean concentration—I.
        Bootstrapping vs other methods", Atmos. Environ. 29 (1995).
        doi:10.1016/1352-2310(94)00210-C
    """
    #AI-Assisted: <Opus 5> ; (Review Pooled Mean implementation and refine waring print to proper warning message)
    n = areas.size
    if n <= 1:
        warnings.warn(f" Pooled mean standard error calculation on {n} passes. Expected at least n > 1")
        return np.full(normalized_matrix.shape[1], np.nan)
    w = areas[:, None]
    w_b = np.mean(areas)
    x_w = pooled_mean_shape(normalized_matrix, areas, dt)
    wx_dev = w * normalized_matrix - w_b * x_w
    w_dev = w - w_b
    var = n / ((n - 1) * np.sum(areas) ** 2) * (
            np.sum(wx_dev ** 2, axis=0)
            - 2 * x_w * np.sum(w_dev * wx_dev, axis=0)
            + x_w ** 2 * np.sum(w_dev ** 2, axis=0))
    # Rounding can push var slightly below zero where the true value is ≈ 0
    se = np.sqrt(np.maximum(var,0))
    return se
