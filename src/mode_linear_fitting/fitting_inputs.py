"""Inputs of the Mode L fitting derived from the extraction results (pipeline data, not configuration).

Contents:
    ShapeTemplate: Unit-area plume shape, one column of the design matrix. Built from
        extracted plumes (from_combined, optionally for a vehicle subgroup) or as a
        composite of several templates for non-separable passes (composite).
    PeakWindow: Initial peak position and refine band of a channel relative to its reference point
        (light barrier trigger for CO₂, refined CO₂ peak of the same pass for pollutants).
The user-facing configuration (ModeLConfig, GroupTemplates, ChannelSetup, PairSetup) is in fitting_config.

"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Assistance with Docstring writing / provenance parameter)
# AI-Assisted: <Opus 5.5> ; (Assistance with optimizing the code structure)

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from src.plume_template_extraction.extraction_result import CombinedResult
from src.plume_template_extraction.normalization import pooled_mean_shape, pooled_median

# Template statistics callable: used so mean calculation of the individual plumes for template generation can be
# swapped easily.
# This needs to be extended if any mean calculation is implemented that needs more parameters
# A new function for mean calculation works directly with the algorithm if it follows the input output structure:
# def xy(unit-area plumes (np.ndarray), original areas of the plumes (np.ndarray), dt (float)) -> unit-area-shape-template (np.ndarray)
ShapeStatistic = Callable[[np.ndarray, np.ndarray, float], np.ndarray]

# For example: The point-wise median function pooled_median needs to be wrapped because by default it only has 2 needed inputs
# It now follows the ShapeStatistics Callable structure

def median_shape(normalized_matrix: np.ndarray, areas: np.ndarray, dt: float) -> np.ndarray:
    """Pointwise median of the unit-area plumes, no area weighting (areas unused, signature of pooled_mean_shape)."""
    return pooled_median(normalized_matrix,dt)


@dataclass(frozen=True)
class ShapeTemplate:
    """Unit-area plume shape s used as basis function of the fit.

    Attributes:
        values: Sampled shape with Δt · Σ values = 1.
        peak_index: Index of the peak in values.
        provenance: Origin, e.g. "empirical:CO2:HGV:n=100" (empirical template of
            channel CO2, subgroup HGV, built from 100 plumes).
    """

    values: np.ndarray
    peak_index: int
    provenance: str

    @classmethod
    def from_combined(cls, cr: CombinedResult, mask: np.ndarray | None = None,
                      label: str = "", shape: ShapeStatistic = pooled_mean_shape) -> ShapeTemplate:
        """Build an empirical template as pooled mean of extracted plumes.

        The mask selects a vehicle subgroup (vehicle class, Euro class, exhaust side);
        it is built outside the pipeline from cr.source_days and cr.pass_indices.

        Args:
            cr: Combined extraction result of one channel.
            mask: Bool per plume of cr. None averages all plumes. Defaults to None.
            label: Subgroup name, stored in the provenance. Defaults to "".
            shape: Template statistic over the plumes, e.g. median_shape. Defaults to pooled_mean_shape.

        Returns:
            Template with the shape of the statistic; a statistic other than the default is named in the provenance
            (e.g. "empirical:BC:median_shape:n=107").

        Raises:
            ValueError: If the mask selects no plume.
        """
        matrix = cr.normalized_matrix if mask is None else cr.normalized_matrix[mask]
        areas = cr.areas if mask is None else cr.areas[mask]
        if matrix.shape[0] == 0:
            raise ValueError(f"mask selects no plumes for channel {cr.channel}")
        # The shape is re-normalized to guarantee a unit area
        values = shape(matrix, areas, cr.dt)
        statistic = "" if shape is pooled_mean_shape else f"{shape.__name__}:"
        provenance = f"empirical:{cr.channel}:{label + ':' if label else ''}{statistic}n={matrix.shape[0]}"
        return cls(values, cr.peak_index, provenance)

    @classmethod
    def composite(cls, members: list[ShapeTemplate], offsets: np.ndarray) -> ShapeTemplate:
        """Combine the templates of non-separable passes into one template.

        Passes too close in time give near-collinear columns. Following Farren et al.
        (2025), the group gets one column instead: the mean of the member templates,
        each shifted to its own position. Fitted against the signal, its amplitude is
        the total area of the group.

        Args:
            members: Templates of the group members.
            offsets: Peak position of every member relative to the composite center
                (offset 0) in samples.

        Returns:
            Composite template spanning all shifted members, unit area if no member
            is truncated.
        """

        #AI-Assisted: <Fable 5> ; (Review of implementation)

        # Calculating resulting length of template
        offsets = np.asarray(offsets, dtype=int)
        lo = min(int(d) - t.peak_index for t, d in zip(members, offsets))
        hi = max(int(d) + t.values.size - t.peak_index for t, d in zip(members, offsets))
        values = np.zeros(hi - lo)
        # Joining member templates
        x = np.arange(lo, hi)
        for t, d in zip(members, offsets):
            values += t.sample(x - d)
        values /= len(members)
        provenances = list(dict.fromkeys(t.provenance for t in members))
        return cls(values, -lo, f"composite(n={len(members)}):" + "|".join(provenances))

    def sample(self, offsets: np.ndarray) -> np.ndarray:
        """Sample the template at offsets relative to its peak.

                Args:
                    offsets: Sample offsets, 0 is the peak in samples.

                Returns:
                    Template values, 0 outside the template support.
                """
        idx = self.peak_index + np.asarray(offsets, dtype=int)
        out = np.zeros(idx.shape, dtype=float)
        in_support = (idx >= 0) & (idx < self.values.size)
        out[in_support] = self.values[idx[in_support]]
        return out


@dataclass(frozen=True)
class PeakWindow:
    """Storage Clas for Peak Window Properties
    Gets the recorded trigger delays from a CombinedResult (LB -> Co2 Peak, Co2 Peak -> Poll Peak)
    Derives the peak search band as the defined quantile range of the delays
    Derives the initial peak positon from the median of the delays

    usage -> from_extraction()
    """

    center: int
    lo: int
    hi: int
    reference: str
    dt: float
    n: int
    @classmethod
    def from_extraction(cls, cr: CombinedResult, quantiles: tuple[float, float] = (5.0, 95.0)) -> PeakWindow:
        trigger = cr.trigger_delays is not None
        d = np.asarray(cr.trigger_delays if trigger else cr.pollutant_offsets, dtype=float)
        d = d[np.isfinite(d)]
        if d.size == 0:
            raise ValueError(f"{cr.channel}: no finite delays for the peak window")
        q_lo, q50, q_hi = np.percentile(d, [quantiles[0], 50.0, quantiles[1]])

        return cls(int(round(q50)), int(np.floor(q_lo)), int(np.ceil(q_hi)),
                   "trigger" if trigger else "co2_peak", cr.dt, d.size)

    @property
    def seconds(self) -> tuple[float, float, float]:
        """(lo, center, hi) in s."""
        return self.lo * self.dt, self.center * self.dt, self.hi * self.dt