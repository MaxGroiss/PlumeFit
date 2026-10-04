"""Input structures of the Mode L fitting: shape templates, segments and configuration.

Contents:
    emg: Exponentially modified Gaussian, analytic reference shape for synthetic self-checks.
    ShapeTemplate: Unit-area plume shape, one column of the design matrix. Built from
        extracted plumes (from_combined, optionally for a vehicle subgroup) or as a
        composite of several templates for non-separable passes (composite).
    Segment: One anchor-bounded part of a measurement segment, an independent fit problem.
    ModeLConfig: Parameters of the day fit (windows, baseline rules, refinement, detection).

The composite strategy follows the plume regression of Farren et al. (2025): vehicles
that cannot be separated in time, share one basis function instead of near-collinear
individual columns.

N. J. Farren, M. Knoll, A. Bergmann, R. L. Wagner, M. D. Shaw, S. Wilson, Y. Bernard,
D. C. Carslaw, "Highly Disaggregated Particulate and Gaseous Vehicle Emission Factors and
Ambient Concentration Apportionment Using a Plume Regression Technique",
Environ. Sci. Technol. 59 (23), 11698–11707 (2025). doi:10.1021/acs.est.5c05015
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Assistance with Docstring writing / provenance parameter)


from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.special import erfcx

from src.plume_template_extraction.extraction_result import CombinedResult
from src.plume_template_extraction.normalization import pooled_mean_shape


def emg(t: np.ndarray, mu: float, sigma: float, tau: float) -> np.ndarray:
    """Exponentially modified Gaussian with unit area.

    Gaussian rise convolved with an exponential decay, a common analytic plume shape.
    Evaluated via the scaled complementary error function erfcx, which stays finite
    where the textbook form overflows.

    Args:
        t: Time axis [s].
        mu: Center of the Gaussian component [s].
        sigma: Standard deviation of the Gaussian component [s].
        tau: Exponential decay time constant [s].

    Returns:
        EMG values on t, integral 1.
    """
    z = (sigma / tau - (t - mu) / sigma) / np.sqrt(2)
    gauss = np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    return erfcx(z) * gauss / (2 * tau)


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
                      label: str = "") -> ShapeTemplate:
        """Build an empirical template as pooled mean of extracted plumes.

        The mask selects a vehicle subgroup (vehicle class, Euro class, exhaust side);
        it is built outside the pipeline from cr.source_days and cr.pass_indices.

        Args:
            cr: Combined extraction result of one channel.
            mask: Bool per plume of cr. None averages all plumes. Defaults to None.
            label: Subgroup name, stored in the provenance. Defaults to "".

        Returns:
            Template with the pooled mean shape, see pooled_mean_shape.

        Raises:
            ValueError: If the mask selects no plume.
        """
        matrix = cr.normalized_matrix if mask is None else cr.normalized_matrix[mask]
        areas = cr.areas if mask is None else cr.areas[mask]
        if matrix.shape[0] == 0:
            raise ValueError(f"mask selects no plumes for channel {cr.channel}")
        # The mean shape is re-normalized to guarantee a unit area
        mean = pooled_mean_shape(matrix, areas, cr.dt)
        provenance = f"empirical:{cr.channel}:{label + ':' if label else ''}n={matrix.shape[0]}"
        return cls(mean, cr.peak_index, provenance)

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
class Segment:
    """Part of a measurement segment between two anchor runs, one independent fit problem.

    It contains all passes that may influence each other. An anchor point is a sample
    outside every influence window whose residual lies inside the noise band; an anchor
    run is a run of at least min_anchor_run of them. The bounding anchor runs are part of
    the segment, neighboring segments share them.

    Attributes:
        start: First sample of the segment (start of the left anchor run, or 0).
        stop: Sample after the segment (end of the right anchor run, or n), exclusive.
        pass_ids: Indices into the peak_positions passed to fit_day of the passes in
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

@dataclass(frozen=True)
class ModeLConfig:
    """Parameters of the Mode L day fit.

    Attributes:
        influence_before: Influence window before the expected peak position.
            Default 5 s, typical 3–10 s.
        influence_after: Influence window after the expected peak position.
            Default 15 s, typical 10–25 s.
        min_anchor_run: Minimum duration T_A of an anchor run. Default 5 s.
            Shorter gives more, smaller segments but less reliable anchors.
        t_min_linear: Minimum segment duration for a linear baseline. Default 60 s.
        n_anchor_min: Minimum anchor samples for a fitted baseline, below the baseline is fixed (anchored_fixed).
            Default 10 [samples]; sampling rate dependent.
        merge_vif_max: VIF_max, template columns above are merged with a neighbor.
            Default 10, typical 5–10 (rule of thumb in Fahrmeir et al.).
        noise_band_sigma: Half width k₃ of the anchor noise band in σ. Default 3
        min_physical_value: Values below count as faulty recording (set to NaN), in
            the channel unit. Default 0.0. Channel dependent.
        min_physical_run: Minimum duration below min_physical_value. Default 2 s.
        bg_percentile: Percentile q of the rolling background. Default 2.
        bg_rolling_window: Rolling background window T_RW. Default 100 s. Keep identical to the extraction.
        force_anchored_fixed: Use the fixed baseline in every segment. Default False.
        refine_sweeps: Maximum sweeps of the position refinement, stops early once a
            sweep moves nothing. Default 10, (needs to be high enough to give the methode the chance to converge)
        refine_min_dsse: κ, a shift is accepted only if the SSE drops by more than
            κ · σ̂². Default 8. Smaller lets templates move more easily.
        detect_sigma: k_det, a pass counts as detected if amplitude ≥ k_det · SE. Default 2.
    """

    influence_before: np.timedelta64 = np.timedelta64(5, "s")
    influence_after: np.timedelta64 = np.timedelta64(15, "s")
    min_anchor_run: np.timedelta64 = np.timedelta64(5, "s")

    t_min_linear: np.timedelta64 = np.timedelta64(60, "s")
    n_anchor_min: int = 10

    merge_vif_max: float = 10.0

    noise_band_sigma: float = 3.0

    min_physical_value: float = 0.0
    min_physical_run: np.timedelta64 = np.timedelta64(2, "s")

    bg_percentile: float = 2.0
    bg_rolling_window: np.timedelta64 = np.timedelta64(100, "s")

    force_anchored_fixed: bool = False

    refine_sweeps: int = 10
    refine_min_dsse: float = 8.0
    detect_sigma: float = 2.0