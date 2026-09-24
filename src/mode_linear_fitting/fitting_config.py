"""Input structures of the Mode L fitting pipeline: shape templates, segments and configuration.

Contents:
    emg            Exponentially modified Gaussian (numerically stable via erfcx); analytic
                   reference shape, used by the synthetic self-checks
    ShapeTemplate  Unit-area plume shape used as basis function of the fit. Built empirically
                   from the plume-template-extraction stage (from_combined, optionally masked
                   to a vehicle subgroup) or as a Farren-like composite of several templates
                   at fixed offsets (composite) for declared inseparable vehicle clusters
    Segment        One anchor-bounded slice of a measurement day — the independent fit unit
    ModeLConfig    All tuning knobs of the day fitting (windows, baseline rules, refine,
                   detection)

The composite strategy is inspired by the plume regression technique of Farren et al. [1]:
vehicles that cannot be separated on the time axis are represented by one shared basis
function instead of near-collinear individual columns.

Citation:
Naomi J. Farren, Markus Knoll, Alexander Bergmann, Rebecca L. Wagner, Marvin D. Shaw,
Samuel Wilson, Yoann Bernard, David C. Carslaw; Highly Disaggregated Particulate and
Gaseous Vehicle Emission Factors and Ambient Concentration Apportionment Using a Plume
Regression Technique. Environ. Sci. Technol. 17 June 2025; 59 (23): 11698-11707.
https://doi.org/10.1021/acs.est.5c05015
"""
# This file contains code created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Docstrings)

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.special import erfcx

from src.plume_template_extraction.extraction_result import CombinedResult


def emg(t: np.ndarray, mu: float, sigma: float, tau: float) -> np.ndarray:
    """Exponentially modified Gaussian (EMG), numerically stable via erfcx.

    :param t: Time axis
    :param mu: Mean of the Gaussian component
    :param sigma: Standard deviation of the Gaussian component
    :param tau: Exponential decay time
    :return: EMG values sampled on t
    """
    z = (sigma / tau - (t - mu) / sigma) / np.sqrt(2)
    gauss = np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    return erfcx(z) * gauss / (2 * tau)


@dataclass(frozen=True)
class ShapeTemplate:
    """Plume shape template used as basis function for the Mode L fit.

    Attributes:
        values: (np.ndarray) Sampled template values (unit area)
        peak_index: (int) Index of the peak inside values
        provenance: (str) Origin of the template, e.g. "empirical:channel:n=100"
                    meaning empirical template, channel xy, built from 100 passes
    """

    values: np.ndarray
    peak_index: int
    provenance: str

    @classmethod
    def from_combined(cls, cr: CombinedResult, mask: np.ndarray | None = None,
                      label: str = "") -> ShapeTemplate:
        """Builds an empirical template from the unit-area mean shape of a CombinedResult.

        With a mask only the selected plume rows are averaged. This allows creating Templates for
        specific vehicle groups (fuel type, euro, vehicle type) out of one CombinedResult.
        Creating the mask is not part of the fitting pipeline. Specific vehicles can be derived by fetching
        pass_indices and source_days out of CombinedResult.

        :param cr: (CombinedResult) Combined Extraction Result
        :param mask: (np.ndarray) Boolean Mask, None Averages all plumes
        :param label: (str) Optional subgroup label (carried trough pipeline provenance)
        :return: (ShapeTemplate) ShapeTemplate with the mean shape normalized to unit area
        :raises ValueError: If the mask selects no plumes
        """
        matrix = cr.normalized_matrix if mask is None else cr.normalized_matrix[mask]
        if matrix.shape[0] == 0:
            raise ValueError(f"mask selects no plumes for channel {cr.channel}")
        # The mean shape is re-normalized to guarantee a unit area
        mean = np.mean(matrix, axis=0)
        mean = mean / (np.sum(mean) * cr.dt)
        provenance = f"empirical:{cr.channel}:{label + ':' if label else ''}n={matrix.shape[0]}"
        return cls(mean, cr.peak_index, provenance)

    @classmethod
    def composite(cls, members: list[ShapeTemplate], offsets: np.ndarray) -> ShapeTemplate:
        """ Some vehicle passes are to close to be separated even trough fitting.
        Templates that almost collide on the time axis caused by tight traffic cause collinearities in the design
        matrix. By using a Farren et al. [1] like approach mean of the member templates, each shifted
        to its own offset relative to the composite center (offset = 0) these inseparable groups still yield results.

        :param members: (list[ShapeTemplate]) Templates of the group members
        :param offsets: (np.ndarray) Sample offset per member relative to the composite center
        :return: (ShapeTemplate) spanning the union of the shifted member supports
        """

        #AI-Assisted: <Fable 5> ; (Review of implementation)

        # Calculating resulting length of template
        offsets = np.asarray(offsets, dtype=int)
        lo = min(int(d) - t.peak_index for t, d in zip(members, offsets))
        hi = max(int(d) + t.values.size - t.peak_index for t, d in zip(members, offsets))
        values = np.zeros(hi - lo)
        x = np.arange(lo, hi)
        # Joining member templates
        for t, d in zip(members, offsets):
            values += t.sample(x - d)
        values /= len(members)
        provenances = list(dict.fromkeys(t.provenance for t in members))
        return cls(values, -lo, f"composite(n={len(members)}):" + "|".join(provenances))

    def sample(self, offsets: np.ndarray) -> np.ndarray:
        """Samples the template at the given offsets, the template peak sits at offset = 0.

        :param offsets: Sample offsets relative to the peak
        :return: Template values at the offsets, 0 outside the template support
        """
        idx = self.peak_index + np.asarray(offsets, dtype=int)
        out = np.zeros(idx.shape, dtype=float)
        in_support = (idx >= 0) & (idx < self.values.size)
        out[in_support] = self.values[idx[in_support]]
        return out


@dataclass(frozen=True)
class Segment:
    """One segment of a measurement day, bounded by anchor runs.

    The day fit is split into independent problems: a segment contains all vehicle passes
    that may influence each other, while its start and end lie in areas free of vehicle
    influence. An anchor point is a sample estimated to be vehicle free (quiet signal
    inside the noise band); an anchor run is a sufficiently long run of consecutive
    anchor points.

    Attributes:
        start: (int) Start sample index of the segment
        stop: (int) Stop sample index of the segment
        pass_ids: (np.ndarray) IDs of the vehicles in the segment ordered by appearance
        anchor_mask: (np.ndarray) Samples considered an anchor point in the segment
        anchored_left: (bool) Segment is bounded by a clear anchor run on the left
        anchored_right: (bool) Segment is bounded by a clear anchor run on the right
    """

    start: int
    stop: int
    pass_ids: np.ndarray
    anchor_mask: np.ndarray
    anchored_left: bool
    anchored_right: bool

@dataclass(frozen=True)
class ModeLConfig:
    """Configuration of the Mode L day fitting.

    Attributes:
        influence_before: (timedelta64(x, "s")) Window before a pass where the signal is considered
                          influenced by the vehicle
        influence_after: (timedelta64(x, "s")) Window after a pass where the signal is considered
                         influenced by the vehicle
        min_anchor_run: (timedelta64(x, "s")) Minimum duration of vehicle absence for a quiet signal
                        to be considered an anchor
        t_min_linear: (timedelta64(x, "s")) Minimum segment span for a linear baseline fit
        n_anchor_min: (int) Minimum anchor samples for a fitted baseline (below -> anchored_fixed)
        merge_vif_max: (float) Maximum variance inflation factor of a template column before it is merged with a neighbor
        noise_band_sigma: (float) Width of the quiet-detection noise band in sigma
        min_physical_value: (float) All values below this are counted as faulty (set to nan)
        min_physical_run: (timedelta64(x, "s")) Minimum consecutive below-threshold seconds to count as faulty
        bg_percentile: (float) Percentile for pandas.rolling-quantile background calculation
        bg_rolling_window: (timedelta64(x, "s")) Width of the rolling background window,
        force_anchored_fixed: (bool) Forces the anchored_fixed baseline mode for all segments
        refine_sweeps: (int) Maximum number of coordinate-descent sweeps
        refine_min_dsse: (float) Refine accepts a move only if the SSE drops by more than refine_min_dsse * sigma_hat^2
        detect_sigma: (float) Detection: amplitude >= detect_sigma * SE
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