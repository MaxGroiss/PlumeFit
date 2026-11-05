"""QA checks and non-detect policies of the Mode L fitting, applied per pass after the fit of a segment.

Contents:
    SegmentContext: What a QA check may look at: the fitted segment and the per-pass inputs.
    residual_above_noise, position_at_band_edge, co2_reference: Default QA checks (DEFAULT_QA_CHECKS), they only add
        flags to PassFit.qa_flags and never remove a pass.
    multiple_peaks: Optional QA check, an example of how further checks can be added.
    run_checks: Runs the QA checks of one pass.
    keep_non_detects: Default non-detect policy, the fitted amplitude stays.
    non_detects_to_noise_level: Example policy, a not detected pass reports its detection limit.
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Assistance with Docstring writing / provenance parameter)
# AI-Assisted: <Opus 5.5> ; (Assistance with optimizing the code structure for ease of use)
# AI-Assisted: <Opus 5.5> ; (Module docstring corrected for the public release)


from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from src.mode_linear_fitting.fitting_result import PassFit
from src.plume_template_extraction.extraction_config import ChannelQAConfig
from src.plume_template_extraction.extraction_quality import check_multiple_peaks


if TYPE_CHECKING:
    from src.mode_linear_fitting.fitting_config import ModeLConfig, ShapeTemplate


# AI-Assisted: <Opus 5.5> ; (Help with decision on class fields)
@dataclass(frozen=True)
class SegmentContext:
    """What a QA check may look at: the fitted segment and the per-pass inputs.

    Attributes:
        y: Segment signal the fit ran on (fixed baseline already subtracted for anchored_fixed).
        model: Fitted model over the segment, None if the segment failed.
        start: First sample of the segment in the register.
        sigma: Noise level σ̂ of the channel and day.
        cfg: Fit parameters.
        templates: Template per pass_index of the segment.
        contributions: Column contribution a · s(τ̂) over the segment per group_id (empty if failed).
        bands: Search band (lo, hi) in register samples per pass_index, empty without refinement.
        reference: CO₂ PassFit of the same pass per pass_index (pollutant fits only, empty for CO₂).
    """

    y: np.ndarray
    model: np.ndarray | None
    start: int
    sigma: float
    cfg: ModeLConfig
    templates: dict[int, ShapeTemplate]
    contributions: dict[int, np.ndarray] = field(default_factory=dict)
    bands: dict[int, tuple[int, int]] = field(default_factory=dict)
    reference: dict[int, PassFit] = field(default_factory=dict)

    def window(self, p: PassFit) -> slice:
        """Template window of a pass (support of its template at τ̂) inside the segment."""
        t = self.templates[p.pass_index]
        lo = int(p.position) - self.start - t.peak_index
        return slice(max(lo, 0), min(lo + t.values.size, self.y.size))

# This is done so QA checks do not rely on anything the pipeline does not provide
# A QA Check function takes PassFit, SegmentContext and returns a string flag or none if the pass passed the qa
QACheck = Callable[[PassFit, SegmentContext], "str | None"]
# How NonDetectPolicy functions should be structured they take PassFit, "ModeLConfig" as input and can manipulate
# PassFit depending on the active policy
NonDetectPolicy = Callable[[PassFit, "ModeLConfig"], PassFit]

# -------------------------------------------------------------------------------------------- QA checks

def residual_above_noise(p: PassFit, ctx: SegmentContext) -> str | None:
    """ "residual_high" if RMS(y-model) in the template window of the pass exceeds k₃σ̂ """
    if not p.fitted or ctx.model is None:
        return None
    r = (ctx.y - ctx.model)[ctx.window(p)]
    r = r[np.isfinite(r)]
    if r.size == 0:
        return None
    return "residual_high" if np.sqrt(np.mean(r ** 2)) > ctx.cfg.noise_band_sigma * ctx.sigma else None

def position_at_band_edge(p: PassFit, ctx: SegmentContext) -> str | None:
    """ "band_edge" if the refined position equals the lower or upper end of the search band C_j -> indicating that
        the refinment may have gotten better if the band would be broader.
    """
    if not p.fitted or p.pass_index not in ctx.bands:
        return None
    lo, hi = ctx.bands[p.pass_index]
    return "band_edge" if p.position in (lo, hi) else None


def co2_reference(p: PassFit, ctx: SegmentContext) -> str | None:
    """"co2_missing" / "co2_not_detected" for pollutant passes whose CO₂ result is missing / not detected."""
    ref = ctx.reference.get(p.pass_index)
    if ref is None:
        return None
    if not ref.fitted:
        return "co2_missing"
    return None if ref.detected else "co2_not_detected"

# AI-Assisted: <Opus 5.5> ; (Port of the multi peak check from extraction to demonstrate how to integrate further cheks)
# Is not used by default only for demonstration
def multiple_peaks(p: PassFit, ctx: SegmentContext) -> str | None:
    """"multiple_peaks" if the own plume estimate (residual + own column) has more than one prominent peak.

    Uses the extraction check check_multiple_peaks with the extraction thresholds: prominence above
    max(ρ_min, ρ_rel · h) with ρ_min = k₂σ̂ (ChannelQAConfig defaults). Not in the default checks.
    """
    if not p.fitted or ctx.model is None or p.group_id not in ctx.contributions:
        return None
    w = ctx.window(p)
    own = (ctx.y - ctx.model + ctx.contributions[p.group_id])[w]
    if not np.all(np.isfinite(own)):
        return None
    qa = ChannelQAConfig()
    peak = int(p.position) - ctx.start - w.start
    if not 0 <= peak < own.size:
        return None
    many = check_multiple_peaks(own, peak, 0.0, qa.min_prominence_ratio, qa.prominence_floor_sigma * ctx.sigma)
    return "multiple_peaks" if many else None


# -------------------------------------------------------------------------------------------- define defaults
# Enter QA functions here so they are ran by default when running run_checks
DEFAULT_QA_CHECKS: tuple[QACheck, ...] = (residual_above_noise, position_at_band_edge, co2_reference)

def run_checks(p: PassFit, ctx: SegmentContext, checks: tuple[QACheck, ...]) -> PassFit:
    """PassFit with the flags of all failed checks appended to qa_flags."""
    flags = tuple(f for f in (check(p, ctx) for check in checks) if f)
    return replace(p, qa_flags=p.qa_flags + flags) if flags else p


# -------------------------------------------------------------------------------------------- non-detect policies
# Defines how passes marked as not detected should be handled
# e.g. replace the fitted amplitude by a value derived from noise
# ! How passes are marked as not detected is not handled here !
# If you want to change the detection mechanism see criteria_of_detection in mode_l_fitting
# criteria_of_detection is currently also used in the pollutant refinement mechanism so be aware when changing it

# The default handling can be changed by assigning a different function in fitting_config class PairSetup
# The non-detect handling runs on the qa checked data so qa does not need to be able to handel the possible
# amplitude changes by the non-detected handling
def keep_non_detects(p: PassFit, cfg: ModeLConfig) -> PassFit:
    """Default: The fitted amplitude of a not detected pass stays as it is.
       Note: There is still the not detected flag bound to it, the amplitude is just not changed
    """
    return p

# AI-Assisted: <Opus 5.5> ; (Handle not detected passes by setting their amplitude to the limit of detection -> noise)
# Example how passes flagt as not detected could be handled diffrently
def non_detects_to_noise_level(p: PassFit, cfg: ModeLConfig) -> PassFit:
    """Example policy: a not detected pass reports the noise level of its column instead of its fitted area.

    "Noise level" = detection limit LOD_j = k_det · se_j
    """
    if not p.fitted or p.detected or p.group_size != 1 or not np.isfinite(p.se):
        return p
    limit = cfg.detect_sigma * p.se
    return replace(p, amplitude=limit, group_amplitude=limit, replaced_by="noise_level")
