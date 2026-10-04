"""Result structures of the Mode L fitting, from one segment fit up to a whole day.

From inside out:
    ModeLFitResult: One BVLS fit: amplitudes, baseline coefficients, diagnostics.
    PassAmplitude: The fit result of one vehicle pass. Members of a merged group share
        amplitude, SE and detection flag.
    SegmentRecord: All PassAmplitudes of one segment plus baseline mode and diagnostics.
    DayFitResult: All SegmentRecords of one measurement segment and channel.
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Assistance with Docstring writing)

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from src.mode_linear_fitting.fitting_config import Segment

@dataclass(frozen=True)
class PassAmplitude:
    """Fit result of one vehicle pass.

    Attributes:
        pass_id: Index into the peak_positions passed to fit_day.
        amplitude: Fitted plume area a_j in channel unit · s. For a merged group this
            is the total area of the whole group, repeated for every member.
        group: pass_ids of all passes sharing this column (just (pass_id,) if not merged).
        tail_degenerate: True for the last group of a segment without right anchor run
            (e.g. end of the measurement): part of the template lies outside the
            segment, the amplitude is extrapolated.
        shift: Position correction τ_j − τ_j_initial by the refinement in samples.
        se: Standard error of the amplitude in channel unit · s (shared within a group).
        detected: True if amplitude ≥ detect_sigma · se.
    """

    pass_id: int
    amplitude: float
    group: tuple[int, ...]
    tail_degenerate: bool
    shift: int
    se:float
    detected: bool

@dataclass(frozen=True)
class ModeLFitResult:
    """Result of one BVLS fit of a segment.

    Attributes:
        amplitudes: Amplitude per template column = plume area in channel unit · s.
        baseline_coeffs: Baseline coefficients β (empty, [offset] or [offset, slope]).
        residual_rms: RMS of the residual over the valid samples.
        condition_number: Condition number of the design matrix (collinearity diagnostic).
        amplitudes_se: Standard error per amplitude, NaN if computed with with_se=False.
        sse: Sum of squared residuals.
        dof: Degrees of freedom n − p (valid samples minus columns).
        model: Fitted signal A · [a, β] over the whole segment, for plots.
    """

    amplitudes: np.ndarray
    baseline_coeffs: np.ndarray
    residual_rms: float
    condition_number: float

    amplitudes_se: np.ndarray
    sse: float
    dof: int
    model: np.ndarray


@dataclass(frozen=True)
class SegmentRecord:
    """Fit outcome of one segment.

    Attributes:
        segment: The fitted segment.
        baseline_mode: "anchored_fixed", "fitted_const", "fitted_linear", or "failed"
            if the segment could not be fitted (then passes is empty).
        baseline_coeffs: Fitted baseline coefficients, empty for anchored_fixed.
        n_anchor: Number of anchor samples in the segment.
        condition_number: Condition number of the design matrix.
        residual_rms: RMS of the fit residual.
        passes: Result per pass of the segment.
        model: Fitted signal including the fixed baseline, only with
            fit_day(..., keep_signals=True).
    """

    segment: Segment
    baseline_mode: str
    baseline_coeffs: tuple[float, ...]
    n_anchor: int
    condition_number: float
    residual_rms: float
    passes: list[PassAmplitude]
    model: np.ndarray | None = None



@dataclass(frozen=True)
class DayFitResult:
    """Fit result of one measurement segment and channel.

    Attributes:
        channel: Fitted channel.
        segments: One record per segment.
        residual: Background-subtracted, median-centered signal the fits ran on,
            only with fit_day(..., keep_signals=True).
    """

    channel: str
    segments: list[SegmentRecord]
    residual: np.ndarray | None = None

    @property
    def pass_amplitudes(self) -> dict[int, PassAmplitude]:
        """All pass results keyed by pass_id. Passes of failed segments are missing."""
        return {p.pass_id: p for s in self.segments for p in s.passes}
