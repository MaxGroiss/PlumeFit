"""Result structures of the Mode L fitting pipeline, from a single segment fit up to a full day.

Hierarchy (inside out):
    ModeLFitResult  One BVLS fit of a segment: amplitudes, baseline coefficients and
                    diagnostics (standard errors, SSE, condition number, model signal)
    PassAmplitude   The fit result mapped back to ONE vehicle pass; members of a merged
                    (non-separable) group share amplitude, SE and detection flag
    SegmentRecord   All PassAmplitudes of one segment plus baseline mode and fit diagnostics
    DayFitResult    All SegmentRecords of one day and channel; pass_amplitudes flattens
                    them into {pass_id: PassAmplitude}

"""
# This file contains code created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from src.mode_linear_fitting.fitting_config import Segment

@dataclass(frozen=True)
class PassAmplitude:
    """Fit result per vehicle pass.

    Attributes:
        pass_id: (int) ID of the vehicle pass
        amplitude: (float) Fitted amplitude, equals the area under the plume
        group: (tuple[int, ...]) pass_ids of contributing passes that were considered non-splittable
               (single-pass groups included)
        tail_degenerate: (bool) Last group of a segment without right anchor (e.g. end of day)
                         -> amplitude unreliable
        shift: (int) Position correction through refine in samples
        se: (float) Standard error of the amplitude (shared by all members of a merged group)
        detected: (bool) amplitude >= detect_sigma * se
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
    """Result of one Mode L segment fit.

    Attributes:
        amplitudes: (np.ndarray) Fitted amplitude per column, equals the area under the plume in (signal unit)*s.
        baseline_coeffs: (np.ndarray) The fitted baseline coefficients (empty / const / linear)
        residual_rms: (float) RMS of the fit residual over the valid segment samples
        condition_number: (float) Condition number of the design matrix (collinearity diagnostic)
        amplitudes_se: (np.ndarray) Standard error per amplitude (nan if with_se=False)
        sse: (float) Sum of squared residuals
        dof: (int) Degrees of freedom (valid samples - columns)
        model: (np.ndarray) Fitted model over the whole segment (A @ x, incl. baseline columns)
               for plotting / diagnostics
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
    """Contains information about the fit of a segment.

    Attributes:
        segment: (Segment) The fitted segment
        baseline_mode: (str) Baseline mode used: anchored_fixed | fitted_const | fitted_linear | failed
        baseline_coeffs: (tuple[float, ...]) The fitted baseline coefficients
        n_anchor: (int) Number of anchor samples in the segment
        condition_number: (float) Condition number of the design matrix
        residual_rms: (float) RMS of the fit residual
        passes: (list[PassAmplitude]) Results per vehicle
        model: (np.ndarray | None) Optional (fit_day(..., keep_signals=True)): fitted model over
               the segment incl. baseline (for plotting)
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
    """Result of the day fitting for one channel.

    Attributes:
        channel: (str) The channel that was fitted
        segments: (list[SegmentRecord]) One record per segment of the day
        residual: (np.ndarray | None) Optional (fit_day(..., keep_signals=True)): the
                  background-corrected, median-centred day signal the fits ran on
    """

    channel: str
    segments: list[SegmentRecord]
    residual: np.ndarray | None = None

    @property
    def pass_amplitudes(self) -> dict[int, PassAmplitude]:
        """Returns all pass amplitudes of the day fit keyed by pass_id."""
        return {p.pass_id: p for s in self.segments for p in s.passes}
