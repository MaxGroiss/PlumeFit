"""Result structures of the Mode L fitting, from one on BVLS fit up to one instuement pair and day

Form inside out:
    ModeLFitResult: One BVLS fit of a segment: amplitudes, baseline coefficients, SE, SSE, model.
    PassFit: Result of one vehicle pass in one channel. Every pass handed to the fit gest one, with status.
    SegmentDiagnostics: Optional per-segment diagnostics (model, column contributions, condition number),
                        only active with keep_diagnostics = True,
    ChannelDayResult: All PassFits of one channel and day (one per light barrier pass) and σ estimation of the day per channel
    PairDayResult: CO₂ and pollutant result of one instrument pair and day, one table row per pass.

Group areas: a merged group (non-separable passes) has one column and ine area. That area is stored once,
	as group_amplitude of the group representative (its first member). The amplitude of a merged member is NaN, it has
	no won area.
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Assistance with Docstring writing / Unicode equations)
# AI-Assisted: <Opus 5.5> ; (Output Structure Suggestions for ease of use)

from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd


STATUS_FITTED = "fitted"
STATUS_NOT_IDENTIFIABLE = "not_identifiable"
STATUS_SEGMENT_FAILED = "segment_failed"
STATUS_OUTSIDE_SIGNAL = "outside_signal"
# Status of a pass (PassFit.status)
#   fitted: the pass has a fitted column (own or shared with merged group)
#   not_identifiable: its template lies (almost) entirely on faulty (NaN) samples, see valid_area_fraction
#   segment_failed: its segment could not be fitted (SegmentNotIdentifiable)
#   outside_signal: its expected peak lies outside the signal (e.g. data cut off at end of day), no segment


@dataclass(frozen=True)
class ModeLFitResult:
    """Result of one BVLS fit of a segment (fit_mode_l).

    Attributes:
        amplitudes: Amplitude per template column a = plume area in channel unit · s.
        baseline_coeffs: Baseline coefficients β (empty, [offset] or [offset, slope]).
        amplitudes_se: Standard error per amplitude (Gl. 3.25/3.26), channel unit · s.
        sse: Sum of squared residuals over the valid samples.
        model: Fitted signal A · [a; β] over the whole segment.
    """

    amplitudes: np.ndarray
    baseline_coeffs: np.ndarray
    amplitudes_se: np.ndarray
    sse: float
    model: np.ndarray


@dataclass(frozen=True)
class PassFit:
    """ Fit result of oe vehicle pass in one channel

    Attributes:
        pass_index: Light barrier pass index (position in register.vehicle_pass_times)
                    Together wit source_day of the day result it identifies the vehicle (vehicle mapping key)
        status: fitted, not_identifiable, segment_failed or outside_signal (STATUS_* constants).
        segment_status: Baseline mode of the segment (anchored_fixed, fitted_const, fitted_linear), "failed", or ""
                        without segment (outside_signal).
        amplitude: Own plume area a_j in channel units ⋅ s. Nan if the pass is merged into a group or not fitted.
        se: Standard error of the column amplitude in channel unit ⋅ s (shared within a group). NaN if not fitted.
        detected: True if the column amplitude >= k_det ⋅ se. (shared within a group)
        group_size: Number of passes sharing the column (1 = separable), 0 if not fitted.
        group_amplitude: Area of the column (= amplitude if not merged), only set on the group representative
                         (pass_index == group_id), NaN on the other members.
        position: Peak position τ̂_j in samples of the register (τ_j⁽⁰⁾ if not fitted, which lies outside the signal
                  for outside_signal).
	    shift: τ̂_j − τ_j⁽⁰⁾ in samples (refinement), 0 if not refined.
        edge_extrapolated: True if the pass is the first column of a segment without left anchor run or the last in
                           a segment without right anchor run. (happens at edges of measurement data, extrapolation
                           fia template renormalization at cutoff)
        qa_flags: QA flags of the fit checks (fit_checks), empty if all checs passed
        replaced_by: Name of the non-detect policy that replaced the amplitude, for the fitted value.
        template: Provenance of the temaplte of the pass (ShapeTemplate.provenance)
    """

    pass_index: int
    status: str
    segment_status: str = ""
    amplitude: float = np.nan
    se: float = np.nan
    detected: bool = False
    # Pass index of the first member
    group_id: int = -1
    group_size: int = 0
    group_amplitude: float = np.nan
    position: float = np.nan # has to be a float because int cannot be np.nan
    shift: int = 0
    edge_extrapolated: bool = False
    qa_flags: tuple[str, ...] = ()
    replaced_by: str = ""
    template: str = ""

    @property
    def fitted(self) -> bool:
        """True if the pass has a fitted column."""
        return self.status == STATUS_FITTED

@dataclass(frozen=True)
class SegmentDiagnostics:
    """Diagnostics of one fitted segment, only with keep_diagnostics = True

    Attributes:
        start: First sample of the segment in the register.
        stop: Sample after the segment (exclusive).
        baseline_mode: Baseline mode or "failed"
        pass_indicies: PAsses of the segment.
        baseline_coeffs: Fitted baseline coefficients β (empty for anchored_fixed).
        condition_number: Condition number of the final design matrix (collinearity diagnostic).
        residual_rms: RMS of the fit residual over the valid samples.
        model: Fitted signal including the fixed baseline (anchor fixed), length stop - start.
        columns: Contribution a · s(τ̂) of every column as (pass_indices of the column, values over the segment).
    """

    start: int
    stop: int
    baseline_mode: str
    pass_indices: tuple[int, ...]
    baseline_coeffs: tuple[float, ...] = ()
    condition_number: float = np.nan
    residual_rms: float = np.nan
    model: np.ndarray | None = None
    columns: tuple[tuple[tuple[int, ...], np.ndarray], ...] = ()

# Columns of the per-pass table (to_dataframe), shift in s instead of samples
# Done so the CSV columns automatically adjust to field changes in PassFit class
_PASS_COLUMNS = [f.name for f in fields(PassFit) if f.name not in ("pass_index", "shift")]

@dataclass(frozen=True)
class ChannelDayResult:
    """Fit result of one channel on one measurement day (one MeasurementRegister).

    Attributes:
        channel: Fitted channel.
        source_day: Day of the register ("YYYY-MM-DD").
        dt: Sampling interval in s.
        passes: One PassFit per light barrier pass of the register, in pass_index order.
        sigma: Noise level σ estimation of the channel on this day (channel unit)
        diagnostics: SegmentDiagnostics per segment, only with keep_diagnostics = True
        residual: Background-subtracted, median-centered signal the fits ran on, only with keep_diagnostics = True
    """

    channel: str
    source_day: str
    dt: float
    passes: list[PassFit]
    sigma: float
    diagnostics: list[SegmentDiagnostics] | None = None
    residual: np.ndarray | None = None

    def column_amplitudes(self) -> np.ndarray:
        """ Area of the column of every pass (own area, or the group area for merged members), Nan if not fitted."""
        area = {p.group_id: p.group_amplitude for p in self.passes if p.fitted and p.pass_index == p.group_id}
        return np.array([area[p.group_id] if p.fitted else np.nan for p in self.passes])

    def to_dataframe(self) -> pd.DataFrame:
        """One row per pass: source_day, pass_index, channel and the PassFit fields (shift in s, flags joined by ';')."""

        rows = [{"source_day": self.source_day, "pass_index": p.pass_index, "channel":self.channel,
                 **{c: getattr(p, c) for c in _PASS_COLUMNS}, "shift_s": p.shift * self.dt} for p in self.passes]
        df = pd.DataFrame(rows)
        df["qa_flags"] = df["qa_flags"].map(";".join)
        return df

    # AI-Assisted: <Opus 5.5> ; (Write the diagnostics of one channel and day as CSVs)
    def to_csv(self, folder: Path) -> None:
        """Write the diagnostics (keep_diagnostics=True) of this result as two gzip-compressed CSVs into folder.

        Files (one set per channel, one folder per day):
            <channel>_segments.csv.gz   one row per SegmentDiagnostics: segment_id (= first sample, the key of the
                                        segment), stop, duration_s, baseline_mode, n_passes, pass_indices and
                                        baseline_coeffs (';'-joined), condition_number, residual_rms, sigma and
                                        rms_over_sigma (≈ 1: the model explains the signal).
            <channel>_curves.csv.gz     long table segment, curve, sample, value (≈ 3 MB per channel and day):
                                        the residual of the day ("signal", segment -1), the model of every segment
                                        ("model") and the contribution of every column (curve = its pass indices
                                        ';'-joined, only the samples where it is not 0).

        Raises:
            ValueError: If the result has no diagnostics (fitted with keep_diagnostics=False).
        """
        if self.diagnostics is None or self.residual is None:
            raise ValueError(f"{self.channel} {self.source_day}: no diagnostics, fit with keep_diagnostics=True")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"segment_id": d.start, "stop": d.stop, "duration_s": (d.stop - d.start) * self.dt,
                       "baseline_mode": d.baseline_mode, "n_passes": len(d.pass_indices),
                       "pass_indices": ";".join(map(str, d.pass_indices)),
                       "baseline_coeffs": ";".join(repr(float(b)) for b in d.baseline_coeffs),
                       "condition_number": d.condition_number, "residual_rms": d.residual_rms, "sigma": self.sigma,
                       "rms_over_sigma": d.residual_rms / self.sigma}
                      for d in self.diagnostics]).to_csv(folder / f"{self.channel}_segments.csv.gz", index=False)
        curves = [pd.DataFrame({"segment": -1, "curve": "signal", "sample": np.arange(self.residual.size),
                                "value": self.residual})]
        for d in self.diagnostics:
            if d.model is None:
                continue
            curves.append(pd.DataFrame({"segment": d.start, "curve": "model", "sample": np.arange(d.start, d.stop),
                                        "value": d.model}))
            for members, values in d.columns:
                nz = np.flatnonzero(values)
                curves.append(pd.DataFrame({"segment": d.start, "curve": ";".join(map(str, members)),
                                            "sample": d.start + nz, "value": values[nz]}))
        pd.concat(curves, ignore_index=True).to_csv(folder / f"{self.channel}_curves.csv.gz", index=False,
                                                    float_format="%.6g")


@dataclass(frozen=True)
class PairDayResult:
    """CO₂ and pollutant result of one instrument pair on one day.

    Attributes:
        name: Pair name (e.g. "NO2").
        co2: Result of the CO₂ channel (positions relative to the LB-Trigger
        pollutant: Result of the pollutant channel (positions coupled to the CO2 result fo the same pass).
                   None for a CO2 only setup. Defaults to None.
    """

    name: str
    co2: ChannelDayResult
    pollutant: ChannelDayResult | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """One row per pass with both channels

        ! Channel ratios are calculated without unit conversion -> needs to be done at evaluation point !
        Co2 only (pollutant None): source_day, pass_index, pair and the co2_<field> columns.

        Columns: source_day, pass_index, pair, co2_<field>, poll_<field> (Pass fit dields),
                 coupled (CO2 pass fitted, i.e. the pollutant was placed ath the refined CO2 peak),
                 area_ratio = a_poll / a_CO2 of passes seperable in both cahnnels (NaN otherwise) and its relatice SE by
                 delta method √((se_poll/a_poll)² + (se_CO₂/a_CO₂)²), assuming independent errors of both instruments.

        Output Example:
        pass_index  co2_status      poll_status  poll_shift_s  poll_qa_flags   coupled   area_ratio  area_ratio_rel_se
          0         fitted          fitted       0.5                           True      0.5         0.035355
          1         segment_failed  fitted       0.0           co2_missing     False     Nan         Nan
        """

        co2 = self.co2.to_dataframe().drop(columns="channel")
        key = ["source_day", "pass_index"]
        df = co2.rename(columns={c: f"co2_{c}" for c in co2.columns if c not in key})
        df.insert(2, "pair", self.name)

        if self.pollutant is None:
            return df
        poll = self.pollutant.to_dataframe().drop(columns="channel")
        df = df.merge(poll.rename(columns={c: f"poll_{c}" for c in poll.columns if c not in key}), on=key)
        df["coupled"] = df["co2_status"] == STATUS_FITTED
        # errstate is only here to suppress the division by NaN warnings for not fitted passes -> Division by NaN = Nan
        # division by zero -> inf
        with np.errstate(divide="ignore", invalid="ignore"):
            df["area_ratio"] = df["poll_amplitude"] / df["co2_amplitude"]
            # np.hypot = (= √(x² + y²)
            df["area_ratio_rel_se"] = np.hypot(df["poll_se"] / df["poll_amplitude"], df["co2_se"] / df["co2_amplitude"])
        return df
