"""Output structures of the template extraction pipeline.

ExtractionResult: plumes of one channel from one segment (one extract_plumes call).
BatchResult: all ExtractionResults of a run_batch call.
CombinedResult: plumes of one channel stacked over all segments, the input for templates.

Flow: run_batch -> BatchResult -> combined_by_channel() -> {channel: CombinedResult}
-> optional CSV export with key (source_day, pass_index) for the vehicle mapping
-> ShapeTemplate.from_combined (mode_linear_fitting).
"""

# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose/extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Assistance with Docstring writing)

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.plume_template_extraction.extraction_config import ExtractionConfig
from src.plume_template_extraction.normalization import normalize_plumes, pooled_mean_shape, pooled_mean_se
from src.plume_template_extraction.plume_status import PlumeStatus


# AI-Assisted: <Fable 5> ; (Deduplicate Properties by using a Property only Mixin)
class _PlumeStats:
    """Statistics shared by ExtractionResult and CombinedResult.

    Property-only mixin, the inheriting dataclass provides normalized_matrix, areas,
    peak_index and dt.
    """

    @property
    def n_valid(self) -> int:
        """Number of plumes in the result."""
        return self.normalized_matrix.shape[0]

    @property
    def time_axis(self) -> np.ndarray:
        """Time relative to the peak in s, peak at t = 0."""
        n_samples = self.normalized_matrix.shape[1]
        return (np.arange(n_samples) - self.peak_index) * self.dt

    @property
    def se_mean_envelope(self) -> np.ndarray:
        """Standard error of the pooled mean shape per sample"""
        return pooled_mean_se(self.normalized_matrix, self.areas, self.dt)

    @property
    def mean_shape(self) -> np.ndarray:
        """Area-weighted (pooled) mean shape with unit area, the template of all plumes."""
        return pooled_mean_shape(self.normalized_matrix, self.areas, self.dt)


@dataclass(frozen=True)
class ExtractionResult(_PlumeStats):
    """Extracted plumes of one channel from one segment.

    All per-plume arrays are row aligned: row i belongs to the pass pass_indices[i].

    Attributes:
        channel: Extracted channel.
        config: Config that produced this result.
        areas: Plume area A_i in channel unit · s, one per plume.
        normalized_matrix: Unit-area plumes s_i, peak-centered, one per row.
        centered_matrix: Background-subtracted plumes before zeroing and
            normalization, peak-centered, one per row.
        peak_index: Column of the peak.
        dt: Sampling interval in s.
        pass_indices: Light barrier pass index per plume, unique within source_day.
        source_day: Segment date, see MeasurementRegister.source_day.
        n_isolated: Number of isolated passes before QA.
        qa_counts: Number of isolated passes per PlumeStatus.
        pollutant_offsets: Pollutant only. Pollutant peak relative to the CO₂ peak
            in samples (sensor delay). None for CO₂.
        trigger_delays: CO₂ only. CO₂ peak relative to the light barrier trigger
            in samples. None for pollutants.
    """

    channel: str
    config: ExtractionConfig

    areas: np.ndarray
    normalized_matrix: np.ndarray
    centered_matrix: np.ndarray

    peak_index: int
    dt: float

    pass_indices: np.ndarray
    source_day: str

    n_isolated: int
    qa_counts: dict[PlumeStatus, int]

    pollutant_offsets: np.ndarray | None = None
    trigger_delays: np.ndarray | None = None



@dataclass(frozen=True)
class BatchResult:
    """All ExtractionResults of one run_batch call.

    Attributes:
        results: One ExtractionResult per segment and channel.
    """

    results: list[ExtractionResult]

    def grouped_by_channel(self) -> dict[str, list[ExtractionResult]]:
        """Group the results by channel, one result per segment.

        A CO₂ channel that serves as reference for several pollutant configs appears
        once per config. These duplicates are dropped, so combining does not stack the
        same CO₂ plumes twice.

        Returns:
            {channel: [result per segment]}.

        Raises:
            ValueError: If two results of the same channel and segment selected
                different plumes (e.g. configs with different drop_co2_invalid_poll).
        """
        grouped: dict[str, dict[str, ExtractionResult]] = {}
        for r in self.results:
            per_day = grouped.setdefault(r.channel, {})
            existing = per_day.get(r.source_day)

            # The duplication check is guarded here the odd case could occur that the by name duplicated channel
            # has different plume counts for example when two extraction configs one with drop_co2_invalid_poll = False
            # and one with drop_co2_invalid_poll = True got used in the same batch extraction
            if existing is None:
                per_day[r.source_day] = r
            elif not np.array_equal(existing.pass_indices, r.pass_indices):
                raise ValueError(
                    f"Conflicting results for {r.channel} on {r.source_day}: "
                    f"different plume selection — clashing configs?"
                )
        return {ch: list(per_day.values()) for ch, per_day in grouped.items()}

    def combined_by_channel(self) -> dict[str, CombinedResult]:
        """Combine the per-segment results of every channel.

        Shortcut for grouped_by_channel + combine_results. Use grouped_by_channel if
        the per-segment results are needed.

        Returns:
            {channel: CombinedResult over all segments}.
        """
        return {ch: combine_results(res) for ch, res in self.grouped_by_channel().items()}


@dataclass(frozen=True)
class CombinedResult(_PlumeStats):
    """Plumes of one channel stacked over several segments.

    Individual plumes are kept, not averaged, so templates can be built for any
    subgroup of vehicles (see ShapeTemplate.from_combined with a mask). All per-plume
    arrays are row aligned: row i belongs to pass_indices[i] on source_days[i].

    Attributes:
        areas: Plume area A_i in channel unit · s.
        normalized_matrix: Unit-area plumes s_i, one per row.
        centered_matrix: Background-subtracted plumes before zeroing and
            normalization. NaN if the result was read from a normalized CSV.
        peak_index: Column of the peak.
        dt: Sampling interval in s.
        channel: Channel name.
        pass_indices: Light barrier pass index per plume, unique only within a day.
        source_days: Segment date per plume; (source_day, pass_index) identifies a vehicle.
        pollutant_offsets: Pollutant only, see ExtractionResult.
        trigger_delays: CO₂ only, see ExtractionResult.
    """
    areas: np.ndarray
    normalized_matrix: np.ndarray
    centered_matrix: np.ndarray
    peak_index: int
    dt: float
    channel: str
    pass_indices: np.ndarray
    source_days: np.ndarray
    pollutant_offsets: np.ndarray | None = None
    trigger_delays: np.ndarray | None = None

    def to_dataframe(self, normalized: bool = False) -> pd.DataFrame:
        """Convert to a DataFrame with one row per plume.

        Metadata columns first (channel, source_day, pass_index, area, normalized,
        trigger_delay / pollutant_offset), then one column per sample named
        "t_<time relative to the peak in s>".

        Args:
            normalized: Export the normalized (True) or the centered plumes (False).
                The normalized plumes can be rebuilt from the centered ones, not vice
                versa. Defaults to False.

        Returns:
            The plume table.
        """
        attach_matrix = self.normalized_matrix if normalized else self.centered_matrix
        meta = {"channel": self.channel,
                "source_day": self.source_days.astype(str),
                "pass_index": self.pass_indices.astype(int),
                "area": self.areas,
                "normalized": normalized}
        if self.trigger_delays is not None:
            meta["trigger_delay"] = self.trigger_delays
        if self.pollutant_offsets is not None:
            meta["pollutant_offset"] = self.pollutant_offsets
        samples = pd.DataFrame(attach_matrix, columns=[f"t_{t:.3f}" for t in self.time_axis])
        return pd.concat([pd.DataFrame(meta),samples],axis=1)


    def to_csv(self, path: Path, normalized: bool = False) -> None:
        """Write to_dataframe as CSV.

        Args:
            path: Target file.
            normalized: See to_dataframe.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe(normalized).to_csv(path, index=False, float_format="%.6g")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, dt_rounding: int = 6, baseline_anchor_s = 1.5) -> CombinedResult:
        """Rebuild a CombinedResult from a DataFrame in the to_dataframe format.

        From a normalized table the centered plumes cannot be recovered and are NaN.
        From a centered table the normalized plumes are rebuilt with normalize_plumes.

        Args:
            df: Plume table, see to_dataframe. May be filtered to a vehicle subgroup.
            dt_rounding: Decimals dt is rounded to, absorbs float noise in the column names.
            baseline_anchor_s: Zeroing duration in s, for rebuilding normalized plumes must match the extraction config.

        Returns:
            The rebuilt result.

        Raises:
            ValueError: If the table is empty, the time axis is not equidistant with
                t = 0 at the peak, or channel / normalized are mixed.

        TODO: Hardcoded Parameters like baseline_anchor should be included in the from_dataframe / to_dataframe chain
        """
        if df.empty:
            raise ValueError("Empty dataframe")

        sample_columns = df.columns[df.columns.str.startswith("t_")]
        # Sample times start at character 3 in header
        times = sample_columns.str[2:].astype(float).to_numpy()
        dt = round(float(np.median(np.diff(times))), dt_rounding)
        # The peak column is the one closest to t = 0 (robust against float formatting)
        peak_index = int(np.argmin(np.abs(times)))
        # AI-Assisted: <Opus 5> ; (Catch not equidistant time axis with peak not at 0)
        if not np.allclose(np.diff(times), dt) or not np.isclose(times[peak_index], 0.0):
            raise ValueError("Sample columns are not an equidistant time axis with t = 0 at the peak")
        # checks if all rows in the dataframe are form the same channel and are the same matrix type
        for col in ("channel", "normalized"):
            if df[col].nunique() != 1:
                raise ValueError(f"Mixed values in column '{col}': {df[col].unique()}")
        channel = str(df["channel"].iloc[0])
        # Read Matrix and if centered reconstruct the normalized matrix
        df_matrix = df[sample_columns].to_numpy(dtype=np.float64)
        n_anchor = round(baseline_anchor_s / dt)
        if bool(df["normalized"].iloc[0]):
            normalized_matrix, centered_matrix = df_matrix, np.full_like(df_matrix, np.nan)
            areas = df["area"].to_numpy(dtype=float)
        else:
            centered_matrix = df_matrix
            normalized_matrix, areas = normalize_plumes(df_matrix, dt, n_anchor, channel_name=channel)

        return cls(normalized_matrix=normalized_matrix,
                   areas=areas,
                   centered_matrix=centered_matrix,
                   peak_index=peak_index,
                   dt=dt,
                   channel=channel,
                   pass_indices=df["pass_index"].to_numpy(dtype=int),
                   source_days=df["source_day"].astype(str).to_numpy(),
                   pollutant_offsets=df["pollutant_offset"].to_numpy() if "pollutant_offset" in df.columns else None,
                   trigger_delays=df["trigger_delay"].to_numpy() if "trigger_delay" in df.columns else None)

    @classmethod
    def from_csv(cls, path: Path) -> CombinedResult:
        """Read a CSV written by to_csv, see from_dataframe.

        Args:
            path: CSV file.

        Returns:
            The rebuilt result.
        """
        return cls.from_dataframe(pd.read_csv(path, dtype={"source_day": str}),baseline_anchor_s= 1.5)



def combine_results(channel_results: list[ExtractionResult], dt_rounding: int = 6) -> CombinedResult:
    """Stack the per-segment results of one channel into one CombinedResult.

    Args:
        channel_results: Results of one channel, one per segment.
        dt_rounding: Decimals dt is rounded to before comparing segments.


    Returns:
        Stacked plumes and metadata, in the order of channel_results.

    Raises:
        ValueError: If the list is empty or the results differ in channel, dt, peak
            column or window width (they could not be stacked).
    """

    # AI-Assisted: <Opus 5> ; (Catching Value Consistency errors)
    if not channel_results:
        raise ValueError("No results provided")

    # All results must share the same channel and window properties to be stackable
    consistency = [("channel", {r.channel for r in channel_results}),
                   ("dt", {round(r.dt, dt_rounding) for r in channel_results}),
                   ("peak_index", {r.peak_index for r in channel_results}),
                   ("window width", {r.normalized_matrix.shape[1] for r in channel_results})]
    for name, values in consistency:
        if len(values) > 1:
            raise ValueError(f"All results must share the same {name}, got {values}")

    normalized_matrix = np.vstack([r.normalized_matrix for r in channel_results])
    areas = np.concatenate([r.areas for r in channel_results])
    centered_matrix = np.vstack([r.centered_matrix for r in channel_results])
    pass_indices = np.concatenate([r.pass_indices for r in channel_results])
    source_days = np.concatenate(
        [np.full(r.normalized_matrix.shape[0], r.source_day) for r in channel_results]
    )
    # CO₂ results carry no pollutant offsets, pollutant results no trigger delays
    offsets = [r.pollutant_offsets for r in channel_results]
    pollutant_offset = None if any(o is None for o in offsets) else np.concatenate(offsets)
    delays = [r.trigger_delays for r in channel_results]
    trigger_delays = None if any(d is None for d in delays) else np.concatenate(delays)

    return CombinedResult(
        areas=areas,
        normalized_matrix=normalized_matrix,
        centered_matrix=centered_matrix,
        peak_index=channel_results[0].peak_index,
        dt=round(channel_results[0].dt, dt_rounding),
        channel=channel_results[0].channel,
        pass_indices=pass_indices,
        source_days=source_days,
        pollutant_offsets=pollutant_offset,
        trigger_delays=trigger_delays
    )