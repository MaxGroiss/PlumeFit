"""This module contains the output structure of the template extraction pipeline.

- ExtractionResult bundles the results from one channel over one extraction day/segment
- BatchResult is a helper class that holds a list of ExtractionResults derived in batch template extraction
- CombinedResult bundles Extractions Results from one channel over x Days in one Object

Extraction Pipeline:
template_extraction_batch(loops: template_extraction -> One Channel One Day -> ExtractionResult) -> BatchResult
-> Get Data for one Channel of batch days -> BatchResult.combined_by_channel(does: grouped_by_channel + combine_results)
-> Returns a dict with key = Channel Name, Value = CombinedResult

"""
# This file contains code created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose/extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Catch potential dimension/value errors and guard them with named raises)

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from src.plume_template_extraction.extraction_config import ExtractionConfig
from src.plume_template_extraction.plume_status import PlumeStatus


# AI-Assisted: <Fable 5> ; (Deduplicate Properties by using a Property only Mixin)
class _PlumeStats:
    """Shared statistics over the normalized plume matrix.

    Property-only mixin for the result containers — expects the inheriting dataclass
    to provide normalized_matrix, peak_index and dt.
    """

    @property
    def n_valid(self) -> int:
        """Number of plumes in the final normalized matrix."""
        return self.normalized_matrix.shape[0]

    @property
    def time_axis(self) -> np.ndarray:
        """Relative time axis in seconds, peak at t = 0."""
        n_samples = self.normalized_matrix.shape[1]
        return (np.arange(n_samples) - self.peak_index) * self.dt

    @property
    def std_envelope(self) -> np.ndarray:
        """Standard deviation of the normalized plume matrix (axis=0)."""
        return np.std(self.normalized_matrix, axis=0)

    @property
    def std_mean_envelope(self) -> np.ndarray:
        """Standard error of the mean (σ/√n)."""
        return self.std_envelope / np.sqrt(self.n_valid)


@dataclass(frozen=True)
class ExtractionResult(_PlumeStats):
    """Extraction result of ONE channel for ONE measurement day.

    Per plume arrays are row aligned by passes in the same order as pass_indices.
    Meaning row i of the plume array belongs to the pass_indices[i] vehicle.

    Attributes:
        channel: (str) Extracted channel
        config: (ExtractionConfig) Config leading to this result
        normalized_matrix: (np.ndarray) Plumes normalized to A = 1 centered around the peak
        centered_matrix: (np.ndarray) Plumes centered around the peak
        peak_index: (int) Peak column in plume matrix
        dt: (float) Sampling interval in seconds
        pass_indices: (np.ndarray) Light-barrier pass index per plume row (vehicle link within source_day)
        source_day: (str) Measurement day the plumes were extracted from
        n_isolated: (int) Number of isolated passes before QA
        qa_counts: (dict[PlumeStatus, int]) Number of plumes per QA status
        pollutant_offsets: (np.ndarray | None) Offset (sensor delay) of the pollutant peak relative to the
                           associated co2 peak in samples (pollutant channels only)
        trigger_delays: (np.ndarray | None) CO2 peak delay after the light-barrier trigger in samples
                        (CO2 channels only)
    """

    channel: str
    config: ExtractionConfig

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

    @property
    def mean_shape(self) -> np.ndarray:
        """Mean Shape of the normalized plume matrix (axis=0)."""
        return np.mean(self.normalized_matrix, axis=0)


@dataclass(frozen=True)
class BatchResult:
    """Container class for batch extractions
    Gathers all ExtractionResults of a batch run in a list

    Calling:
        - grouped_by_channel: -> (dict) channel_name: list[ExtractionResult]
        - combined_by_channel: -> (dict) channel_name: CombinedResult

    Attributes:
        results: (list[ExtractionResult]) One ExtractionResult per day and channel

    """

    results: list[ExtractionResult]

    def grouped_by_channel(self) -> dict[str, list[ExtractionResult]]:
        """ Returns a dict with key = channel and value = list of results for that channel per day

        This method automatically catches the case that one co2 channel is used as reference for
        multiple pollutant channels. Combining all results blindly would duplicate this co2 channel.

        :return: dict with key = channel and value = list of results for that channel per day
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
        """Combines the per-day results of every channel into one CombinedResult.

        Convenience wrapper around :func:`grouped_by_channel` + :func:`combine_results`,
        use grouped_by_channel directly if the per-day results are needed.

        :return: dict with key = channel and value = CombinedResult over all days
        """
        return {ch: combine_results(res) for ch, res in self.grouped_by_channel().items()}


@dataclass(frozen=True)
class CombinedResult(_PlumeStats):
    """Cross-day combination of the ExtractionResults of ONE channel.

    All per-plume arrays are row-aligned: row i of the matrices belongs to pass_indices[i]
    on source_days[i]. Individual plumes are kept (not averaged) so they stay filterable
    by vehicle, e.g. for subgroup means per vehicle class.


    Attributes:
        channel: (str) Combined channel
        normalized_matrix: (np.ndarray) Stacked Plumes normalized to A = 1 centered around the peak
        centered_matrix: (np.ndarray) Stacked Plumes centered around the peak
        peak_index: (int) Peak column in plume matrix
        dt: (float) Sampling interval in seconds

        pass_indices: (np.ndarray) Light-barrier pass index per plume row only per day unique
        source_days: (np.ndarray) Measurement day per plume row (together with pass_indices the vehicle key)

        pollutant_offsets: (np.ndarray | None) Offset (sensor delay) of the pollutant peak relative to the
                           associated co2 peak in samples (pollutant channels only)
        trigger_delays: (np.ndarray | None) CO2 peak delay after the light-barrier trigger in samples
                        (CO2 channels only)
    """

    normalized_matrix: np.ndarray
    centered_matrix: np.ndarray
    peak_index: int
    dt: float
    channel: str
    pass_indices: np.ndarray
    source_days: np.ndarray
    pollutant_offsets: np.ndarray | None = None
    trigger_delays: np.ndarray | None = None

    @property
    def mean_shape(self) -> np.ndarray:
        """Mean of the normalized plume matrix (axis=0)."""
        # The Mean Shape is re normalized to guarantee a unit-area
        mean = np.mean(self.normalized_matrix, axis=0)
        nm = mean / (np.sum(mean) * self.dt)

        return nm


def combine_results(channel_results: list[ExtractionResult], dt_rounding: int = 6) -> CombinedResult:
    """Combines per-day ExtractionResults into a single Matrix.

    It is assumed that all extractions have the same window properties ("Verbally" enforced in ExtractionConfig).

    :param channel_results: Per-day ExtractionResults from one channel
    :param dt_rounding: All ExtractionResults must have the same dt, to avoid float error dt is rounded
    :return: CombinedResult with stacked normalized and centered matrices and concatenated metadata
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
    centered_matrix = np.vstack([r.centered_matrix for r in channel_results])
    pass_indices = np.concatenate([r.pass_indices for r in channel_results])
    source_days = np.concatenate(
        [np.full(r.normalized_matrix.shape[0], r.source_day) for r in channel_results]
    )
    offsets = [r.pollutant_offsets for r in channel_results]
    # None check for Co2 Results
    pollutant_offset = None if any(o is None for o in offsets) else np.concatenate(offsets)
    delays = [r.trigger_delays for r in channel_results]
    # None check for pollutant results
    trigger_delays = None if any(d is None for d in delays) else np.concatenate(delays)


    return CombinedResult(
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