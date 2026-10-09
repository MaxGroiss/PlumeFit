"""Input configuration of the template extraction pipeline.

ExtractionConfig holds the window and background parameters of one CO₂ channel or one
CO₂/pollutant channel pair, ChannelQAConfig the QA parameters of one channel. They are
separated because the windows should stay the same across all channels of an analysis,
while the QA thresholds are channel dependent.

Pipeline: one MeasurementRegister and one ExtractionConfig per channel (pair)
-> extract_plumes -> ExtractionResult. All QA checks live in extraction_quality.py.

Typical values are derived by looking at plumes during the development process. May vary for different campaigns.
"""
# This file contains docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Assistance with Docstring writing)

#TODO: Here and in QA: Add some sort of maximum threshold for co2/pollutants to filter calibration / testing plumes with generally way higher amplitudes as a regular plume

from dataclasses import dataclass, field

import numpy as np

@dataclass(frozen=True)
class ChannelQAConfig:
    """QA parameters of one channel.

    Thresholds set to None are derived per measurement segment from the vehicle-free
    residual (noise σ̂ or tail statistic), see extraction_quality.resolve_qa_thresholds.

    Attributes:
        band_before_co2_peak: Pollutant only. Accepted pollutant peak position before
            the CO₂ peak. Default 2 s. Sensor dependent, derive from the distribution of
            ExtractionResult.pollutant_offsets.
        band_after_co2_peak: Pollutant only. Accepted pollutant peak position after
            the CO₂ peak. Default 4 s. Sensor dependent, see band_before_co2_peak.

        smooth_window: Pollutant only. Savitzky-Golay window for the peak search.
            Default 2.5 s, typical 1–4 s. Must be shorter than the plume peak;
            converted to an odd number of samples > smooth_polyorder.
        smooth_polyorder: Pollutant only. Polynomial order of the Savitzky-Golay filter.
            Default 2

        min_physical_value: Lowest physically plausible value incl. background in the
            channel unit. Default 5.0. Channel dependent, no universal default.
        min_physical_run: Minimum duration below min_physical_value to count as a
            faulty recording (NULL_DATA). Default 2 s

        min_peak_above_bg: Minimum peak height above the background, h_min.
            None (default) derives h_min = median(r_F) + peak_above_bg_sigma · σ.
        peak_above_bg_sigma: k₁ for the derived h_min. Default 3.0.

        min_effective_width: Minimum effective plume width A / h after zeroing,
            T_w,min (NON_PLAUSIBLE_AREA). Default 1.5 s, typical 1–3 s. Below the
            narrowest physically plausible plume.

        min_prominence_ratio: A second peak counts if its prominence exceeds this
            fraction of the main peak height, ρ_rel. Default 0.3, typical 0.2–0.5.
            Smaller is stricter.
        min_prominence_floor: Absolute prominence floor against noise peaks, ρ_min.
            None (default) derives ρ_min = prominence_floor_sigma · σ.
        prominence_floor_sigma: k₂ for the derived ρ_min. Default 3.0, typical 3–5.

        tail_rise_ratio: CO₂ only, alternative tail threshold relative to the peak
            height of each plume. Default None (inactive).
        tail_rise_abs: CO₂ only, absolute tail threshold R_krit in the channel unit.
            None (default) calibrates it from vehicle-free windows (see tail_percentile).
            Takes precedence over tail_rise_ratio if both are set.
        tail_percentile: CO₂ only. Percentile q_an of the vehicle-free tail statistic
            used as R_krit. Default 95, typical 90–99. Higher is more tolerant.
    """

    band_before_co2_peak: np.timedelta64 = np.timedelta64(2, "s")
    band_after_co2_peak: np.timedelta64 = np.timedelta64(4, "s")

    smooth_window: np.timedelta64 = np.timedelta64(2500, "ms")
    smooth_polyorder: int = 2

    min_physical_value: float = 5.0
    min_physical_run: np.timedelta64 = np.timedelta64(2,"s")

    min_peak_above_bg: float | None = None
    peak_above_bg_sigma: float = 3.0

    min_effective_width: np.timedelta64 = np.timedelta64(1500, "ms")

    min_prominence_ratio: float = 0.3
    min_prominence_floor: float | None = None
    prominence_floor_sigma: float = 3.0

    tail_rise_ratio: float | None = None
    tail_rise_abs: float | None = None
    tail_percentile: float = 95.0

@dataclass(frozen=True)
class ExtractionConfig:
    """Window, background and QA parameters of the extraction for one channel (pair).

    Pollutant templates always need the coupled CO₂ channel: the pollutant peak is
    searched around the CO₂ peak. Window parameters should be identical for all
    channels of an analysis, otherwise the results cannot be combined.

    Attributes:
        co2_channel: Name of the CO₂ channel in the MeasurementRegister.
        poll_channel: Name of the coupled pollutant channel. None (default) runs a
            CO₂-only extraction.
        drop_co2_invalid_poll: If True, CO₂ plumes whose pollutant plume failed QA
            are dropped as well. Default False (keep for template generation).

        min_gap: Minimum distance of both neighboring light barrier triggers for a pass
            to count as isolated. Default 30 s, typical 30–60 s.
            Should satisfy min_gap ≥ window_before_peak + window_after_peak + spread
            of the trigger-to-peak delay, otherwise a neighbor's tail can reach the
            zeroing window.

        window_before: Light barrier window before the trigger (NULL_DATA check).
            Default 15 s, must be ≤ min_gap.
        window_after: Light barrier window after the trigger. Default 25 s, must be
            ≥ peak_search_after and ≤ min_gap.

        peak_search_after: Time after the trigger in which the CO₂ peak is searched.
            Default 15 s, typical 5–20 s. At least the largest plausible
            trigger-to-peak delay (see ExtractionResult.trigger_delays).
        window_before_peak: Cutout before the peak, start of the template window.
            Default 10 s, typical 5–15 s. Rise time + baseline_anchor must fit.
        window_after_peak: Cutout after the peak, also tail length T_tail of the tail
            check. Default 20 s, typical 15–30 s.

        baseline_anchor: Duration at the start of the peak window whose mean is
            subtracted before normalization (zeroing). Default 1.5 s, typical 1–3 s.
            Must be < window_before_peak.

        bg_percentile: Percentile q of the rolling background in percent.
            Default 2, typical 1–5. Needs ≥ q % vehicle-free time per window.
        bg_rolling_window: Rolling background window T_RW. Default 100 s,
            typical 60–300 s; much longer than the plume window.

        co2_qa: QA parameters of the CO₂ channel.
        pollutant_qa: QA parameters of the pollutant channel, required if
            poll_channel is set.

    Raises:
        ValueError: If the window parameters are inconsistent or pollutant_qa is
            missing for a pollutant channel.
    """

    co2_channel: str
    poll_channel: str | None = None
    drop_co2_invalid_poll: bool = False

    min_gap: np.timedelta64 = np.timedelta64(30, "s")
    window_before: np.timedelta64 = np.timedelta64(15, "s")
    window_after: np.timedelta64 = np.timedelta64(25, "s")

    peak_search_after: np.timedelta64 = np.timedelta64(15, "s")


    # TODO: Could get problematic: Same Cutout for Plumes with varying rise/decay behaviour
    # The whole Template System builds upon creating templates for different vehicles with varying rise/decay times
    # The cutout window is constant over one channel meaning all plumes get the same cutout if a vehicle class shows
    # slower plume behaviour it may not fit in this window and if it fits does it hold the same information like a
    # faster plume with more padding around the start and "end" of the plume ? Maybe a gradient based approach like
    # in TUG-PDA is better but then the algorithm has to deal with varying window sizes ?
    window_before_peak: np.timedelta64 = np.timedelta64(10, "s")
    window_after_peak: np.timedelta64 = np.timedelta64(20, "s")

    baseline_anchor: np.timedelta64 = np.timedelta64(1500, "ms")

    bg_percentile: float = 2.0
    bg_rolling_window: np.timedelta64 = np.timedelta64(100, "s")

    co2_qa: ChannelQAConfig = field(default_factory=ChannelQAConfig)
    pollutant_qa: ChannelQAConfig | None = None

    def __post_init__(self):
        if self.poll_channel is not None and self.pollutant_qa is None:
            raise ValueError("Pollutant channel specified without pollutant QA config.")
        if self.window_after < self.peak_search_after:
            raise ValueError("Peak Search Window must be smaller than Window After.")
        if self.window_after > self.min_gap:
            raise ValueError("Window After must be smaller than Minimum Gap.")
        if self.window_before > self.min_gap:
            raise ValueError("Window Before must be smaller than Minimum Gap.")
        if self.baseline_anchor >= self.window_before_peak:
            raise ValueError("Baseline Anchor needs to fit in the window before the peak.")


    @staticmethod
    def as_samples(td: np.timedelta64, dt: float) -> int:
        """Convert a duration into a number of samples.

        Args:
            td: Duration.
            dt: Sampling interval [s].

        Returns:
            Number of samples, rounded to the nearest integer.
        """
        return round(td / np.timedelta64(1, "s") / dt)
