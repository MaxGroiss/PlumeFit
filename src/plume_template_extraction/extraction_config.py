"""This module contains the input configuration structure of the template extraction pipeline.

- ExtractionConfig holds the Extraction configuration for ONE Channel / Co2 Poll Couple
- ChannelQAConfig  holds the extraction QA Parameters of ONE Channel

The separation between ExtractionConfig and ChannelQAConfig was done because most of the Extraction Config can/should
be kept the same across different channels while ChannelQAConfig is highly channel dependent

All QA Checks happen in extraction_quality.py

Extraction Pipeline:
Expects One MeasurementRegistry and One Extraction Config per Channel -> Extraction and QA Checks are performed using
the defined Config Parameters -> ExtractionResult.

Disclaimer:
Defaults are mainly derived from observing pipeline behavior on measurement data from the CARES_Milan Campaign 2021
QA Parameters have to be tuned for the given sensors to reduce/avoid unexpected extraction behavior.
"""

from dataclasses import dataclass, field

import numpy as np

@dataclass(frozen=True)
class ChannelQAConfig:
    """QA Parameters of ONE Channel

    Setting Attributes to None if available derives those parameters directly from the measurement data.

    Attributes:
        band_before_co2_peak: (timedelta64(x, "s")) Poll Peak search band before co2 peak
        band_after_co2_peak: (timedelta64(x, "s")) Poll Peak search band after co2 peak

        smooth_window: (timedelta64(x, "ms")) Filter window has to fit in a plume peak, needs enough samples for noise
                        suppression, converted into an uneven sample window > polyorder
        smooth_polyorder: (int) Order of savitzky-golay filter

        min_physical_value: (float) Faulty Value Threshold (including background)
        min_physical_run: (timedelta64(x, "s")) Minimum consecutive below-threshold seconds to count as faulty

        min_peak_above_bg: (float | None) Values below this threshold can't be a peak (above background)
                            None -> Derived trough vehicle free residual noise * peak_above_bg_sigma
        peak_above_bg_sigma: (float) Sigma multiplier for the derived min_peak_above_bg

        min_prominence_ratio: (float) Second peak counts as a real peak if its prominence exceeds this fraction of the
                              main peak
        min_prominence_floor: (float | None) Absolute prominence floor for multi peak detection (noise suppression)
                            None -> Derived trough noise estimation * prominence_floor_sigma
        prominence_floor_sigma: (float) Sigma multiplier for the derived min_prominence_floor

        tail_rise_ratio: (float | None) Cumulative rise in tail flagged if exceeding this faction of peak height
                         None -> Runs Check trough tail_rise_abs instead
        tail_rise_abs: (float | None) Absolute Threshold for cumulative rise, avoids bias behavior between peak heights
                         None -> Derives the value by rise caused by noise in vehicle free windows of tail length
                                A tail is flagged if its cumulative re-rise exceeds the value that noise alone stays
                                below in tail_percentile % of vehicle-free windows
        tail_percentile: (float) Percentile of the vehicle-free tail statistic used as tail_rise_abs
                                    Increasing the value lowers the filter strength
    """

    # Search band where the pollutant peak is expected to occur around the associated Co2 peak
    # Manually derive trough sensor/setup parameters
    band_before_co2_peak: np.timedelta64 = np.timedelta64(2, "s")
    band_after_co2_peak: np.timedelta64 = np.timedelta64(4, "s")

    # Savitzky-Golay Parameters used in pollutant peak finding in the plume_quality/assess_pollutant_peak_centered fct.
    # Filter window has to fit in a plume peak, needs enough samples for noise suppression, has to be an uneven window
    # size and samples > smooth polyorder
    smooth_window: np.timedelta64 = np.timedelta64(2500, "ms")
    smooth_polyorder: int = 2

    # QA-Parameters
    min_physical_value: float = 5.0
    min_physical_run: np.timedelta64 = np.timedelta64(2,"s")

    min_peak_above_bg: float | None = None
    peak_above_bg_sigma: float = 3.0

    min_prominence_ratio: float = 0.3
    min_prominence_floor: float | None = None
    prominence_floor_sigma: float = 3.0

    tail_rise_ratio: float | None = None
    tail_rise_abs: float | None = None
    tail_percentile: float = 95.0

@dataclass(frozen=True)
class ExtractionConfig:
    """ Extraction Parameters

    Parameters that are not obviously channel dependent should be kept the same across an analysis.
    Configs that are aiming to extract pollutant templates need to have a co2 config included because co2 and pollutant
    channels are tightly coupled and the co2 channel is used to assist pollutant extraction.

    Attributes:
        co2_channel: (str) Name of the Co2 Channel present in MeasurementRegister this config is for
        poll_channel: (str | None) Name of the pollutant channel (coupled with co2_channel)
        drop_co2_invalid_poll: (bool) Decides if the pipeline drops all co2 plumes that are connected to
                               an invalid pollutant plume. For Template generation this should be set to false

        min_gap: (timedelta64(x, "s")) Minimum gap between passing "Vehicle Neighbors" to be counted as isolated

        window_before: (timedelta64(x, "s")) Left border of the isolation window around the LightBarrier Trigger
        window_after: (timedelta64(x, "s")) Right border of the isolation window around the LightBarrier Trigger

        peak_search_after: (timedelta64(x, "s")) TimeFrame in wich Co2 peak is expected after LB Trigger
        window_before_peak: (timedelta64(x, "s")) TimeFrame before the Co2 peak (Cutout Window)
        window_after_peak: (timedelta64(x, "s")) TimeFrame after the Co2 peak (Cutout Window)

        bg_percentile: (float) Choosing a low percentile ensures that peaks don't influence the background calculation
        bg_rolling_window: (timedelta64(x, "s")) Time width of the rolling background window,
                           converted to samples during extraction should be comfortably greater than a plume window
    """

    co2_channel: str
    poll_channel: str | None = None
    drop_co2_invalid_poll: bool = False

    min_gap: np.timedelta64 = np.timedelta64(30, "s")
    window_before: np.timedelta64 = np.timedelta64(10, "s")
    window_after: np.timedelta64 = np.timedelta64(25, "s")

    peak_search_after: np.timedelta64 = np.timedelta64(15, "s")


    # TODO: Could get problematic: Same Cutout for Plumes with varying rise/decay behaviour
    # The whole Template System builds upon creating templates for different vehicles with varying rise/decay times
    # The cutout window is constant over one channel meaning all plumes get the same cutout if a vehicle class shows
    # slower plume behaviour it may not fit in this window and if it fits does it hold the same information like a
    # faster plume with more padding around the start and "end" of the plume ? Maybe a gradient based approach like
    # in TUG-PDA is better but then the algorithm has to deal with varying window sizes ?
    window_before_peak: np.timedelta64 = np.timedelta64(5, "s")
    window_after_peak: np.timedelta64 = np.timedelta64(20, "s")


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


    @staticmethod
    def as_samples(td: np.timedelta64, dt: float) -> int:
        """Calculate the number of samples in a time duration.

        :param td: Time duration
        :param dt: Sampling interval in seconds
        :return: Number of samples (rounded)
        """
        return round(td / np.timedelta64(1, "s") / dt)
