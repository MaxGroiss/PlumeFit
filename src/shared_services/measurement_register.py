"""Measurement register

Contains the datastructures the package builds upon.
Designed to keep the package agnostic against input data format.
"""
# This file contains code created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd


@dataclass
class MeasurementRegister:
    """Datainterface for the package pipelines

    One MeasurementRegister holds relevant data of one measurement day/segment

    Implements __post_init__ validation checks for pipeline compatibility

    Default Factory for MeasurementRegister is from_dataframe

    Attributes:
        timestamps: (np.ndarray) Holds the Time Series of the Measurement Data
        channel_matrix: (np.ndarray) Holds the Pollutant measurements
        channel_names: (list[str]) Holds the Name of the Pollutants in the channel_matrix
        vehicle_pass_times: (np.ndarray) Holds the Light Barrier Detection Timestamps
        timestamp_tolerance_ms: (float) Tolerance for timestamp equidistance validation in milliseconds
        skip_dt_validation: (bool) Skips the timestamps delta validation (always estimates from median)
    """
    timestamps: np.ndarray
    channel_matrix: np.ndarray
    channel_names: list[str]


    vehicle_pass_times: np.ndarray


    timestamp_tolerance_ms: float = 5.00
    skip_dt_validation: bool = False

    def __post_init__(self):

        # Performing "Datatype" checks
        if not np.issubdtype(self.vehicle_pass_times.dtype, np.datetime64):
            raise TypeError("vehicle_pass_times must be datetime64 dtype")
        if not np.issubdtype(self.timestamps.dtype, np.datetime64):
            raise TypeError("timestamps must be datetime64 dtype")

        # Performing missing / invalid dimensions check
        if len(self.timestamps) != self.channel_matrix.shape[0]:
            raise ValueError("The number of timestamps must match the number of pollutant measurements.")
        if self.channel_names is None:
            raise ValueError("The channel matrix names must be provided.")
        if len(self.channel_names) != self.channel_matrix.shape[1]:
            raise ValueError("The number of channel names must match the number of channels")

        # Timestamp Equidistance Calculation
        t_diffs_ns = np.diff(self.timestamps).astype('timedelta64[ns]').astype(np.int64)
        t_diffs_median = np.median(t_diffs_ns)
        max_deviation = np.max(np.abs(t_diffs_ns - t_diffs_median))

        # AI-Assisted: Claude <Opus 4.6> ; (Usage of the warning package to display warnings to the user)
        # If the skip_dt_validation flag is set, the code accepts non-equidistant timestamps
        # The Sampling Time is estimated from the median
        if max_deviation / 1e6 > self.timestamp_tolerance_ms:
            if self.skip_dt_validation:
                warnings.warn(
                    f"Timestamp equidistance check skipped. "
                    f"Max deviation: {max_deviation / 1e6:.3f} ms "
                    f"(tolerance: {self.timestamp_tolerance_ms:.3f} ms). "
                    f"dt will be estimated from median. ({self.dt})  ",
                    UserWarning,
                    stacklevel=2
                )
            else:
                raise ValueError(
                    f"Timestamps must be equidistant. "
                    f"Max deviation: {max_deviation / 1e6:.3f} ms "
                    f"(tolerance: {self.timestamp_tolerance_ms:.3f} ms)."
                )


    @property
    def dt(self) -> float:
        """ Returns the median time difference between timestamps in seconds (Sampling Time)."""
        return float(np.median(np.diff(self.timestamps)/np.timedelta64(1, 's')))

    @property
    def source_day(self)-> str:
        """Returns the measurement day as string in format "datetime64[D]" derived from the first timestamp."""
        return str(self.timestamps[0].astype("datetime64[D]"))

    def get_channel_data_by_name(self, name: str) -> np.ndarray:
        """ Returns data of the named channel

        :param name: (str) Name of the channel
        :return: (np.ndarray) Data from the channel
        """

        if name not in self.channel_names:
            raise KeyError(
                f"Channel {name!r} not found. Available channels: {self.channel_names}"
            )
        idx = self.channel_names.index(name)
        return self.channel_matrix[:, idx]



    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, vehicle_pass_times: np.ndarray, channels: list[str] | None = None,
                       skip_dt_validation: bool = False ) -> MeasurementRegister:
        """ Returns an object of :class:`MeasurementRegister` from a pandas DataFrame

        This factory method takes a pandas DataFrame of the MeasurementData and a numpy array of the vehicle pass times.
        Optionally, a list of channel names can be provided to select specific channels. Conversions to numpy arrays are
        performed internally.

        :param df: (pd.DataFrame) DataFrame of the MeasurementData including the time series at column index 0 !
        :param vehicle_pass_times: (np.ndarray) Array of the vehicle pass times Light Barrier Trigger times
        :param channels:(list[str] | None) List of channel names to select; None -> all Channels are selected
        :param skip_dt_validation: (bool) Skip the timestamp equidistance validation, takes the mean

        :return: MeasurementRegister
        """

        # Type Checking
        if not isinstance(vehicle_pass_times, np.ndarray):
            raise TypeError("vehicle_pass_times must be np.ndarray")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df (Measurement Data) must be pd.DataFrame")

        # Extracts Timestamps, expects them at column index 0 in datetime format
        timestamps = pd.to_datetime(df.iloc[:,0]).to_numpy(dtype="datetime64[ns]")

        # Channel Selection
        if channels is not None:
            # AI-Assisted: Claude <Opus 4.6> ; (Notify the user if and what channel is missing)
            try:
                channel_matrix = df[channels].to_numpy(dtype=np.float64)
                channel_names = channels
            except KeyError:
                missing = set(channels) - set(df.columns)
                raise ValueError(f"One or more channels not found in the dataframe: {missing}")
        else:
            channel_matrix = df.iloc[:,1:].to_numpy(dtype=np.float64)
            channel_names = df.columns[1:].tolist()

        return cls(timestamps=timestamps,channel_matrix=channel_matrix,channel_names= channel_names,
                   vehicle_pass_times=vehicle_pass_times,skip_dt_validation=skip_dt_validation)