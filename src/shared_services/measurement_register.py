"""Input data structure of both PlumeFit pipelines.

MeasurementRegister holds the channel signals and the light barrier triggers of one
measurement segment (in the CARES Milan campaign: one measurement day). It keeps the
pipelines independent of the input file format: every data source only needs a
converter into this structure, e.g. MeasurementRegister.from_dataframe.
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Assistance with Docstring writing -> mainly latex equation to Unicode symbol equations)

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd


def as_samples(td: np.timedelta64, dt: float) -> int:
    """Convert a duration into a number of samples.

    Args:
        td: Duration.
        dt: Sampling interval [s].

    Returns:
        Number of samples, rounded to the nearest integer.
    """
    return round(td / np.timedelta64(1, "s") / dt)



@dataclass
class MeasurementRegister:
    """Channel signals and light barrier triggers of one measurement segment.

    The pipelines address samples by index and convert time windows with a single
    sampling interval dt, so the timestamps must be (nearly) equidistant. This is
    validated on construction. Use from_dataframe to build a register from a
    pandas DataFrame.

    Attributes:
        timestamps: Sample times of the channel signals, datetime64, sorted ascending.
        channel_matrix: Channel signals, one column per channel (CO₂ and pollutants).
        channel_names: Name per column of channel_matrix.
        vehicle_pass_times: Light barrier trigger time per vehicle pass, datetime64, sorted ascending.
         Must use the same time reference as timestamps.
        timestamp_tolerance_ms: Maximum allowed deviation of a single sampling interval from the median interval.
        skip_dt_validation: If True, a violated tolerance only warns and dt is taken from the median interval.
    Raises:
        TypeError: If timestamps or vehicle_pass_times are not datetime64.
        ValueError: If the dimensions of timestamps, channels and names disagree,
            or if the timestamps are not equidistant and skip_dt_validation is False.
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
        """Sampling interval Δt [s], median of all timestamp differences."""
        return float(np.median(np.diff(self.timestamps)/np.timedelta64(1, 's')))

    @property
    def source_day(self)-> str:
        """Date of the first timestamp as "YYYY-MM-DD".

        Together with a pass index it identifies a vehicle pass across segments.
        """
        return str(self.timestamps[0].astype("datetime64[D]"))

    def get_channel_data_by_name(self, name: str) -> np.ndarray:
        """Return the signal of one channel.

        Args:
            name: Channel name.

        Returns:
            Channel signal.

        Raises:
            KeyError: If the channel does not exist.
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
        """Build a register from a pandas DataFrame of one measurement segment.

        Args:
            df: Measurement data. Column 0 must hold the timestamps, the other columns the channels.
            vehicle_pass_times: Light barrier trigger time per vehicle pass, datetime64, same time reference as the
                timestamps in df.
            channels: Columns to take over. None takes all columns after column 0.
            skip_dt_validation: If True, a violated tolerance only warns and dt is taken from the median interval.

        Returns:
            The register of the segment.

        Raises:
            TypeError: If df is not a DataFrame or vehicle_pass_times is not an ndarray.
            ValueError: If a requested channel is missing in df.
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