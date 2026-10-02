"""This module contains the status flags that are assigned to every plume during extraction

After the Extraction these flags help to identify what qa or guard marked a plume as invalid

"""

from enum import Enum

class PlumeStatus(Enum):
    """Quality label assigned to an extracted plume.

    Attributes:
        VALID: Passes all quality checks.
        NULL_DATA: Contains values below the physical minimum.
        NO_PEAK: Peak height above background is below threshold.
        MULTIPLE_PEAKS: More than one prominent peak detected.
        TAIL_ANOMALY: Sustained re-rise in the tail region.
        WINDOW_EDGE: Cutout window exceeds the measurement data bounds.
        NON_PLAUSIBLE_AREA: Peak Area can't be normalized due to negative area.
    """

    VALID = "valid"
    NULL_DATA = "null_data"
    NO_PEAK = "no_peak"
    MULTIPLE_PEAKS = "multiple_peaks"
    TAIL_ANOMALY = "tail_anomaly"
    WINDOW_EDGE = "window_edge"
    NON_PLAUSIBLE_AREA = "non_plausible_area"
