"""Status labels assigned to every plume during extraction.

Every isolated plume ends with exactly one status. ExtractionResult.qa_counts counts
them per status, so the QA statistics show which check removed how many plumes.
"""

from enum import Enum

class PlumeStatus(Enum):
    """Quality label of an extracted plume.

    Attributes:
        VALID: Passed all checks, enters the template.
        NULL_DATA: Window contains NaN or a faulty recording (persistently below
            min_physical_value).
        NO_PEAK: No peak above h_min, CO₂ peak at the edge of the search window, or
            pollutant peak outside the band around the CO₂ peak.
        MULTIPLE_PEAKS: More than one prominent peak in the window (likely overlap).
        TAIL_ANOMALY: CO₂ tail re-rises more than noise alone explains (R > R_krit).
        WINDOW_EDGE: Cutout window exceeds the segment bounds.
        NON_PLAUSIBLE_AREA: After zeroing the peak height is ≤ 0 or the effective
            width A / h is below min_effective_width.
    """

    VALID = "valid"
    NULL_DATA = "null_data"
    NO_PEAK = "no_peak"
    MULTIPLE_PEAKS = "multiple_peaks"
    TAIL_ANOMALY = "tail_anomaly"
    WINDOW_EDGE = "window_edge"
    NON_PLAUSIBLE_AREA = "non_plausible_area"
