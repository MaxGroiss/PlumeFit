import numpy as np


def zero_baseline_start(centered_normalized_matrix: np.ndarray, n_start: int = 3) -> np.ndarray:
    """Shift each plume so its leading background sits at zero.

    Subtracts a per-plume offset (median of the first n_start samples)
    from the whole plume.

    :param centered_normalized_matrix: (np.ndarray) Normalized plumes.
    :param n_start: (int) Samples for offset calculation should be a small value.
    :returns: (np.ndarray) Shifted plume matrix
    """
    offset = np.nanmedian(centered_normalized_matrix[:, :n_start], axis=1)
    return centered_normalized_matrix - offset[:, None]


def normalize_area(centered_matrix: np.ndarray, dt: float, channel_name = "", day = "" ) -> np.ndarray:
    """Scale each plume so its integral (sum × dt) equals 1.


    :param centered_matrix: (np.ndarray) Centered plumes.
    :param dt: (float) Sampling interval in seconds.
    :param day:
    :param channel_name:
    :returns: (np.ndarray) Area-normalized plume matrix.
    :raises ValueError: If any plume has zero area.
    """
    areas = (np.sum(centered_matrix, axis=1, keepdims=True) * dt)
    if np.any(areas <= 0):
        raise ValueError(f"{channel_name} on {day} : Zero Emission area detected")

    return centered_matrix / areas

def positive_area_mask(centered_matrix: np.ndarray, dt: float) -> np.ndarray:
    """True for every plume whose area after the baseline offset (zero_baseline_start) is > 0.

    Exactly the area normalize_plumes divides by -> plumes with False cannot be normalized.
    """
    return np.sum(zero_baseline_start(centered_matrix), axis=1) * dt > 0

def normalize_plumes(centered_matrix: np.ndarray, dt: float, channel_name: str = "", day: str = "") -> np.ndarray:
    # To keep the order consistent
    return normalize_area(zero_baseline_start(centered_matrix), dt, channel_name=channel_name, day=day)