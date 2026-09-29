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