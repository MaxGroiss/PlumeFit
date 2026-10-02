import numpy as np


def zero_baseline_start(centered_normalized_matrix: np.ndarray, n_anchor: int) -> np.ndarray:
    """Shifts each plume so the mean of its first n_anchor samples sits at zero"""
    offset = np.nanmean(centered_normalized_matrix[:, :n_anchor], axis=1)
    return centered_normalized_matrix - offset[:, None]

def area_plausibility_check(centered_matrix: np.ndarray, dt: float, n_anchor: int,
                            peak_index: int, min_width_s: float) -> np.ndarray:
    """Retruns true if the window area after the baseline shift is at least min_width_s * peak_height
       This is introduced, because if the samples used for zeroing sit on a devaying tail of a neighbor they can
       push the tail of the current plume below zero the net area of the plume goes near zero and the unit-area
       normalization (Value of Sample / A Near Zero ) -> Peak "explodes".
    """
    z = zero_baseline_start(centered_matrix, n_anchor)
    area = np.sum(z, axis=1)*dt
    height = z[:, peak_index]
    return (height > 0) & (area >= min_width_s * height)

def normalize_plumes(centered_matrix: np.ndarray, dt: float, n_anchor: int,
                     channel_name: str = "", day: str = "") -> tuple[np.ndarray, np.ndarray]:
    """ Shifts the Baseline and normalizes to unit area"""
    z = zero_baseline_start(centered_matrix, n_anchor)
    areas = np.sum(z, axis=1)*dt
    if np.any(areas <= 0):
        raise ValueError(f"{channel_name} on {day}: non-positive emission area")
    return z / areas[:, None], areas

def pooled_mean_shape(normalized_matrix: np.ndarray, areas: np.ndarray, dt: float) -> np.ndarray:
    """Calculates the area weighted mean (sum of plumes / sum of areas)"""
    mean = np.average(normalized_matrix, axis=0, weights=areas)
    return mean / (np.sum(mean) * dt)

def pooled_mean_se(normalized_matrix: np.ndarray, areas: np.ndarray, dt: float,
                   ) -> np.ndarray:
    """standard error of the pooled mean Gatz & Smith (1995) 10.1016/1352-2310(94)00210-C"""
    n = areas.size
    if n <= 1:
        print(f" Warning ! Pooled mean standard error calculation on {n} passes. Expected at least n > 1")
        return 0
    w = areas[:, None]
    w_b = np.mean(areas)
    x_w = pooled_mean_shape(normalized_matrix, areas, dt)
    wx_dev = w * normalized_matrix - w_b * x_w
    w_dev = w - w_b
    var = n / ((n - 1) * np.sum(areas) ** 2) * (
            np.sum(wx_dev ** 2, axis=0)
            - 2 * x_w * np.sum(w_dev * wx_dev, axis=0)
            + x_w ** 2 * np.sum(w_dev ** 2, axis=0))
    # Guarding against potentially negative var
    se = np.sqrt(np.maximum(var,0))
    return se
