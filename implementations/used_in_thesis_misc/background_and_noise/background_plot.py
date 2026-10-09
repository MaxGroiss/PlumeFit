"""Thesis figures for the rolling percentile background estimation b̂(t).

Figure "bg_rolling_percentile":
    (a) Measured CO₂ signal x(t) of a section with isolated passes and the background
        estimate b̂(t) at the extraction defaults (q = 2 %, T_RW = 100 s).
    (b) Residual r(t) = x(t) − b̂(t).
Figure "bg_percentile_variation":
    b̂(t) of the same section for different percentiles q, rolling window unchanged.

The background is computed with compute_background_series from PlumeFit, parametrized
exactly as in template_extraction.extract_plumes (ExtractionConfig → samples), on the whole
measurement day. The section is cut out afterwards, so the rolling window has no edge
effects at the section borders.
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5.5> ; (Plot script: section selection, figure layout)

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from implementations.helpers.plotting_template import plotting_template as tp
from implementations.templates.extraction_template import CO2_QA, MERGED_DATA, PASS_TIMES_CSV
from src.plume_template_extraction.extraction_config import ExtractionConfig
from src.shared_services.measurement_register import MeasurementRegister
from src.shared_services.noise_and_background import compute_background_series

# ----------------------------------------------------------------------------- section
# Quiet morning traffic: isolated passes on a slowly drifting background, no faulty recordings
DAY = "2021-09-28"
SECTION_START = np.datetime64("2021-09-28T06:00:00")
SECTION_DURATION = np.timedelta64(720, "s")
CHANNEL = "CO2_TUG_BCT1"

# Extraction defaults, only min_physical_value set as in the extraction template
CONFIG = ExtractionConfig(co2_channel=CHANNEL, co2_qa=CO2_QA)
# Percentiles of the variation plot (q = 1 ... 5 % practically coincide with the default)
PERCENTILES = (2, 10, 25, 50)

# ----------------------------------------------------------------------------- output
FIGURES_DIR = Path(__file__).parent / "figures"
# Optional: additionally write an SVG copy into the image folder of the thesis
THESIS_PLOT_DIR: Path | None = None   # e.g. Path(r"...\Typst-Arbeiten\Bachelor-Thesis\pic\plots")


def load_day(day: str) -> MeasurementRegister:
    """MeasurementRegister of one measurement day, same input files as the extraction template."""
    pass_times = pd.read_csv(PASS_TIMES_CSV)
    pass_times["Timestamp"] = pd.to_datetime(pass_times["Timestamp"])
    lb_triggers = (pass_times.loc[pass_times["Timestamp"].dt.date.astype(str) == day, "Timestamp"]
                   .to_numpy(dtype="datetime64[ns]"))
    data = pd.read_csv(MERGED_DATA / f"CARES_Milan_MadreCabrini_instrument_data_time_aligned_{day}.csv")
    return MeasurementRegister.from_dataframe(data, lb_triggers, skip_dt_validation=True)


def background(register: MeasurementRegister, config: ExtractionConfig) -> np.ndarray:
    """b̂(t) of the CO₂ channel of the whole day, parametrized as in extract_plumes."""
    dt = register.dt
    return compute_background_series(
        channel=register.get_channel_data_by_name(config.co2_channel),
        percentile=config.bg_percentile,
        rolling_window=config.as_samples(config.bg_rolling_window, dt),
        min_physical_value=config.co2_qa.min_physical_value,
        min_physical_run=config.as_samples(config.co2_qa.min_physical_run, dt))


def section(register: MeasurementRegister) -> tuple[slice, np.ndarray, np.ndarray]:
    """Sample slice of the section, time axis t in s and light barrier triggers in s (relative to the start)."""
    end = SECTION_START + SECTION_DURATION
    i0, i1 = np.searchsorted(register.timestamps, [SECTION_START, end])
    t = (register.timestamps[i0:i1] - register.timestamps[i0]) / np.timedelta64(1, "s")
    passes = register.vehicle_pass_times
    triggers = (passes[(passes >= SECTION_START) & (passes < end)] - register.timestamps[i0]) / np.timedelta64(1, "s")
    return slice(i0, i1), t, triggers


def plot_background_effect(register: MeasurementRegister, name: str = "bg_rolling_percentile") -> None:
    """Figure (a) x(t) with b̂(t) at the default parameters, (b) residual r(t) = x(t) − b̂(t)."""
    sl, t, triggers = section(register)
    x = register.get_channel_data_by_name(CHANNEL)[sl]
    b = background(register, CONFIG)[sl]

    fig, (ax_x, ax_r) = plt.subplots(2, 1, sharex=True, figsize=tp.figsize("single", aspect=0.62),
                                     gridspec_kw=dict(height_ratios=(2, 1)))
    tp.mark_events(ax_x, triggers)
    tp.mark_events(ax_r, triggers)
    ax_x.plot(t, x, color=tp.C["data"], lw=0.7, label=r"Measured signal $x(t)$")
    ax_x.plot(t, b, color=tp.C["reference"], lw=1.4,
              label=rf"Background estimate $\hat{{b}}(t)$, $q = {CONFIG.bg_percentile:g}\,\%$")
    # Proxy entry for the trigger lines
    ax_x.plot([], [], color=tp.C["marker"], lw=0.6, ls=(0, (2, 2)), label="Light barrier trigger")
    ax_x.set_ylabel(tp.label("x", "ppm", name=r"CO$_2$ concentration"))
    ax_x.legend(loc="upper left")

    ax_r.axhline(0, color="black", lw=0.5, zorder=1)
    ax_r.plot(t, x - b, color=tp.C["derived"], lw=0.7)
    ax_r.set_ylabel(tp.label("r", "ppm", name="Residual"))
    ax_r.set_xlabel(tp.label("t", "s"))

    fig.align_ylabels((ax_x, ax_r))
    tp.subplot_labels((ax_x, ax_r))
    tp.save(fig, name)
    plt.close(fig)


def plot_percentile_variation(register: MeasurementRegister, name: str = "bg_percentile_variation") -> None:
    """Figure b̂(t) for the percentiles in PERCENTILES, measured signal as context."""
    sl, t, _ = section(register)
    x = register.get_channel_data_by_name(CHANNEL)[sl]
    # Ordinal parameter -> sequential colormap, low q dark
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.85, len(PERCENTILES)))

    fig, ax = plt.subplots(figsize=tp.figsize("wide"))
    ax.plot(t, x, color=tp.C["context"], lw=0.7, label=r"Measured signal $x(t)$")
    for q, color in zip(PERCENTILES, colors):
        b = background(register, replace(CONFIG, bg_percentile=q))[sl]
        default = " (default)" if q == CONFIG.bg_percentile else ""
        ax.plot(t, b, color=color, lw=1.2, label=rf"$\hat{{b}}(t)$, $q = {q:g}\,\%${default}")
    ax.set_xlabel(tp.label("t", "s"))
    ax.set_ylabel(tp.label("x", "ppm", name=r"CO$_2$ concentration"))
    # Legend above the axes: inside it would cover the plumes
    fig.legend(loc="outside upper center", ncols=3)
    tp.save(fig, name)
    plt.close(fig)


if __name__ == "__main__":
    tp.setup(figures_dir=FIGURES_DIR, extra_dirs=[THESIS_PLOT_DIR] if THESIS_PLOT_DIR else None)
    reg = load_day(DAY)
    rolling_window_s = CONFIG.bg_rolling_window / np.timedelta64(1, "s")
    print(f"{DAY} {CHANNEL}: dt = {reg.dt} s, q = {CONFIG.bg_percentile} %, T_RW = {rolling_window_s:g} s")
    plot_background_effect(reg)
    plot_percentile_variation(reg)