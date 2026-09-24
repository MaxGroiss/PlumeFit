from operator import eq
from pathlib import Path

import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
from fontTools.misc.cython import returns

import implementations.plotting.template.plotting_template as tp
tp.setup(figures_dir="../plt_figures", textwidth_mm=160)

from src.shared_services.noise_and_background import compute_background_series

IMPL_DIR = Path(__file__).resolve().parents[2]
DATA = IMPL_DIR / "data" / "NOX" / "isolated_neighbors.csv"

data = pandas.read_csv(DATA)
sample_channel = data["CO2_ICAD_2"]
sample_channel_poll = data["NOX_ICAD"]
timeseries = np.arange(sample_channel.shape[0])
events = timeseries[data["lb_trigger"].to_numpy()]
background = compute_background_series(sample_channel.to_numpy(),2.0,100,0,10)
background_poll = compute_background_series(sample_channel_poll.to_numpy(),2.0,100,0,10)
print(sample_channel)
print(events[[0]])
fig, ax = plt.subplots(figsize=tp.figsize("wide"))
ax.plot(timeseries, sample_channel, color=tp.C["data"], lw=0.7, label=r"Messsignal $x(s)$")
ax.plot(timeseries, background, color=tp.C["reference"], lw=1.4, label=r"Baseline $\hat{b}(s)$")
tp.mark_events(ax, events[[0]], color=tp.C["marker"], label=tp.label("LB-Trigger"))
tp.mark_events(ax, events[1:], color=tp.C["marker"])
ax.set_xlabel(tp.label("s", "", name="Samples"))
ax.set_ylabel(tp.label("x(s)", "", name="Konzentrationssignal"))
ax.legend(loc="upper right")
tp.save(fig, "rolling_filter/isolated_neighbors")

fig, ax = plt.subplots(figsize=tp.figsize("wide"))
ax.plot(timeseries, sample_channel_poll, color=tp.C["data"], lw=0.7, label=r"Messsignal $x(s)$")
ax.plot(timeseries, background_poll, color=tp.C["reference"], lw=1.4, label=r"Baseline $\hat{b}(s)$")
tp.mark_events(ax, events[[0]], color=tp.C["marker"], label=tp.label("LB-Trigger"))
tp.mark_events(ax, events[1:], color=tp.C["marker"])
ax.set_xlabel(tp.label("s", "", name="Samples"))
ax.set_ylabel(tp.label("x(s)", "", name="Konzentrationssignal Poll"))
ax.legend(loc="upper right")
tp.save(fig, "rolling_filter/isolated_neighbors_poll")