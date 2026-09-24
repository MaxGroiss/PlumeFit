"""Beispiel für thesis_plot - synthetische Daten, zur Kontrolle des Stils nach Anpassungen."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import implementations.plotting.template.plotting_template as tp

tp.setup(figures_dir="../plt_figures", textwidth_mm=160)

# --- synthetisches Signal: langsame Drift + Rauschen + einzelne Peaks --------
rng = np.random.default_rng(1)
dt = 0.5
t = np.arange(0, 600, dt)
drift = 420 + 6 * np.sin(2 * np.pi * t / 900)
events = np.array([60, 150, 162, 171, 290, 420, 432, 520])
x = drift + rng.normal(0, 1.5, t.size)
for e in events:
    s = t - e - 3
    x += np.where(s > 0, 40 * (s / 3) * np.exp(1 - s / 3), 0)

# Basislinie über rollierendes 2. Perzentil
base = pd.Series(x).rolling(int(100 / dt), center=True, min_periods=1).quantile(0.02).to_numpy()

# --- Bild 1: volle Breite, flach ------------------------------------------
fig, ax = plt.subplots(figsize=tp.figsize("wide"))
ax.plot(t, x, color=tp.C["data"], lw=0.7, label=r"Messsignal $x(t)$")
ax.plot(t, base, color=tp.C["reference"], lw=1.4, label=r"Basislinie $\hat{b}(t)$")
tp.mark_events(ax, events)
ax.set_xlabel(tp.label("t", "s"))
ax.set_ylabel(tp.label("x", "ppm", name="Konzentration"))
ax.legend(loc="upper right")
tp.save(fig, "beispiel_wide")

# --- Bild 2: zwei Subplots nebeneinander ----------------------------------
fig, axs = plt.subplots(1, 2, figsize=tp.figsize("single", aspect=0.38), sharey=True)
r = x - base
axs[0].plot(t, r, color=tp.C["derived"], lw=0.7)
axs[0].set_xlim(130, 200)
axs[0].set_xlabel(tp.label("t", "s"))
axs[0].set_ylabel(tp.label("r", "ppm"))
axs[1].hist(r[r < 8], bins=40, orientation="horizontal", color=tp.C["context"])
axs[1].set_xlabel(tp.label("Anzahl", math=False))
tp.subplot_labels(axs)
tp.save(fig, "beispiel_subplots")
