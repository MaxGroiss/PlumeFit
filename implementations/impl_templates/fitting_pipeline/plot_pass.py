"""Optional: fit figure of one light barrier pass in every channel, to inspect a fit or tune its parameters.

Reads only the run folder (no fit is done here; the run needs SAVE_DIAGNOSTICS = True) and is the only part of PlumeFit
that needs matplotlib:
    passes/<day>/<channel>.csv                  the rows of the pass
    peak_windows.csv, run_config.json           CO₂ window center and dt: trigger = τ̂ − shift − center
    diagnostics/<day>.zip                       signal, segment model and column contributions per channel

One panel per channel (CO₂ channel, then its pollutants): measured signal, baseline, contribution of every column on top
of the baseline (the pass colored, neighbors grey), fit, trigger line (black), fitted peaks of other passes (grey lines).

Output: results/<run>/pass_plots/pass_<day>_<pass_index>.png

Usage: python -m implementations.impl_templates.fitting_pipeline.plot_pass 2024-05-13 12
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5.5> ; (Fit figure of one pass in every channel, from the run tables)
# AI-Assisted: <Opus 5.5> ; (Rework for the public release: plain matplotlib, reads the zipped diagnostics)

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from implementations.impl_templates.fitting_pipeline.fitting_template import RESULTS_DIR

from implementations.helpers.run_dirs import run_dir
from implementations.helpers.run_tables import load_passes

# ============================================================================== SETTINGS (adjust here)
RUN: str | None = None            # fitting run, e.g. "2026-10-18_162952"; None: the newest
WINDOW_S = (-15.0, 40.0)          # plot window around the light barrier trigger of the pass, in s

# ============================================================================== AUTOMATIC (nothing to adjust below)

def channel_order(rows: pd.DataFrame) -> list[str]:
    """Channels of the pass: every CO₂ channel followed by the pollutants coupled to it."""
    order = []
    for co2 in rows.loc[rows["role"] == "co2", "channel"]:
        order.append(co2)
        order += list(rows.loc[rows["reference_channel"] == co2, "channel"])
    return order


def trigger_sample(rows: pd.DataFrame, centers: pd.Series, dt: float) -> int:
    """Light barrier trigger of the pass in samples: τ_j⁽⁰⁾ − center = τ̂_j − shift_j − center (a CO₂ row)."""
    co2 = rows[(rows["role"] == "co2") & rows["position"].notna()]
    if co2.empty:
        raise ValueError("the pass lies outside the signal in every CO₂ channel, nothing to plot")
    co2 = co2.iloc[0]
    return int(round(co2["position"] - co2["shift_s"] / dt - centers[co2["channel"]]))


def channel_curves(archive: zipfile.ZipFile, channel: str, segment_id: float, lo: int, hi: int):
    """Signal, model and column contributions of one channel in the sample range [lo, hi).

    Returns:
        signal: Background-subtracted signal per sample (index lo … hi-1).
        model: Fitted model of the segment of the pass (only its samples inside the window, empty without segment).
        columns: Pass indices of a column → its contribution a · s(τ̂) on the samples of model.
    """
    with archive.open(f"{channel}_curves.csv.gz") as f:
        curves = pd.read_csv(f, compression="gzip", dtype={"curve": str})
    curves = curves[(curves["sample"] >= lo) & (curves["sample"] < hi)]
    signal = curves[curves["segment"] == -1].set_index("sample")["value"].reindex(range(lo, hi))
    segment = curves[curves["segment"] == segment_id]
    model = segment[segment["curve"] == "model"].set_index("sample")["value"]
    columns = {}
    for name, c in segment[segment["curve"] != "model"].groupby("curve"):
        members = tuple(int(i) for i in name.split(";"))
        columns[members] = c.set_index("sample")["value"].reindex(model.index, fill_value=0.0)
    return signal, model, columns


def describe(row: pd.Series) -> str:
    """Fit result of one channel as text: a ± SE, detection, group, QA flags."""
    if row["status"] != "fitted":
        return row["status"]
    if row["group_size"] > 1:
        text = f"merged with {int(row['group_size']) - 1} other pass(es) in group {int(row['group_id'])}"
        if np.isfinite(row["group_amplitude"]):             # the area is stored only at the first member
            text += f", group a = {row['group_amplitude']:.3g}"
    else:
        text = f"a = {row['amplitude']:.3g} ± {row['se']:.2g}"
    text += ", detected" if row["detected"] else ", not detected"
    return text + (f"\n{row['qa_flags']}" if row["qa_flags"] else "")


def plot_pass(result_dir: Path, source_day: str, pass_index: int) -> Path:
    """One panel per channel for one pass, saved to result_dir/pass_plots/. Returns the file."""
    passes = load_passes(result_dir, source_day)
    rows = passes[passes["pass_index"] == pass_index].set_index("channel", drop=False)
    if rows.empty:
        raise KeyError(f"pass ({source_day}, {pass_index}) is not in the results of {result_dir}")
    channels = channel_order(rows)
    dt = json.loads((result_dir / "run_config.json").read_text(encoding="utf-8"))["dt_s"]
    centers = pd.read_csv(result_dir / "peak_windows.csv", index_col="channel")["center"]
    trigger = trigger_sample(rows, centers, dt)
    lo, hi = trigger + int(WINDOW_S[0] / dt), trigger + int(WINDOW_S[1] / dt)

    fig, axes = plt.subplots(len(channels), 1, figsize=(8, 2.2 * len(channels)), sharex=True, squeeze=False)
    with zipfile.ZipFile(result_dir / "diagnostics" / f"{source_day}.zip") as archive:
        for ax, ch in zip(axes[:, 0], channels):
            row = rows.loc[ch]
            signal, model, columns = channel_curves(archive, ch, row["segment_id"], lo, hi)
            ax.plot((signal.index - trigger) * dt, signal, color="0.45", lw=0.8, label="Measured")
            if not model.empty:
                x = (model.index - trigger) * dt
                baseline = model - sum(columns.values(), pd.Series(0.0, index=model.index))
                for members, values in columns.items():
                    own = pass_index in members
                    ax.fill_between(x, baseline, baseline + values, lw=0, alpha=0.4 if own else 0.25,
                                    color="tab:blue" if own else "0.6")
                ax.plot(x, baseline, color="k", lw=0.9, ls="--", label="Baseline")
                ax.plot(x, model, color="tab:red", lw=1.1, label="Fit")
            others = passes[(passes["channel"] == ch) & (passes["pass_index"] != pass_index)
                            & passes["position"].between(lo, hi)]
            for position in others["position"]:
                ax.axvline((position - trigger) * dt, color="0.6", lw=0.7, ls=":")
            ax.axvline(0.0, color="k", lw=0.7, ls=":")
            ax.text(0.99, 0.95, describe(row), transform=ax.transAxes, ha="right", va="top", fontsize="x-small")
            ax.set_ylabel(ch, fontsize="small")
    axes[0, 0].legend(loc="upper left", fontsize="x-small")
    axes[0, 0].set_title(f"{source_day} pass {pass_index}", fontsize="small", loc="left")
    axes[-1, 0].set_xlabel("Time after the light barrier trigger in s")
    fig.tight_layout()
    out = result_dir / "pass_plots" / f"pass_{source_day}_{pass_index}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m implementations.impl_templates.fitting_pipeline.plot_pass <YYYY-MM-DD> <pass_index>")
    print(plot_pass(run_dir(RESULTS_DIR, RUN), sys.argv[1], int(sys.argv[2])))
