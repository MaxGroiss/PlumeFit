# This file contains code created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
#  AI-Assisted: <OPUS 5> ; (Individual Passes in Template Plot limited for svg performance -> seed so they are chosen
#  randomly -> This only applies visually the template contains all passes)
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from src.plume_template_extraction.extraction_result import CombinedResult
# Change / Delete for normal plt plotting style
from implementations.plotting.template import plotting_template as tp

# Directory where the grouped csv folders are located
GROUP_CSV_DIRECTORY = Path(__file__).parent / "output"
FIGURE_DIRECTORY = GROUP_CSV_DIRECTORY / "figures"

# Instrument pairs: CO2 reference channel, pollutant channel
PAIRS = {
    "BCT1": ("CO2_TUG_BCT1", "BC_TUG_BCT1"),
    "BCT2": ("CO2_TUG_BCT2", "BC_TUG_BCT2"),
    "PN":   ("CO2_TUG_BCT2", "PN_TUG_DC"),
    "NO2":  ("CO2_ICAD_1",   "NO2_ICAD"),
    "NOX":  ("CO2_ICAD_2",   "NOX_ICAD"),
}
# Mapping fot LaTeX typing
DISPLAY = {
    "CO2_TUG_BCT1": r"CO$_2$ (BCT1)", "CO2_TUG_BCT2": r"CO$_2$ (BCT2)",
    "CO2_ICAD_1":   r"CO$_2$ (ICAD 1)", "CO2_ICAD_2": r"CO$_2$ (ICAD 2)",
    "BC_TUG_BCT1":  "BC (BCT1)", "BC_TUG_BCT2": "BC (BCT2)",
    "PN_TUG_DC":    "PN (DC)", "NO2_ICAD": r"NO$_2$ (ICAD)", "NOX_ICAD": r"NO$_x$ (ICAD)",
}
# Y_Label for the PLot if the template is used
Y_LABEL = tp.label("s", r"s^{-1}", name="normierte Vorlage")
# Plots a Template with individual passes as semitrans lines and template full opacity
def _plot_single(ax, cr: CombinedResult, max_plumes: int | None, seed: int) -> None:
    t = cr.time_axis
    rows = cr.normalized_matrix
    if max_plumes is not None and rows.shape[0] > max_plumes:
        rows = rows[np.random.default_rng(seed).choice(rows.shape[0], max_plumes, replace=False)]
    ax.plot(t, rows.T, color=tp.C["context"], lw=0.3, alpha=0.3, rasterized=True)
    ax.plot(t, cr.mean_shape, color=tp.C["model"], lw=1.2, label="Vorlage")
    ax.set_title(f"{DISPLAY.get(cr.channel, cr.channel)}, $n = {cr.n_valid}$")
    ax.set_xlabel(tp.label("t", "s"))

def load_combined_from_folder(folder: Path) -> dict[str, CombinedResult]:
    """Reads every plume CSV of a folder -> {channel: CombinedResult}, same structure as batch.combined_by_channel()."""
    combined = {}
    for csv in sorted(folder.glob("*.csv")):
        try:
            cr = CombinedResult.from_csv(csv)
        except ValueError as e:                 # e.g. empty group after filtering
            print(f"  skipped {csv.name}: {e}")
            continue
        if cr.channel in combined:
            raise ValueError(f"Channel {cr.channel} occurs twice in {folder}")
        combined[cr.channel] = cr
    return combined

def filter_text(folder: Path) -> str:
    """Group filter from filter.json as one readable line, e.g. 'vehicle_category = Passenger cars, euro_stage = 5..6'."""
    # AI-Assisted: <Opus 5> ; (Regex)
    path = folder / "filter.json"
    if not path.exists():
        return folder.name
    flt = json.loads(path.read_text(encoding="utf-8"))
    return ", ".join(f"{k} = {v[0]}..{v[1]}" if isinstance(v, list) and len(v) == 2 and k.endswith(("stage", "kg", "_m", "v"))
                     else f"{k} = {v}" for k, v in flt.items())

def plot_representative_templates(combined: dict[str, CombinedResult], co2_channel: str, poll_channel: str,
                                  name: str, group: str | None = None,
                                  max_plumes: int | None = 500, seed: int = 0) -> None:
    fig, axes = plt.subplots(1, 2, figsize=tp.figsize("single", aspect=0.45), sharex=True)
    for ax, ch in zip(axes, (co2_channel, poll_channel)):
        _plot_single(ax, combined[ch], max_plumes, seed)
    axes[0].set_ylabel(Y_LABEL)
    axes[0].legend(loc="upper right")
    tp.subplot_labels(axes)
    if group:
        fig.suptitle(group, fontsize="small")
    tp.save(fig, name)
    plt.close(fig)

def plot_group(folder: Path, show_group: bool = True) -> None:
    """Representative template plot for every complete instrument pair of one group folder."""
    combined = load_combined_from_folder(folder)
    group = f"{folder.name}: {filter_text(folder)}" if show_group else None
    for pair, (co2, poll) in PAIRS.items():
        if co2 in combined and poll in combined:
            plot_representative_templates(combined, co2, poll, f"template_{folder.name}_{pair}", group)
        else:
            print(f"  {folder.name}: pair {pair} not complete -> no plot")

def plot_group_comparison(group_names: list[str], pair: str, name: str,
                          labels: dict[str, str] | None = None, show_sem: bool = True) -> None:
    #  AI-Assisted: <OPUS 5> ; (Plotting Loop)
    """Templates (mean shapes) of several groups on top of each other: CO2 left, pollutant right.

    group_names: folder names in GROUP_DIRECTORY, e.g. ["Pkw_Exhaust_left", "Pkw_Exhaust_right"]
    pair:        key of PAIRS, e.g. "BCT1"
    labels:      optional legend names per group, e.g. {"Pkw_Exhaust_left": "Auspuff links"}
    show_sem:    shaded band = standard error of the mean (sigma / sqrt(n))
    """
    co2, poll = PAIRS[pair]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=tp.figsize("single", aspect=0.45), sharex=True)

    for i, group in enumerate(group_names):
        combined = load_combined_from_folder(GROUP_CSV_DIRECTORY / group)
        label = (labels or {}).get(group, group)
        color = colors[i % len(colors)]                      # same color for a group in both panels
        for ax, ch in zip(axes, (co2, poll)):
            if ch not in combined:
                print(f"  {group}: no {ch} -> not in comparison")
                continue
            cr = combined[ch]
            ax.plot(cr.time_axis, cr.mean_shape, color=color, lw=1.2, label=f"{label} ($n = {cr.n_valid}$)")
            if show_sem:
                ax.fill_between(cr.time_axis, cr.mean_shape - cr.std_mean_envelope,
                                cr.mean_shape + cr.std_mean_envelope, color=color, alpha=0.2, lw=0)

    for ax, ch in zip(axes, (co2, poll)):
        ax.set_title(DISPLAY.get(ch, ch))
        ax.set_xlabel(tp.label("t", "s"))
        if ax.get_legend_handles_labels()[0]:                # only if something was plotted
            ax.legend(loc="upper right", fontsize="small")
    axes[0].set_ylabel(Y_LABEL)
    tp.subplot_labels(axes)
    tp.save(fig, name)
    plt.close(fig)

# For COMPARISONS plots
COMPARISONS = {
        "exhaust":        (["Pkw_Exhaust_left", "Pkw_Exhaust_right"], {"Pkw_Exhaust_left": "Auspuff links",
                                                                         "Pkw_Exhaust_right": "Auspuff rechts"}),
        "exhaust_petrol": (["Pkw_Petrol_left", "Pkw_Petrol_right"], {"Pkw_Petrol_left": "Benzin, links",
                                                                       "Pkw_Petrol_right": "Benzin, rechts"}),
        "category":       (["LTV", "HDV_Buses", "L_all"], {"LTV": "LTV", "HDV_Buses": "HDV und Busse",
                                                            "L_all": "L-Fahrzeuge"}),
        "diesel_euro":    ([f"Pkw_Diesel_Euro{n}" for n in (4, 5, 6)], {f"Pkw_Diesel_Euro{n}": f"Diesel Euro {n}"
                                                                         for n in (4, 5, 6)}),
        }


if __name__ == "__main__":
    # Creates Plots for all Groups in the Output folder
    plot_output_dir = FIGURE_DIRECTORY / "comparisons"

    tp.setup(figures_dir=plot_output_dir, textwidth_mm=160)
    # -> Template Plot per Group and Channel
    #for folder in sorted(p for p in GROUP_CSV_DIRECTORY.iterdir() if p.is_dir() and p != FIGURE_DIRECTORY):
    #    plot_group(folder)
    # -> Comp Plots
    for comp, (groups, labels) in COMPARISONS.items():
        for pair in PAIRS:
            plot_group_comparison(groups, pair, f"compare_{comp}_{pair}", labels)