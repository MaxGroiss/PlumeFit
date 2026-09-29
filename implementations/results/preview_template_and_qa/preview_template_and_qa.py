# This file contains the code that lead to the results in "Extrahierte Vorlagen und QA-Statistik"
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from implementations.plotting.template import plotting_template as tp

from src.plume_template_extraction.plume_status import PlumeStatus
from src.shared_services.measurement_register import MeasurementRegister
from src.plume_template_extraction.extraction_config import ExtractionConfig, ChannelQAConfig
from src.plume_template_extraction.extraction_result import BatchResult, CombinedResult
from src.plume_template_extraction.template_extraction import run_batch

# ----------------------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parents[3]
MERGED_DATA = ROOT_DIR / "campaign_data" / "Merged_data"
PASS_TIMES_CSV = ROOT_DIR / "campaign_data" / "CARES_Milan_Madre_Cabrini_TUG_emission_ratios_allemissions_co2_4_80_ppm_3s.csv"

# Days with a lot of faulty data can be excluded here (e.g. "2021-10-11")
EXCLUDE_DAYS = ["2021-10-11"]

# ----------------------------------------------------------------------------- configuration
# Everything except min_physical_value stays at the defaults -> noise-derived thresholds (None) are used
CO2_QA = ChannelQAConfig(min_physical_value=5.0)
BC_QA = ChannelQAConfig(min_physical_value=1.0)
PN_QA = ChannelQAConfig(min_physical_value=1000.0)
NO2_QA = ChannelQAConfig(min_physical_value=3.8)
NOX_QA = ChannelQAConfig(min_physical_value=4.0)

# One entry per instrument pair: CO2 reference channel, pollutant channel, pollutant QA
PAIRS = {
    "BCT1": dict(co2="CO2_TUG_BCT1", poll="BC_TUG_BCT1", qa=BC_QA),
    "BCT2": dict(co2="CO2_TUG_BCT2", poll="BC_TUG_BCT2", qa=BC_QA),
    "PN":   dict(co2="CO2_TUG_BCT2", poll="PN_TUG_DC",   qa=PN_QA),
    "NO2":  dict(co2="CO2_ICAD_1",   poll="NO2_ICAD",    qa=NO2_QA),
    "NOX":  dict(co2="CO2_ICAD_2",   poll="NOX_ICAD",    qa=NOX_QA),
}

def build_configs(pairs: dict[str, dict]) -> list[ExtractionConfig]:
    """One ExtractionConfig per pair, extraction parameters at their defaults (shared across all pairs)."""
    return [ExtractionConfig(co2_channel=p["co2"], poll_channel=p["poll"],
                             co2_qa=CO2_QA, pollutant_qa=p["qa"])
            for p in pairs.values()]

# ----------------------------------------------------------------------------- data loading
def load_registers(exclude_days: set[str] = EXCLUDE_DAYS) -> list[MeasurementRegister]:
    """Reads the light-barrier triggers and the merged instrument data -> one MeasurementRegister per day."""
    pass_times = pd.read_csv(PASS_TIMES_CSV)
    pass_times["Timestamp"] = pd.to_datetime(pass_times["Timestamp"])
    days = [d for d in pass_times["Timestamp"].dt.date.unique() if str(d) not in exclude_days]
    print(f"Days excluded from input data: {sorted(exclude_days)}")

    registers = []
    for day in days:
        data_path = MERGED_DATA / f"CARES_Milan_MadreCabrini_instrument_data_time_aligned_{day}.csv"
        if not data_path.exists():
            print(f"  {day}: no merged data -> skipped")
            continue
        lb_triggers = (pass_times.loc[pass_times["Timestamp"].dt.date == day, "Timestamp"]
                       .to_numpy(dtype="datetime64[ns]"))
        # skip_dt_validation=False would raise if the sampling intervals differ more than allowed
        registers.append(MeasurementRegister.from_dataframe(pd.read_csv(data_path), lb_triggers,
                                                            skip_dt_validation=True))
    print(f"Loaded {len(registers)} measurement days")
    return registers

# ----------------------------------------------------------------------------- extraction
def run_extraction(registers: list[MeasurementRegister],
                   configs: list[ExtractionConfig]) -> BatchResult:
    """Runs every config on every day."""
    return run_batch([(reg, configs) for reg in registers])



# ----------------------------------------------------------------------------- file specific

def get_qa_statistics(batch: BatchResult, out_path: Path | None = None, per_day: bool = False) -> pd.DataFrame:
    """ Creates a .csv from the qa summary of a batch extraction"""
    rows = []
    for channel, results in batch.grouped_by_channel().items():
        for r in results:
            is_poll = r.config.poll_channel == channel
            rows.append({"channel": channel,
                         "reference": r.config.co2_channel if is_poll else "",
                         "day": r.source_day,
                         "n_isolated": r.n_isolated,
                         **{s.value: r.qa_counts[s] for s in PlumeStatus},
                         "n_template": r.n_valid})

    table = pd.DataFrame(rows)
    if not per_day:
        table = (table.drop(columns="day")
                 .groupby(["channel", "reference"], as_index=False, sort=False).sum())

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_path, index=False)
    return table

#---------------------------- Channel Names for Display
DISPLAY = {
    "CO2_TUG_BCT1": r"CO$_2$ (BCT1)", "CO2_TUG_BCT2": r"CO$_2$ (BCT2)",
    "CO2_ICAD_1":   r"CO$_2$ (ICAD 1)", "CO2_ICAD_2": r"CO$_2$ (ICAD 2)",
    "BC_TUG_BCT1":  "BC (BCT1)", "BC_TUG_BCT2": "BC (BCT2)",
    "PN_TUG_DC":    "PN (DC)", "NO2_ICAD": r"NO$_2$ (ICAD)", "NOX_ICAD": r"NO$_x$ (ICAD)",
}
Y_LABEL = tp.label("s", r"s^{-1}", name="normierte Vorlage")

def _plot_single(ax, cr: CombinedResult, max_plumes: int | None, seed: int) -> None:
    t = cr.time_axis
    rows = cr.normalized_matrix
    if max_plumes is not None and rows.shape[0] > max_plumes:
        rows = rows[np.random.default_rng(seed).choice(rows.shape[0], max_plumes, replace=False)]
    ax.plot(t, rows.T, color=tp.C["context"], lw=0.3, alpha=0.3, rasterized=True)
    ax.plot(t, cr.mean_shape, color=tp.C["model"], lw=1.2, label="Vorlage")
    ax.set_title(f"{DISPLAY.get(cr.channel, cr.channel)}, $n = {cr.n_valid}$")
    ax.set_xlabel(tp.label("t", "s"))

def plot_representative_templates(combined: dict[str, CombinedResult], co2_channel: str, poll_channel: str,
                                  name: str, max_plumes: int | None = 500, seed: int = 0) -> None:

    fig, axes = plt.subplots(1, 2, figsize=tp.figsize("single", aspect=0.45), sharex=True)
    for ax, ch in zip(axes, (co2_channel, poll_channel)):
        _plot_single(ax, combined[ch], max_plumes, seed)
    axes[0].set_ylabel(Y_LABEL)
    axes[0].legend(loc="upper right")
    tp.subplot_labels(axes)
    tp.save(fig, name)
    plt.close(fig)


def plot_template_overlay(combined: dict[str, CombinedResult], channels: list[str], name: str) -> None:
    """Mean shapes of several channels in one axis (all area normalized, peak at t = 0)."""
    fig, ax = plt.subplots(figsize=tp.figsize("single", aspect=0.5))
    for ch in channels:
        cr = combined[ch]
        ax.plot(cr.time_axis, cr.mean_shape, label=f"{DISPLAY.get(ch, ch)} ($n = {cr.n_valid}$)")
    ax.set_xlabel(tp.label("t", "s"))
    ax.set_ylabel(Y_LABEL)
    ax.legend(loc="upper right")
    tp.save(fig, name)
    plt.close(fig)

if __name__ == "__main__":
    registers = load_registers()
    batch = run_extraction(registers, build_configs(PAIRS))
    combined = batch.combined_by_channel()   # channel -> CombinedResult (input for the template plots)
    export_path =  Path(__file__).parent / "result" / "extraction"
    for ch, cr in combined.items():
        cr.to_csv(path=export_path / f"{ch}.csv",normalized=False)
        print(f"  {ch:14s}: {cr.n_valid:5d} valid plumes")

    # File Specifics
    qa = get_qa_statistics(batch, Path(__file__).parent / "result" / "qa_summary.csv")
    tp.setup(figures_dir=Path(__file__).parent / "result" / "figures", textwidth_mm=160)

    co2_channels = sorted({p["co2"] for p in PAIRS.values()})
    poll_channels = [p["poll"] for p in PAIRS.values()]

    plot_representative_templates(combined, "CO2_TUG_BCT1", "BC_TUG_BCT1", "template_representative")
    plot_template_overlay(combined, co2_channels, "templates_co2")
    plot_template_overlay(combined, poll_channels, "templates_pollutant")
