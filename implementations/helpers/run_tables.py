"""Result files of one fitting run: the output of the PlumeFit algorithm.

Not a script and nothing to adjust: used by fitting_template.py and plot_pass.py.

Files of a run folder (results/<run>/):
    run_config.json                 settings of the run: fit parameters, templates, refinement, QA checks, pairs, input runs
    peak_windows.csv                peak window per channel in samples and s
    passes/<day>/<channel>.csv      one row per light barrier pass, see the columns below
    diagnostics/<day>.zip           with SAVE_DIAGNOSTICS: per channel the segments and curves of the fit, see
                                    ChannelDayResult.to_csv

Columns of passes/<day>/<channel>.csv (in this order):
    source_day, pass_index              key of a pass, also the key of the vehicle mapping
    channel                             so that several files can simply be concatenated
    role                                "co2" or "pollutant"
    reference_channel                   CO₂ channel the pollutant was coupled to (empty for CO₂ rows)
    status ... template, shift_s        the PassFit fields (see src/mode_linear_fitting/fitting_result.py)
    sigma                               σ̂ of the channel on this day (channel unit)
    segment_id                          start sample of the segment of the pass, key into the diagnostics
                                        (empty without diagnostics or without segment)
    coupled                             pollutant only: the CO₂ pass of the reference channel is fitted
    area_ratio, area_ratio_rel_se       pollutant only: a_poll / a_CO₂ without unit conversion and its relative SE
                                        (delta method, independent instruments), NaN for merged passes
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5.5> ; (Assistance with docstring writing)
# AI-Assisted: <Opus 5.5> ; (Rework for the public release: one CSV per day and channel, zipped diagnostics)

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["source_day", "pass_index"]                 # key of a pass (vehicle mapping, extraction CSVs)
PASS_KEY = KEY + ["channel"]                       # key of a row across several channels


def channel_results(results: dict) -> dict:
    """Every channel result of one day once (a CO₂ channel shared by several pairs is fitted only once).

    Args:
        results: {pair name: PairDayResult} of one day.

    Returns:
        {channel: ChannelDayResult}.
    """
    channels = {}
    for r in results.values():
        channels[r.co2.channel] = r.co2
        if r.pollutant is not None:
            channels[r.pollutant.channel] = r.pollutant
    return channels


def pass_rows(fit) -> pd.DataFrame:
    """One row per pass of one ChannelDayResult: its to_dataframe() plus σ̂ of the day and the segment of the pass."""
    segment = {i: d.start for d in fit.diagnostics or [] for i in d.pass_indices}
    rows = fit.to_dataframe().assign(sigma=fit.sigma)
    rows["segment_id"] = rows["pass_index"].map(segment).astype("Int64")
    return rows


def add_coupling(passes: pd.DataFrame) -> pd.DataFrame:
    """coupled, area_ratio and area_ratio_rel_se for the pollutant rows (CO₂ rows: empty).

    Same definition as PairDayResult.to_dataframe: area_ratio = a_poll / a_CO₂ (no unit conversion),
    area_ratio_rel_se = √((se_poll / a_poll)² + (se_CO₂ / a_CO₂)²).
    """
    ref = passes[KEY + ["channel", "status", "amplitude", "se"]].rename(
        columns={"channel": "reference_channel", "status": "ref_status", "amplitude": "ref_amplitude",
                 "se": "ref_se"})
    out = passes.merge(ref, on=KEY + ["reference_channel"], how="left")
    poll = out["role"] == "pollutant"
    out["coupled"] = (out["ref_status"] == "fitted").where(poll)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["area_ratio"] = (out["amplitude"] / out["ref_amplitude"]).where(poll)
        out["area_ratio_rel_se"] = np.hypot(out["se"] / out["amplitude"],
                                            out["ref_se"] / out["ref_amplitude"]).where(poll)
    return out.drop(columns=["ref_status", "ref_amplitude", "ref_se"])


def build_passes(passes: pd.DataFrame, reference: dict[str, str]) -> pd.DataFrame:
    """The result table from the rows of every channel of one day.

    Args:
        passes: pass_rows of every channel of the day (a CO₂ channel shared by several pairs only once).
        reference: Pollutant channel → its CO₂ channel, e.g. {"BC_A": "CO2_A", ...}.

    Returns:
        One row per pass and channel, columns see module docstring.
    """
    out = passes.reset_index(drop=True)
    out.insert(3, "role", np.where(out["channel"].isin(list(reference)), "pollutant", "co2"))
    out.insert(4, "reference_channel", out["channel"].map(reference))
    return add_coupling(out).sort_values(PASS_KEY, kind="stable").reset_index(drop=True)


def write_passes(folder: Path, passes: pd.DataFrame) -> None:
    """One CSV per day and channel: folder/passes/<day>/<channel>.csv."""
    for (day, channel), table in passes.groupby(["source_day", "channel"]):
        path = Path(folder) / "passes" / day / f"{channel}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format="%.6g")


def write_peak_windows(folder: Path, setups: dict) -> pd.DataFrame:
    """peak_windows.csv: peak window of every fitted channel in samples and s. Returns the table.

    Args:
        folder: Run folder.
        setups: {pair name: PairSetup}.
    """
    channels = ({s.co2.channel: s.co2 for s in setups.values()}
                | {s.pollutant.channel: s.pollutant for s in setups.values() if s.pollutant is not None})
    windows = pd.DataFrame([{"channel": ch, "reference": c.window.reference, "n": c.window.n,
                             "lo": c.window.lo, "center": c.window.center, "hi": c.window.hi,
                             "lo_s": c.window.seconds[0], "center_s": c.window.seconds[1],
                             "hi_s": c.window.seconds[2]} for ch, c in channels.items()]).set_index("channel")
    windows.to_csv(Path(folder) / "peak_windows.csv", float_format="%.4g")
    return windows


def zip_diagnostics(day_dir: Path) -> None:
    """Bundles the diagnostics of one day into <day>.zip next to the folder and removes the folder."""
    shutil.make_archive(str(day_dir), "zip", day_dir)
    shutil.rmtree(day_dir)


# AI-Assisted: <Opus 5.5> ; (Dtype conversion for .json file format)
def _jsonable(value):
    """Settings value as JSON: timedelta64 → seconds, numpy numbers → Python, functions → their name."""
    if isinstance(value, np.timedelta64):
        return float(value / np.timedelta64(1, "s"))
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return getattr(value, "__name__", repr(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_run_config(folder: Path, config: dict) -> None:
    """run_config.json into the run folder."""
    with open(Path(folder) / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(config), f, indent=1, ensure_ascii=False)


def load_passes(folder: Path, source_day: str) -> pd.DataFrame:
    """All channels of one day with the right types (source_day as text, empty qa_flags as "").

    Raises:
        FileNotFoundError: If the run has no results for this day.
    """
    paths = sorted((Path(folder) / "passes" / source_day).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no results for {source_day} in {folder}")
    passes = pd.concat([pd.read_csv(p, dtype={"source_day": str}, low_memory=False) for p in paths], ignore_index=True)
    for col in ("qa_flags", "replaced_by"):
        passes[col] = passes[col].fillna("").astype(str)
    return passes
