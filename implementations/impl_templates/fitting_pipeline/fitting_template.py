"""Code template for the fitting pipeline (Mode L) of PlumeFit

Every setting of a fit is in the SETTINGS section below (1–7), the functions after it only build the setups and
write the results. Nothing else has to be changed to run or configure a fit.

Steps:
    1. SETTINGS: paths, days and pairs, template groups, peak window quantiles, refinement per channel type, fit
       parameters (ModeLConfig, every field written out), QA checks and non-detect policy, outputs.
    2. build_setups: one PairSetup per instrument pair of extraction_template.PAIRS (peak windows and templates
       from the extraction results, one call each).
    3. Loop over the days: fit_pairs_day fits every CO₂ channel once and every pollutant coupled to the refined
       CO₂ peaks of the same pass (see src/mode_linear_fitting/pair_fitting.py).
    4. Output per day: one CSV per channel and, with SAVE_DIAGNOSTICS, one zip with the diagnostics (helpers/run_tables.py).

Template assignment (GroupTemplates):
    SPLIT_KEY = None    every pass gets the FALLBACK_GROUP template of its channel
    SPLIT_KEY = column  value of this vehicle mapping column -> group in TEMPLATE_GROUPS; passes without value,
                        with a value not in TEMPLATE_GROUPS or a group below MIN_N get the fallback
    TEMPLATE_SHAPE      statistic every template is built with (area-weighted mean or pointwise median),
    TEMPLATE_SHAPE_BY_CHANNEL overrides it per channel

The group folders (grouped plume CSVs of extraction_pipeline/grouping_template.py) and the extraction CSVs are copied
into data_needed_for_fitting/ by extraction_pipeline/copy_needed_data_to_fitting.py. Merged data and metadata are
directly accessed from the extraction template because they don't change per export.

Not set here: the channels of a pair and min_physical_value per channel come from extraction_template (PAIRS,
CO2_QA), so the fit always uses the same channels and faulty-recording limits as the extraction.

Outputs (a new run folder per run, see implementations/helpers/run_dirs.py, columns see helpers/run_tables.py):
    results/<run>/run_config.json               every setting of this run (and the input runs)
    results/<run>/peak_windows.csv              peak windows per channel (sample and s)
    results/<run>/passes/<day>/<channel>.csv    one row per pass: fit fields and coupling, no vehicle data
    results/<run>/diagnostics/<day>.zip         with SAVE_DIAGNOSTICS: <channel>_segments.csv.gz, <channel>_curves.csv.gz

Figure of a single pass from these files: plot_pass.py.
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5.5> ; (Assistance with docstring writing)
# AI-Assisted: <Opus 5.5> ; (Review, Help with including the vehicle mapping for automatic template assignment)
# AI-Assisted: <Opus 5.5> ; (Generalization for the public release, output per day and channel)

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import fields

import numpy as np
import pandas as pd

from src.mode_linear_fitting.fit_checks import (DEFAULT_QA_CHECKS, keep_non_detects)
from src.mode_linear_fitting.fitting_config import ChannelSetup, GroupTemplates, ModeLConfig, PairSetup
from src.mode_linear_fitting.fitting_inputs import ShapeStatistic, median_shape
from src.mode_linear_fitting.pair_fitting import fit_pairs_day, peak_windows_from_extraction
from src.mode_linear_fitting.position_refinement import refine_pollutant_positions, refine_positions
from src.plume_template_extraction.normalization import pooled_mean_shape

from implementations.impl_templates.extraction_pipeline.extraction_template import (PAIRS, CO2_QA, VEHICLE_MAPPING_CSV,
                                                                                    load_registers)
from implementations.impl_templates.extraction_pipeline.grouping_template import KEY, load_mapping

from implementations.helpers.console_output import print_fit_overview, print_peak_windows
from implementations.helpers.run_dirs import new_run_dir
from implementations.helpers.run_tables import (build_passes, channel_results, pass_rows, write_passes,
                                                write_peak_windows, write_run_config, zip_diagnostics)

# ============================================================================== SETTINGS (adjust here)
# ----------------------------------------------------------------------------- 1. paths
DATA_DIR = Path(__file__).parent / "data_needed_for_fitting"   # filled by copy_needed_data_to_fitting.py
TEMPLATE_DIR = DATA_DIR / "templates"                          # grouped plume CSVs, one folder per group
EXTRACTION_DIR = DATA_DIR / "extraction"                       # delay distributions (all extracted plumes)
MAPPING_CSV = VEHICLE_MAPPING_CSV                              # vehicle mapping of the extraction (not copied)
RESULTS_DIR = Path(__file__).parent / "results"                # one subfolder per run

# ----------------------------------------------------------------------------- 2. days to fit and channel pairs
# Excluded and truncated days come from the extraction (extraction_template.EXCLUDE_DAYS / TRUNCATE_DAYS):
# load_registers applies them, so fit and extraction always use the same data
FIT_DAYS: list[str] | None = None           # e.g. ["2024-05-13"]; None: every loaded day
FIT_PAIRS: list[str] | None = None          # keys of extraction_template.PAIRS, e.g. ["BC"]; None: all
                                            # The extraction saves the channel pairing in metadata so it can be
                                            # automatically derived. Channel specific parameters are also derived
                                            # from the extraction metadata through build_setups (e.g. min_physical value)

# ----------------------------------------------------------------------------- 3. templates
SPLIT_KEY: str | None = None                # e.g. "fuel_group"
TEMPLATE_GROUPS: dict[str, str] = {}        # e.g. {"Diesel": "Diesel", "Petrol": "Petrol"}
FALLBACK_GROUP = "All_included"             # template for all passes without a group template
MIN_N = 30                                  # minimum plumes per group and channel
# template statistic: pooled_mean_shape (area-weighted mean) or median_shape (pointwise median, no area weighting)
TEMPLATE_SHAPE: ShapeStatistic = median_shape
# TEMPLATE_SHAPE defines the default for every channel, the following allows changing it per channel
# exceptions per channel, e.g. {"BC_A": pooled_mean_shape}
TEMPLATE_SHAPE_BY_CHANNEL: dict[str, ShapeStatistic] = {}

# ----------------------------------------------------------------------------- 4. peak windows and refinement
DELAY_QUANTILES = (5, 95)                     # search band of the refinement (quantiles of the extraction delays)
REFINE_CO2 = refine_positions                 # CO₂: coordinate descent (thesis appendix, refinement of the placement)
REFINE_POLLUTANT = refine_pollutant_positions # only significant passes move; None: blind at τ̂_CO₂ + offset

# ----------------------------------------------------------------------------- 5. fit parameters
# Shared by all channels; the values are the defaults of ModeLConfig
FIT = ModeLConfig(
    influence_before=np.timedelta64(5, "s"),    # influence window before the expected peak (noise estimate, anchors)
    influence_after=np.timedelta64(15, "s"),    # influence window after the expected peak
    min_anchor_run=np.timedelta64(5, "s"),      # T_A, minimum duration of an anchor run (segment border)
    t_min_linear=np.timedelta64(60, "s"),       # minimum segment duration for a linear baseline
    min_anchor_time=np.timedelta64(5, "s"),     # minimum anchor time for a fitted baseline (twice for linear)
    merge_vif_max=10.0,                         # VIF_max, columns above are merged with a neighbor
    noise_band_sigma=3.0,                       # k₃, anchor noise band half width in σ̂ (also QA residual_high)
    min_physical_run=np.timedelta64(2, "s"),    # minimum duration below min_physical_value = faulty recording
    bg_percentile=2.0,                          # percentile of the rolling background (as in the extraction)
    bg_rolling_window=np.timedelta64(100, "s"), # T_RW, rolling background window (as in the extraction)
    force_anchored_fixed=False,                 # True: fixed baseline in every segment
    refine_sweeps=10,                           # maximum sweeps of the position refinement
    refine_min_dsse=8.0,                        # κ, a move needs ΔSSE > κσ̂²
    detect_sigma=2.0,                           # k_det, detected if a ≥ k_det · SE (see criteria_of_detection)
)
# ----------------------------------------------------------------------------- 6. QA checks and non-detects
QA_CHECKS = DEFAULT_QA_CHECKS                 # QA checks per pass, only flag (see fit_checks), e.g. + (multiple_peaks,)
NON_DETECT = keep_non_detects                 # non-detect policy, e.g. non_detects_to_noise_level (see fit_checks)

# ----------------------------------------------------------------------------- 7. outputs
SAVE_DIAGNOSTICS = True                     # segments and curves per day and channel, input of plot_pass.py

# ============================================================================== AUTOMATIC (nothing to adjust below)

def pass_values() -> dict:
    """SPLIT_KEY value per (source_day, pass_index) from the vehicle mapping, {} without split."""
    if SPLIT_KEY is None:
        return {}
    values = load_mapping(MAPPING_CSV).set_index(KEY)[SPLIT_KEY].dropna()
    return dict(zip(values.index, values))

def channel_templates(channel: str, values: dict) -> GroupTemplates:
    """Template assignment of one channel: TEMPLATE_GROUPS by the SPLIT_KEY value of a pass, else FALLBACK_GROUP.

    Args:
        channel: Channel name (folder entry <group>/<channel>.csv in TEMPLATE_DIR).
        values: SPLIT_KEY value per (source_day, pass_index).
    """
    return GroupTemplates.from_directory(TEMPLATE_DIR, channel, FALLBACK_GROUP, TEMPLATE_GROUPS, values, MIN_N,
                                         TEMPLATE_SHAPE_BY_CHANNEL.get(channel, TEMPLATE_SHAPE))

def build_setups() -> dict[str, PairSetup]:
    """One PairSetup per pair of FIT_PAIRS; a CO₂ channel shared by several pairs gets one ChannelSetup."""
    pairs = {name: p for name, p in PAIRS.items() if FIT_PAIRS is None or name in FIT_PAIRS}
    windows = peak_windows_from_extraction(EXTRACTION_DIR, DELAY_QUANTILES)
    values = pass_values()
    # One ChannelSetup per CO₂ channel; pairs sharing a CO₂ channel get the same object (fitted once per day)
    co2 = {}
    for p in pairs.values():
        ch = p["co2"]
        if ch not in co2:
            co2[ch] = ChannelSetup(ch, CO2_QA.min_physical_value, windows[ch], channel_templates(ch, values),
                                   REFINE_CO2)
    setups = {}
    for name, p in pairs.items():
        pollutant = ChannelSetup(p["poll"], p["qa"].min_physical_value, windows[p["poll"]],
                                 channel_templates(p["poll"], values), REFINE_POLLUTANT)
        setups[name] = PairSetup(name, co2[p["co2"]], pollutant, FIT, QA_CHECKS, NON_DETECT)
    return setups


# AI-Assisted: <Opus 5.5> ; (Settings of the run as run_config.json)
def run_config(setups: dict[str, PairSetup], days: list[str], dt: float) -> dict:
    """Every setting of this run for run_config.json (functions by name, durations in s)."""
    source = DATA_DIR / "source.json"
    return {"source": json.loads(source.read_text()) if source.exists() else None,
            "days": days, "dt_s": dt,
            "pairs": {name: {"co2": s.co2.channel, "pollutant": s.pollutant.channel if s.pollutant else None}
                      for name, s in setups.items()},
            "min_physical_value": {c.channel: c.min_physical_value for s in setups.values()
                                   for c in (s.co2, s.pollutant) if c is not None},
            "templates": {"split_key": SPLIT_KEY, "groups": TEMPLATE_GROUPS, "fallback_group": FALLBACK_GROUP,
                          "min_n": MIN_N, "shape": TEMPLATE_SHAPE, "shape_by_channel": TEMPLATE_SHAPE_BY_CHANNEL},
            "delay_quantiles": DELAY_QUANTILES,
            "refinement": {"co2": REFINE_CO2, "pollutant": REFINE_POLLUTANT},
            "fit": {f.name: getattr(FIT, f.name) for f in fields(FIT)},
            "qa_checks": QA_CHECKS, "non_detect": NON_DETECT}

if __name__ == "__main__":
    setups = build_setups()
    RESULT_DIR = new_run_dir(RESULTS_DIR)
    print(f"Results -> {RESULT_DIR}")
    print_peak_windows(write_peak_windows(RESULT_DIR, setups))

    registers = [r for r in load_registers() if FIT_DAYS is None or r.source_day in FIT_DAYS]
    if FIT_DAYS is not None:
        missing = sorted(set(FIT_DAYS) - {r.source_day for r in registers})
        if missing:
            print(f"  WARNING no data for {', '.join(missing)} -> skipped")
    if not registers:
        raise ValueError(f"No data for {FIT_DAYS}")

    reference = {s.pollutant.channel: s.co2.channel for s in setups.values() if s.pollutant is not None}
    day_tables = []                             # result table of every day, for the fit overview
    for n_day, reg in enumerate(registers, start=1):
        print(f"  fitting {reg.source_day} ({n_day}/{len(registers)})")
        results = channel_results(fit_pairs_day(reg, setups, keep_diagnostics=SAVE_DIAGNOSTICS))
        dt = next(iter(results.values())).dt    # sampling interval in s (the same for every channel)
        passes = build_passes(pd.concat([pass_rows(fit) for fit in results.values()], ignore_index=True), reference)
        write_passes(RESULT_DIR, passes)
        day_tables.append(passes)
        if SAVE_DIAGNOSTICS:
            day_dir = RESULT_DIR / "diagnostics" / reg.source_day
            for fit in results.values():
                fit.to_csv(day_dir)
            zip_diagnostics(day_dir)
    write_run_config(RESULT_DIR, run_config(setups, [r.source_day for r in registers], dt))

    print_fit_overview(pd.concat(day_tables, ignore_index=True))
