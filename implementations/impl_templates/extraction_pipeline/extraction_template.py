"""Code template for the extraction pipeline of PlumeFit

Inputs (data_needed_for_extraction/, columns see its README.md):
    merged_data/<MERGED_FILE>               instrument data, one CSV per day
    metadata/light_barrier_triggers.csv     one row per light barrier pass (Timestamp)
    metadata/vehicle_mapping.csv            vehicle data per pass, only for grouping_template.py

Outputs (a new run folder per run, see implementations/helpers/run_dirs.py):
    results/<run>/extraction/<channel>.csv  plumes per channel, key (source_day, pass_index) for the vehicle mapping
    console                                 tuning overview (helpers/console_output.py): QA shares, tail level at the
                                            window end, peak positions

Next steps: grouping_template.py (vehicle groups) -> copy_needed_data_to_fitting.py -> fitting_pipeline.
The paths, channels and QA values below belong to the synthetic example data, adjust them to your campaign.
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5.5> ; (Generalization for the public release, campaign specific values removed)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.shared_services.measurement_register import MeasurementRegister
from src.plume_template_extraction.extraction_config import ExtractionConfig, ChannelQAConfig
from src.plume_template_extraction.extraction_result import BatchResult
from src.plume_template_extraction.template_extraction import run_batch

from implementations.helpers.run_dirs import new_run_dir
from implementations.helpers.console_output import print_extraction_overview

# ============================================================================== SETTINGS (adjust here)
# ----------------------------------------------------------------------------- paths
# Paths may need adjustment if not used in standard directory structure of PlumeFit
DATA_DIR = Path(__file__).parent / "data_needed_for_extraction" # Structure and columns see the README.md in the directory
MERGED_DATA = DATA_DIR / "merged_data"
MERGED_FILE = "merged_{day}.csv"  # file name per day
PASS_TIMES_CSV = DATA_DIR / "metadata" / "light_barrier_triggers.csv"
VEHICLE_MAPPING_CSV = DATA_DIR / "metadata" / "vehicle_mapping.csv"
RESULTS_DIR = Path(__file__).parent / "results"  # one subfolder per run
# Days with a lot of faulty data can be excluded here (e.g. "2024-05-14")
EXCLUDE_DAYS: list[str] = []
# Days whose data become faulty from a certain time on: the data are cut off there, the light barrier
# triggers are kept (isolation and pass_index stay unchanged, windows beyond the cut are dropped by the bounds check)
TRUNCATE_DAYS: dict[str, str] = {}  # e.g. {"2024-05-14": "07:58:00"}
# ----------------------------------------------------------------------------- configuration
# Everything except min_physical_value stays at the defaults -> noise-derived thresholds (None) are used
# min_physical_value: lowest plausible value incl. background in the channel unit, below it = faulty recording
CO2_QA = ChannelQAConfig(min_physical_value=5.0)
BC_QA = ChannelQAConfig(min_physical_value=0.1)
NOX_QA = ChannelQAConfig(min_physical_value=4.0)

# Window parameters for testing the effect of changed windows
WINDOWS = dict(min_gap=np.timedelta64(30, "s"), window_after_peak=np.timedelta64(20, "s"))

# One entry per instrument pair: CO2 reference channel, pollutant channel, pollutant QA
# The channel names are the column names of the merged data. Several pairs may share one CO2 channel.
PAIRS = {
    "BC":  dict(co2="CO2_A", poll="BC_A",  qa=BC_QA),
    "NOX": dict(co2="CO2_B", poll="NOX_B", qa=NOX_QA),
}

# ----------------------------------------------------------------------------- console output
PRINT_OVERVIEW = True   # tuning overview of the run, see helpers/console_output.py
# Template of the overview: area-weighted mean (True) or median of the area-normalised plumes (False)
AREA_WEIGHTING = True

# ============================================================================== AUTOMATIC (nothing to adjust below)
# ----------------------------------------------------------------------------- data loading
def merged_data_path(day) -> Path:
    """Merged instrument data CSV of one day (MERGED_FILE in MERGED_DATA)."""
    return MERGED_DATA / MERGED_FILE.format(day=day)

def build_configs(pairs: dict[str, dict]) -> list[ExtractionConfig]:
    """One ExtractionConfig per pair, window parameters from WINDOWS (shared across all pairs)."""
    return [ExtractionConfig(co2_channel=p["co2"], poll_channel=p["poll"],
                             co2_qa=CO2_QA, pollutant_qa=p["qa"], **WINDOWS)
            for p in pairs.values()]

def load_registers(exclude_days: list[str] = EXCLUDE_DAYS) -> list[MeasurementRegister]:
    """Reads the light-barrier triggers and the merged instrument data -> one MeasurementRegister per day."""
    pass_times = pd.read_csv(PASS_TIMES_CSV)
    pass_times["Timestamp"] = pd.to_datetime(pass_times["Timestamp"])
    days = [d for d in pass_times["Timestamp"].dt.date.unique() if str(d) not in exclude_days]
    print(f"Days excluded from input data: {sorted(exclude_days)}")

    registers = []
    for day in days:
        data_path = merged_data_path(day)
        if not data_path.exists():
            print(f"  {day}: no merged data -> skipped")
            continue
        lb_triggers = (pass_times.loc[pass_times["Timestamp"].dt.date == day, "Timestamp"]
                       .to_numpy(dtype="datetime64[ns]"))
        # AI-Assisted: <Opus 5.5> ; (Report duplicate light barrier triggers)
        # Identical trigger times give identical template columns: the fit can only merge them into one group
        # (one area for both passes), the extraction never treats them as isolated
        dup = np.flatnonzero(np.diff(lb_triggers) == np.timedelta64(0, "ns"))
        if dup.size:
            print(f"  {day}: {dup.size} duplicate light barrier trigger(s), pass_index "
                  + ", ".join(f"{i}/{i + 1}" for i in dup))
        # AI-Assisted: <Opus 5.5> ; (Data truncation)
        data = pd.read_csv(data_path)
        if str(day) in TRUNCATE_DAYS:
            cut = pd.Timestamp(f"{day} {TRUNCATE_DAYS[str(day)]}")
            data = data[pd.to_datetime(data.iloc[:, 0]) < cut].reset_index(drop=True)
            print(f"  {day}: data cut at {TRUNCATE_DAYS[str(day)]}")
        # skip_dt_validation=False would raise if the sampling intervals differ more than allowed
        registers.append(MeasurementRegister.from_dataframe(data, lb_triggers, skip_dt_validation=True))
    print(f"Loaded {len(registers)} measurement days")
    return registers

# ----------------------------------------------------------------------------- extraction
def run_extraction(registers: list[MeasurementRegister],
                   configs: list[ExtractionConfig]) -> BatchResult:
    """Runs every config on every day."""
    return run_batch([(reg, configs) for reg in registers])

if __name__ == "__main__":
    registers = load_registers()
    batch = run_extraction(registers, build_configs(PAIRS))
    combined = batch.combined_by_channel()
    RESULT_DIR = new_run_dir(RESULTS_DIR)
    print(f"Results -> {RESULT_DIR}")
    export_path = RESULT_DIR / "extraction"
    for ch, cr in combined.items():
        # Export the Results to a CSV per Channel
        cr.to_csv(path=export_path / f"{ch}.csv", normalized=False)

    if PRINT_OVERVIEW:
        print_extraction_overview(batch, combined, PAIRS, AREA_WEIGHTING)
