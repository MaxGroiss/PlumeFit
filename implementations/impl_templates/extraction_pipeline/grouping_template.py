"""Creates the plume CSVs of vehicle groups, the fitting builds its templates from them.

Input:  the plume CSVs of one extraction run (results/<run>/extraction) and metadata/vehicle_mapping.csv
Output: results/<run>/grouping/<grouping run>/<group>/<channel>.csv + filter.json, group_summary.csv

The vehicle mapping needs source_day and pass_index (see data_needed_for_extraction/README.md), every other column is
only needed if a filter in GROUPS uses it, so GROUPS has to be adapted to the vehicle data of your campaign.
Optional column match_status: if it exists, only passes with the value "matched" are grouped (also for All_included).

A filter maps column -> value:
    scalar  -> equal            {"duty_class": "LDV"}
    list    -> one of (OR)      {"fuel_group": ["Diesel", "Petrol"]}
    tuple   -> range min..max   {"euro_stage": (5, 6)}

A lot of filter combinations yield a small number of vehicles, which is not suited for a template: groups below
GROUP_MIN_N are skipped.
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5.5> ; (Generalization for the public release, campaign specific columns and groups removed)

from pathlib import Path
import pandas as pd
import json

from implementations.impl_templates.extraction_pipeline.extraction_template import RESULTS_DIR, VEHICLE_MAPPING_CSV

from implementations.helpers.run_dirs import new_run_dir, run_dir

# ============================================================================== SETTINGS (adjust here)
EXTRACTION_RUN: str | None = None   # extraction run to group, e.g. "2026-10-18_153000"; None: the newest

# Groups: name -> filter, the columns must exist in the vehicle mapping
GROUPS = {
    "LDV":          {"duty_class": "LDV"},
    "Diesel":       {"fuel_group": "Diesel"},
    "Petrol":       {"fuel_group": "Petrol"},
    "LDV_Euro_5_6": {"duty_class": "LDV", "euro_stage": (5, 6)},
}

# Groups below this number of plumes are not written. The number needed for a valid template is re checked in the
# fitting (fitting_template.MIN_N, default 30), a reduced n keeps a group for inspection even if it is not populated enough
GROUP_MIN_N = 10

# ============================================================================== AUTOMATIC (nothing to adjust below)
KEY = ["source_day", "pass_index"]  # -> For Mapping Vehicle Data to Extraction Data
# Fallback template of the fitting (fitting_template.FALLBACK_GROUP), always written
ALL_INCLUDED = {
    "All_included": {}
}


def load_mapping(path: Path) -> pd.DataFrame:
    """Loads the vehicle mapping, source_day stays text like in the plume CSVs."""
    return pd.read_csv(path, dtype={"source_day": str}, low_memory=False)

def filter_mask(vehicles: pd.DataFrame, flt: dict) -> pd.Series:
    """A mask with all vehicles matching the filter set to true"""
    mask = (vehicles["match_status"].eq("matched") if "match_status" in vehicles
            else pd.Series(True, index=vehicles.index))
    for col, value in flt.items():
        if isinstance(value, tuple):
            mask &= vehicles[col].between(*value)
        elif isinstance(value, list):
            mask &= vehicles[col].isin(value)
        else:
            mask &= vehicles[col].eq(value)
    return mask

def create_group_csvs(groups: dict[str, dict], extraction_dir: Path, mapping_csv: Path, output_dir: Path, min_n: int = 30) -> pd.DataFrame:
    """ Creates the grouped csv outputs at output_dir/<group>/<channel>.csv for every group and channel
        min_n allows to pre-filter groups / certain channels when grouped this way to avoid creating a template that is based on < min_n passes
        Returns the number of plumes per group and channel
    """
    mapping = load_mapping(mapping_csv)
    counts = {}
    for csv in sorted(extraction_dir.glob("*.csv")):
        plumes = pd.read_csv(csv, dtype={"source_day": str})
        vehicles = plumes[KEY].merge(mapping, on=KEY, how="left", validate="m:1")  # same row order as plumes
        for name, flt in groups.items():
            selected = plumes[filter_mask(vehicles, flt).to_numpy()]
            counts[(name, csv.stem)] = len(selected)
            if len(selected) >= min_n:
                out = output_dir / name / csv.name
                out.parent.mkdir(parents=True, exist_ok=True)
                selected.to_csv(out, index=False, float_format="%.6g")
                # Keeps the definition of the group next to its plumes
                with open(out.parent / "filter.json", "w", encoding="utf-8") as f:
                    json.dump(flt, f, ensure_ascii=False)
    return pd.Series(counts).unstack()


if __name__ == "__main__":
    extraction_run = run_dir(RESULTS_DIR, EXTRACTION_RUN)
    output_directory = new_run_dir(extraction_run / "grouping")
    print(f"Grouping {extraction_run.name} -> {output_directory}")
    summary = create_group_csvs(ALL_INCLUDED | GROUPS, extraction_run / "extraction", VEHICLE_MAPPING_CSV, output_directory, min_n=GROUP_MIN_N)
    summary.to_csv(output_directory / "group_summary.csv")
    print(summary.to_string())
    # The Output .csv from this script can be directly used to create templates by from_csv in CombinedResult
