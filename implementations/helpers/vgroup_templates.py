"""Used to create Vehicle Group representative templates

Takes in CombinedResult -> Extraction Result .csv and the vehicle mapping .csv
Passes are joined using source_day and pass_index  (can be changed in KEY)

Allows grouping by selecting key filters:
column -> value with
    scalar  -> equal            {"unece_category": "M1"}
    list    -> one of (OR)      {"fuel_group": ["Diesel", "Petrol"]}
    tuple   -> range min..max   {"euro_stage": (5, 6)}

The filters are combined by an AND operation, so this would yield vehicles that are category M1, Diesel or Petrol
that fall into the Euro Classes 5 to 6
"""

from pathlib import Path
import pandas as pd

def filter_plumes(plumes_csv: Path, mapping_csv: Path, g_filter: dict, out_csv) -> pd.DataFrame:
    plumes = pd.read_csv(plumes_csv, dtype={"source_day": str})
    mapping = pd.read_csv(mapping_csv, parse_dates=["lb_time"], low_memory=False)
    mapping["source_day"] = mapping["lb_time"].dt.strftime("%Y-%m-%d")
    mapping = mapping.rename(columns={"er_day_index": "pass_index"})

    vehicles = plumes[["source_day", "pass_index"]].merge(mapping, on=["source_day", "pass_index"], how="left")
    # Not matched Plumes hold no information for template grouping
    mask = vehicles["match_status"].eq("matched")
    for col, value in g_filter.items():
        if isinstance(value, tuple):  # (min, max) -> range, both bounds included
            lo, hi = value
            mask &= vehicles[col].between(lo, hi)
        elif isinstance(value, list):  # [a, b, ...] -> one of (OR)
            mask &= vehicles[col].isin(value)
        else:  # scalar -> equal
            mask &= vehicles[col].eq(value)

    selected = plumes[mask.to_numpy()]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_csv, index=False, float_format="%.6g")
    return selected