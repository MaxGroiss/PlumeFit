# This file implements a demo to create representative templates based on vehicle groups (input pre created csv export)

from pathlib import Path
import pandas as pd
import json
# Possible Filter Categories Supported by the mapping csv:
# Filter Keys : Exact Expression
# Alot of filter Combinations yield a small amount of vehicles -> not really suited for template creation can be pre filtered in create_group_csvs by min_n
#Allows grouping by selecting key filters:
#column -> value with
#    scalar  -> equal            {"unece_category": "M1"}
#    list    -> one of (OR)      {"fuel_group": ["Diesel", "Petrol"]}
#    tuple   -> range min..max   {"euro_stage": (5, 6)}
"""
duty_class: "L", "LDV", "HDV"
vehicle_category: "Passenger cars", "Light commercial vehicles", "L-type vehicles", "Heavy duty, buses", "Other types, not defined"
vehicle_subcategory: "Passenger cars (M1)", "Light commercial vehicles (N1)", "Motorcycles (L3)", "Mopeds (L1, L2)", "Heavy duty (N2, N3)", "Quadricycles (L6, L7)", "Sidecars, tricycles (L4, L5)", "Buses (M2, M3)"
unece_category: "M1", "M1G", "N1", "N1G", "N2", "N3", "N3G", "M2", "M3", "L1", "L3", "L5", "L6", "L7"
fuel_group: "Petrol", "Diesel", "CNG, LPG, bi-fuel", "Electric", "Other types, not defined"
hybrid: True, False -> Bool not a string
euro_class: "Euro 0" ... "Euro 6", "Euro II" ... "Euro VI"
euro_class_detail: "Euro 0", "Euro 1", "Euro 2", "Euro 3", "Euro 4", "Euro 5", "Euro 5a", "Euro 5b", "Euro 6a", "Euro 6b", "Euro 6c", "Euro 6d-TEMP", "Euro 6d-TEMP-ISC", "Euro 6d", "Euro 6d-ISC", "Euro II", "Euro III", "Euro IV", "Euro V", "Euro V-B", "Euro VI-A", "Euro VI-B", "Euro VI-C", "Euro VI-D"
# This is assumed by manufacturer region -> HGV, L-type, electric not applicable by Knoll et al. DOI: 10.1021/acs.est.5c05015
exhaust_side_assumed: "left", "right", "not applicable", "unknown"
manufacturer_region: "Europe", "Asia", "North America", "unknown"
dpf: "S" (Yes (Italien Base Data has si as yes) ), "N" (No, Not Known)
manufacturer: -> Take a look into vehicle_mapping.csv
model: -> Take a look into vehicle_mapping.csv
weight_consistent: True, False -> Bool not a string -> A HGV under 3.5 Tons is questionable (Checks the weight against the vehicle class)
------------------------------------------------------------------------------------------------------------------------
# Some numerical keys:
euro_stage: 0.0 ... 6.0 -> "euro_stage": (5, 6) -> Same for HDV here also arabic numeric
gross_weight_kg: Weight of the vehicle in kg for example: (0, 3500) -> vehicles up to 3,5 t
v: Velocity when crossing the light barrier  in m/s for example: (5, 15)
lb_length_m: Vehicle length in m according to the light barrier data for example: (3.5, 5.0)
"""

# Path Definitions:
OUTPUT_DIRECTORY = Path(__file__).parent / "output"
INPUT_DIRECTORY = Path(__file__).parent / "data"
EXTRACTION_INPUT_DIR = INPUT_DIRECTORY / "extraction"
VEHICLE_MAPPING = INPUT_DIRECTORY / "vehicle_data" / "vehicle_mapping_madre_cabrini.csv"

KEY = ["source_day", "pass_index"] # -> For Mapping Vehicle Data to Extraction Data


def load_mapping(path: Path) -> pd.DataFrame:
    """Loads the mapping and renames the column used for concat fia KEY"""
    mapping = pd.read_csv(path, parse_dates=["lb_time"], low_memory=False)
    mapping["source_day"] = mapping["lb_time"].dt.strftime("%Y-%m-%d")
    return mapping.rename(columns={"er_day_index": "pass_index"})
# Not LB/ANPR matched vehicles are not getting grouped
def filter_mask(vehicles: pd.DataFrame, flt: dict) -> pd.Series:
    """A mask with all vehicles matching the filter set to true"""
    mask = vehicles["match_status"].eq("matched")
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
        vehicles = plumes[KEY].merge(mapping, on=KEY, how="left")  # same row order as plumes
        for name, flt in groups.items():
            selected = plumes[filter_mask(vehicles, flt).to_numpy()]
            counts[(name, csv.stem)] = len(selected)
            if len(selected) >= min_n:
                out = output_dir / name / csv.name
                out.parent.mkdir(parents=True, exist_ok=True)
                selected.to_csv(out, index=False, float_format="%.6g")
                # Keeps the mapping for later plotting (naming)
                with open(out.parent / "filter.json", "w", encoding="utf-8") as f:
                    json.dump(flt, f, ensure_ascii=False)
    return pd.Series(counts).unstack()


# Filter Definition
# ----------------------------------------------------------------
GROUPS = {
    "All_included": {},
    # Exhaust Side
    # One Dict Entry creates one CSV for every channel
    # ----
    "Pkw_Exhaust_left":  {"vehicle_category": "Passenger cars", "exhaust_side_assumed": "left"},
    # ----
    "Pkw_Exhaust_right": {"vehicle_category": "Passenger cars", "exhaust_side_assumed": "right"},
    "Pkw_Petrol_left":   {"vehicle_category": "Passenger cars", "fuel_group": "Petrol", "exhaust_side_assumed": "left"},
    "Pkw_Petrol_right":  {"vehicle_category": "Passenger cars", "fuel_group": "Petrol", "exhaust_side_assumed": "right"},
    # Vehicle Categories
    "LTV":       {"duty_class": "LDV"},
    "HGV":       {"vehicle_subcategory": "Heavy duty (N2, N3)"},
    "HDV_Buses": {"vehicle_category": "Heavy duty, buses"},
    "Motorcycle": {"vehicle_subcategory": "Motorcycles (L3)"},
    "L_all":     {"vehicle_category": "L-type vehicles"},
    # Euro classes of passenger cars
    **{f"Pkw_{fuel}_Euro{n}": {"vehicle_category": "Passenger cars", "fuel_group": fuel, "euro_stage": (n, n)}
       for fuel in ("Petrol", "Diesel") for n in (2, 3, 4, 5, 6)},
}
ALL_INCLUDED = {
    "All_included":{}
}

if __name__ == "__main__":
    summary = create_group_csvs(GROUPS, EXTRACTION_INPUT_DIR, VEHICLE_MAPPING, OUTPUT_DIRECTORY, min_n=0)
    summary.to_csv(OUTPUT_DIRECTORY / "group_summary.csv")
    print(summary.to_string())
    # The Output .csv from this script can be directly used to create templates by from_csv in CombinedResult
    # for creating templates for a fit check ....
