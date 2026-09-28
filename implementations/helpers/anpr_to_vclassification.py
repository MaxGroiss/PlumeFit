"""This file is used to map the LB-Triggers of the CARES Milan to the actual vehicles
LB to TUG_ID mapping is done in pass_to_anpr_mapping.py

The Mapping is needed to create Templates for specific vehicle categories

@article{knoll2023pointsampling,
title={Large-scale automated emission measurement of individual vehicles with point sampling},
author={Markus Knoll and Martin Penz and Hannes Juchem and Christina Schmidt and Denis Pöhler and Alexander Bergmann},
journal={Atmospheric Measurement Techniques},
year={2023},
}

"""
# This file contains code created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# No real data was used for debugging

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEBUG_DIR = Path(__file__).resolve().parent / "results" / "debug"

# Paths
ROOT_DIR = Path(__file__).resolve().parents[2]
CAMPAIGN_DATA = ROOT_DIR / "campaign_data"
ADDITIONAL_DATA = CAMPAIGN_DATA / "vehicle_data"
PASS_LIST_CSV = CAMPAIGN_DATA / "CARES_Milan_Madre_Cabrini_TUG_emission_ratios_allemissions_co2_4_80_ppm_3s.csv"
REFERENCE_CSV = CAMPAIGN_DATA / "CARES_Milan_PS_TUG_EF_04112023.csv"
TECH_DATA_CSV = ADDITIONAL_DATA / "Milan vehicles characteristics_february15.csv"
LB_DIR = ADDITIONAL_DATA / "LB_data_Madre_Cabrini"
# This file was created with AI Assistance  AI-Assisted: <Opus 5.0> ; (Hand Copied all unique manufacturers
# -> get region for exhaust side decision)
MANUFACTURER_REGION_CSV = ADDITIONAL_DATA / "manufacturer_region.csv"

Output = Path(__file__).resolve().parent / "results"
PASS_TUG_MAPPING_CSV = Output / "pass_tug_mapping.csv" # Read
VEHICLE_MAPPING_CSV = Output / "vehicle_mapping_madre_cabrini.csv" #Write
TEMPLATE_GROUPS_CSV = Output / "template_groups_counts.csv" #Write


TECH_COLUMNS = {  #registry column -> short name taken from CARES_Milan_PS_TUG_EF_04112023.csv
    "tug_id": "TUG_ID", "heat_id": "HEAT_ID", "vehicle type": "vehicle_type", "use": "use",
    "manufacturer": "manufacturer", "commercial denomination": "model", "international category": "category_raw",
    "effective fuelling": "fuel_raw", "Euro class": "euro_raw", "Detailed Euro_Class": "euro_detail_raw",
    "engine volume": "engine_volume_ccm", "kw": "power_kw", "Totale weight": "gross_weight_kg",
    "length": "length_mm", "date of first registration": "first_registration", "DPF": "dpf",
    "NEDC CO2 emission": "nedc_co2_gkm", "WLTP CO2 emission": "wltp_co2_gkm",
}

def _to_number(s: pd.Series) -> pd.Series:
    # string to float
    return pd.to_numeric(s.astype(str).str.replace(r"\s", "", regex=True).str.replace(",", "."), errors="coerce")

def load_tech_data() -> pd.DataFrame:
    # Reads the technical vehicle data wit TUG_ID,
    tech = pd.read_csv(TECH_DATA_CSV, sep=";", encoding="utf-8-sig", low_memory=False, dtype=str)
    tech = tech[list(TECH_COLUMNS)].rename(columns=TECH_COLUMNS)
    tech = tech[tech["TUG_ID"].notna()].copy()
    # Converting strings to floats
    for c in ["engine_volume_ccm", "power_kw", "gross_weight_kg", "length_mm", "nedc_co2_gkm", "wltp_co2_gkm"]:
        tech[c] = _to_number(tech[c])
    tech["n_tech_records"] = tech.groupby("TUG_ID")["TUG_ID"].transform("size")
    # Some TUG_IDs are not unique, the row with the least number of entries is dropped
    tech["_filled"] = tech.notna().sum(axis=1)
    tech = tech.sort_values("_filled", ascending=False).drop_duplicates("TUG_ID").drop(columns="_filled")
    tech["has_tech_data"] = tech["manufacturer"].notna() | tech["category_raw"].notna()
    return tech.reset_index(drop=True)

# UNECE-Categories
VEHICLE_CATEGORY_MAPPING = {
    "M1": ("LDV", "Passenger cars", "Passenger cars (M1)"),
    "M1G": ("LDV", "Passenger cars", "Passenger cars (M1)"),
    "N1": ("LDV", "Light commercial vehicles", "Light commercial vehicles (N1)"),
    "N1G": ("LDV", "Light commercial vehicles", "Light commercial vehicles (N1)"),
    "N2": ("HDV", "Heavy duty, buses", "Heavy duty (N2, N3)"),
    "N2G": ("HDV", "Heavy duty, buses", "Heavy duty (N2, N3)"),
    "N3": ("HDV", "Heavy duty, buses", "Heavy duty (N2, N3)"),
    "N3G": ("HDV", "Heavy duty, buses", "Heavy duty (N2, N3)"),
    "M2": ("HDV", "Heavy duty, buses", "Buses (M2, M3)"),
    "M3": ("HDV", "Heavy duty, buses", "Buses (M2, M3)"),
    "L1": ("L", "L-type vehicles", "Mopeds (L1, L2)"),
    "L2": ("L", "L-type vehicles", "Mopeds (L1, L2)"),
    "L3": ("L", "L-type vehicles", "Motorcycles (L3)"),
    "L4": ("L", "L-type vehicles", "Sidecars, tricycles (L4, L5)"),
    "L5": ("L", "L-type vehicles", "Sidecars, tricycles (L4, L5)"),
    "L6": ("L", "L-type vehicles", "Quadricycles (L6, L7)"),
    "L7": ("L", "L-type vehicles", "Quadricycles (L6, L7)"),
}
OTHER = "Other types, not defined"
# Fule Map -> Italien to Fuel grouping used in Knoll et al. 10.5194/amt-17-2481-2024
FUEL_MAP = {
    "GASOL": ("Diesel", False), "IBRIDO GASOLIO/ELETTRICO": ("Diesel", True),
    "BENZ": ("Petrol", False), "IBRIDO BENZINA/ELETTRICO": ("Petrol", True),
    "B/OLIO": ("Petrol", False), "MISCELA": ("Petrol", False),       # two-stroke petrol/oil mixture
    "B/GPL": ("CNG, LPG, bi-fuel", False), "B/MET": ("CNG, LPG, bi-fuel", False),
    "GPL": ("CNG, LPG, bi-fuel", False), "METANO": ("CNG, LPG, bi-fuel", False),
    "B/ETA": ("CNG, LPG, bi-fuel", False),                            # petrol/ethanol flex fuel
    "ELETTR": ("Electric", False),                                    # battery electric, no tailpipe
}

# AI-Assisted: <OPUS 5.0> ; (Help with Regex handling and mapping)
# -> light Duty Euro written with Arabic Numbers, Heavy duty with Roman
ROMAN = {0: "0", 1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
_ROMAN_TO_INT = {v: k for k, v in ROMAN.items()}
_EURO_RE = re.compile(r"^EURO\s*(VI|IV|V|III|II|I|\d)\s*(.*)$")

def parse_euro(euro_raw, detail_raw, duty_class) -> tuple[float, str | None, str | None]:
    # Returns Euro number -> light Duty Euro written with Arabic Numbers, Heavy duty with Roman
    stage, suffix = np.nan, ""
    detail = str(detail_raw).replace(",", "").strip().upper() if isinstance(detail_raw, str) else ""
    m = _EURO_RE.match(detail)
    if m:
        s = m.group(1)
        stage = float(_ROMAN_TO_INT[s]) if s in _ROMAN_TO_INT and not s.isdigit() else float(s)
        suffix = m.group(2).strip()
    raw = pd.to_numeric(euro_raw, errors="coerce")
    if np.isnan(stage) and not np.isnan(raw):
        stage = float(raw)
    if np.isnan(stage):
        return np.nan, None, None
    num = ROMAN[int(stage)] if duty_class == "HDV" else str(int(stage))
    label = f"Euro {num}"
    if suffix:
        suffix = suffix.lstrip("-") if duty_class != "HDV" else suffix
        if duty_class == "HDV":
            detail_label = f"{label}-{suffix.lstrip('-')}"
        else:
            detail_label = f"{label}{suffix[0].lower()}{suffix[1:]}"
    else:
        detail_label = label
    return stage, label, detail_label

def classify(tech: pd.DataFrame) -> pd.DataFrame:
    """Adds the template classes to the technical data"""
    # AI-Assisted: <OPUS 5.0> ; (Check for mapping errors and help with regex)
    t = tech.copy()
    cat = t["category_raw"].astype(str).str.strip().str.upper()
    mapped = cat.map(VEHICLE_CATEGORY_MAPPING)
    t["unece_category"] = cat.where(mapped.notna())
    t["duty_class"] = mapped.map(lambda x: x[0] if isinstance(x, tuple) else None)
    t["vehicle_category"] = mapped.map(lambda x: x[1] if isinstance(x, tuple) else OTHER)
    t["vehicle_subcategory"] = mapped.map(lambda x: x[2] if isinstance(x, tuple) else OTHER)

    fuel = t["fuel_raw"].astype(str).str.strip().str.upper().map(FUEL_MAP)
    t["fuel_group"] = fuel.map(lambda x: x[0] if isinstance(x, tuple) else OTHER)
    t["hybrid"] = fuel.map(lambda x: x[1] if isinstance(x, tuple) else False)

    euro = [parse_euro(e, d, c) for e, d, c in zip(t["euro_raw"], t["euro_detail_raw"], t["duty_class"])]
    t["euro_stage"] = [e[0] for e in euro]
    t["euro_class"] = [e[1] for e in euro]
    t["euro_class_detail"] = [e[2] for e in euro]

    w = t["gross_weight_kg"]
    t["weight_consistent"] = np.where(t["duty_class"] == "LDV", w.le(3500) | w.isna(),
                                      np.where(t["duty_class"] == "HDV", w.gt(3500) | w.isna(), True))

    # Exhaust side assumption European manufacturers usually have the exhaust on the
    # left, from Asia or the United States -> Europe: left, Asia/North America: right.
    reg = pd.read_csv(MANUFACTURER_REGION_CSV, sep=";")
    t["manufacturer_region"] = t["manufacturer"].map(dict(zip(reg["manufacturer"], reg["region"]))).fillna("unknown")
    side = t["manufacturer_region"].map({"Asia": "right", "North America": "right", "Europe": "left"}).fillna("unknown")
    applicable = (t["duty_class"] == "LDV") & (t["fuel_group"] != "Electric")
    t["exhaust_side_assumed"] = side.where(applicable, "not applicable")
    return t
# Based on CARES_Milan_PS_TUG_EF_04112023.csv
OUTPUT_COLUMNS = [
    "pass_id", "er_day_index", "lb_time", "v", "a", "a_error_missing", "lb_length_m", "match_status", "TUG_ID",
    "anpr_index", "anpr_time", "match_residual_s", "duty_class", "vehicle_category", "vehicle_subcategory",
    "unece_category", "fuel_group", "hybrid", "euro_stage", "euro_class", "euro_class_detail",
    "manufacturer", "manufacturer_region", "exhaust_side_assumed", "model", "gross_weight_kg",
    "weight_consistent", "dpf", "first_registration", "n_tech_records", "HEAT_ID",
]

def load_pass_tug_mapping() -> pd.DataFrame:
    """Output of pass_matching.py (run that first)."""
    return pd.read_csv(PASS_TUG_MAPPING_CSV, parse_dates=["lb_time", "anpr_time"])

def build_vehicle_mapping(passes: pd.DataFrame | None = None) -> pd.DataFrame:
    passes = load_pass_tug_mapping() if passes is None else passes
    classes = classify(load_tech_data())
    out = passes.merge(classes, on="TUG_ID", how="left", validate="m:1")
    no_tech = (out["match_status"] == "matched") & ~out["has_tech_data"].fillna(False).astype(bool)
    out.loc[no_tech, "match_status"] = "matched, no technical data"
    for c in ["vehicle_category", "vehicle_subcategory", "fuel_group", "exhaust_side_assumed"]:
        out[c] = out[c].where(out["match_status"] == "matched")
    return out.sort_values("pass_id")[OUTPUT_COLUMNS].reset_index(drop=True)

if __name__ == "__main__":
    mapping = build_vehicle_mapping()
    mapping.to_csv(VEHICLE_MAPPING_CSV, index=False)