"""This file is used to map the LB-Triggers of the CARES Milan to the actual vehicles

The Mapping is needed to create Templates for specific vehicle categories

@article{knoll2023pointsampling,
title={Large-scale automated emission measurement of individual vehicles with point sampling},
author={Markus Knoll and Martin Penz and Hannes Juchem and Christina Schmidt and Denis Pöhler and Alexander Bergmann},
journal={Atmospheric Measurement Techniques},
year={2023},
}


"""

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

LB_DIR = ADDITIONAL_DATA / "LB_data_Madre_Cabrini"
ANPR_CSV = ADDITIONAL_DATA / "CARES_Milan_TUG_vehicle_pass_times_cleaned.csv"
ANPR_BAZZONI_DIR = ADDITIONAL_DATA / "Bazzoni" / "ANPR"
TECH_DATA_CSV = ADDITIONAL_DATA / "Milan vehicles characteristics_february15.csv"

Output = Path(__file__).resolve().parent / "results"
PASS_TUG_MAPPING_CSV = Output / "pass_tug_mapping.csv"
# Collection of constant needed for matching the LB-Triggers
@dataclass
class MatchingConfig:
    """Constant are derived from Knoll RESCampaignProcessing / RESCampaignExample
    """                                       # Knoll variable names
                                              # Parameters from RESCampaignExample
    distance_lb_anpr_m: float = 29.0          # default_distance_lb_anpr_m
    time_diff_lb_anpr_s: float = 3.2          # default_time_diff_LB_ANPR_s
    acceleration_divider: float = 3.0         # acceleration_divider
                                              # Parameters from RESCampaignProcessing
    max_timediff_lb_anpr_s: float = 3.5       # max_timediff_lb_anpr_ns
    acc_comp_factor1: float = 1.1             # lb_acc_comp_factor1
    acc_comp_factor2: float = 1.6             # lb_acc_comp_factor2
    acc_comp_factor3: float = 2.3             # lb_acc_comp_factor3
    acc_corr_factor1: float = 0.75            # lb_acc_corr_factor1
    acc_corr_factor2: float = 0.5             # lb_acc_corr_factor2
    max_acc_anpr_comp: float = 3.0            # lb_max_acc_anpr_comp

# Comparing the Results of the matching with CARES_Milan_PS_TUG_EF_04112023.csv
# yielded the closest match with these two parameters changed from the default
MILAN_CONFIG = MatchingConfig(distance_lb_anpr_m=30.0, time_diff_lb_anpr_s=3.0)
#MILAN_CONFIG = MatchingConfig()
def load_lb_passes ()-> pd.DataFrame:
    # Adds the raw LB acceleration error to the pass list (needed for vehicle to trigger matching) and gives
    # all passes in the campaign a unique identifier.

    # Unnamed 0 is the Pass ID Column, it rises in a day and falls back to 0 on the next day
    passes = pd.read_csv(PASS_LIST_CSV, encoding="latin-1", usecols=["Unnamed: 0", "Timestamp", "Acceleration",
                                                                     "Velocity", "Vehicle length"])
    passes = passes.rename(columns={"Unnamed: 0": "er_day_index", "Timestamp": "lb_time", "Acceleration": "a",
                                    "Velocity": "v", "Vehicle length": "lb_length_m"})
    # New ID that is unique over all days
    passes.insert(0, "pass_id", np.arange(len(passes)))
    passes["lb_time"] = pd.to_datetime(passes["lb_time"])

    # Gathers all the LB Passes in one DataFrame with a new unique index
    lb_raw = pd.concat([pd.read_csv(f) for f in sorted(LB_DIR.glob("veh_*.csv"))], ignore_index=True)
    # Lookup dataframe
    lb_raw = pd.DataFrame({"lb_time": pd.to_datetime(lb_raw["Date_in"]).dt.floor("s"),
                           "v_key": lb_raw["v_mean (m/s)"].round(6),
                           "a_error": lb_raw["a_error (m/s^2)"]}).drop_duplicates(["lb_time", "v_key"])
    passes["v_key"] = passes["v"].round(6)
    # Merges the needed data (a_error) to the LB Frame using the pass time and rounded velocity as pairing keys
    passes = passes.merge(lb_raw, on=["lb_time", "v_key"], how="left").drop(columns="v_key")
    # Needed for corrected_acceleration
    passes["a_error_missing"] = passes["a_error"].isna()
    passes["a_error"] = passes["a_error"].fillna(0)
    # Results in
    """
        pass_id	er_day_index	lb_time	a	v	lb_length_m	a_error
    0	0	2021-09-26 00:03:43.000000	-0,785838769	5,622365976	2,068393622	2,821324036
    1	1	2021-09-26 00:06:44.000000	-3,772971822	9,8381336	6,157200916	2,614065081
    2	2	2021-09-26 00:06:48.000000	1,062799464	5,416860951	3,87953287	0,92035736
    ...
    """
    return passes.sort_values(["lb_time", "pass_id"]).reset_index(drop=True)

def filer_for_bazzoni_anpr_passes() -> pd.DataFrame:
    # Filters out the bazzoni passes, only madre cabrini merged data available
    names = ["anpr_index","TUG_ID", "anpr_time"]
    # The File doesnt have a header row -> Knoll data_column_names is used here
    anpr = pd.read_csv(ANPR_CSV, header=None, names=names)
    # The bazzoni files also have the anpr_index, they are gathered to find the bazzoni entries
    bazzoni = pd.concat([pd.read_csv(f, header=None, names=names) for f in ANPR_BAZZONI_DIR.glob("*.csv")])
    # gets only the bazzoni entries of ..._TUG_vehicle_pass_times_cleaned.csv
    anpr = anpr[~anpr["anpr_index"].isin(bazzoni["anpr_index"])].copy()
    anpr["anpr_time"] = pd.to_datetime(anpr["anpr_time"], format="%d.%m.%Y %H:%M:%S:%f")
    return anpr.sort_values("anpr_time").reset_index(drop=True)


def corrected_acceleration(acceleration: np.ndarray, a_error: np.ndarray, cfg: MatchingConfig) -> np.ndarray:
    # This function implements the acceleration correcting mechanism out of Knoll ->
    # relate_vehicle_passes_and_anpr_camera_data_and_calculate_emission_factors RESCampaignProcessing
    temp_lb_acc = acceleration.copy()
    # Consider acceleration error from LB measurement. If error is higher than acceleration value minimize impact
    # of the acceleration by reducing it
    for pos in range(len(temp_lb_acc)):
        if abs(a_error[pos]) > abs(acceleration[pos] * cfg.acc_comp_factor1) and \
                abs(a_error[pos]) < abs(acceleration[pos] * cfg.acc_comp_factor2):
            temp_lb_acc[pos] = temp_lb_acc[pos] * cfg.acc_corr_factor1
        elif abs(a_error[pos]) > abs(acceleration[pos] * cfg.acc_comp_factor2) and \
                abs(a_error[pos]) < abs(acceleration[pos] * cfg.acc_comp_factor3):
            temp_lb_acc[pos] = temp_lb_acc[pos] * cfg.acc_corr_factor2
    temp_lb_acc[abs(a_error) >= abs(acceleration * cfg.acc_comp_factor3)] = 0
    # Set max acceleration for further processing
    temp_lb_acc[temp_lb_acc > cfg.max_acc_anpr_comp] = cfg.max_acc_anpr_comp
    temp_lb_acc[temp_lb_acc < -cfg.max_acc_anpr_comp] = -cfg.max_acc_anpr_comp
    return temp_lb_acc


def estimated_travel_time_s(lb: pd.DataFrame, cfg: MatchingConfig) -> np.ndarray:
    # Knoll: est_time_diff_lb_to_anpr_s ->  t_LB->ANPR = d / (v + a_corr * T / divider)
    a_c = corrected_acceleration(lb["a"].to_numpy(), lb["a_error"].to_numpy(), cfg)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = cfg.distance_lb_anpr_m / (lb["v"].to_numpy() + a_c * cfg.time_diff_lb_anpr_s / cfg.acceleration_divider)
    t[~np.isfinite(t)] = 0.0
    return t


def match_lb_anpr(lb: pd.DataFrame, anpr: pd.DataFrame, cfg: MatchingConfig) -> pd.DataFrame:
    # Relates the LB-Triggers to the anpr data, based on relate_vehicle_passes_and_anpr_camera_data_and_calculate_emission_factors
    # by Knoll in RESCampaignProcessing
    ns = 1e9
    # Equals the expected camera pass time based on the lb trigger and the travel time estimation in ns
    adapted = lb["lb_time"].to_numpy("datetime64[ns]").astype(np.int64) + (estimated_travel_time_s(lb, cfg) * ns).astype(np.int64)
    t_anpr = anpr["anpr_time"].to_numpy("datetime64[ns]").astype(np.int64)
    # search frame for adapt to match an entry in t_anpr
    max_d = cfg.max_timediff_lb_anpr_s * ns
    order = np.argsort(adapted, kind="stable")
    # Order by time estimated time cought on camera: Problem: Slow Vehicle A Triggers LB first, Fast vehicle B triggers
    # afterward -> by adding the estimated travel time Vehicle B can overtake vehicle A in the estimation,
    # search sorted needs a sorted list (by timestamp) the order is preserved by order
    adapted_sorted = adapted[order]
    used = np.zeros(len(lb), dtype=bool)
    rows = []
    for j in range(len(anpr) - 1):
        # Differs from Knoll implementation, to reduce runtime
        # lo -> index where t_anpr[j] - max_d would fit into adapt_sorted to stay sorted
        # hi -> index where t_anpr[j] + max_d would fit into adapt_sorted to stay sorted
        # The loop then only has to check the passes that fit into this window
        lo = np.searchsorted(adapted_sorted, t_anpr[j] - max_d, side="left")
        hi = np.searchsorted(adapted_sorted, t_anpr[j] + max_d, side="right")
        # The rules that couple a lb trigger to an anpr pass are the same as in the reference implementation
        # abs of time diff between adapted and real anpr pass fits into the allowed frame -> guaranteed by the lo/hi frame
        # the next lb pass would not fit better
        # the next camera pass would not fit better
        # hits are marked trough used so anpr/lb combos are unique
        for i in np.sort(order[lo:hi]):
            if i >= len(lb) - 1:
                continue
            d = t_anpr[j] - adapted[i]
            if abs(d) < max_d and abs(d) < abs(t_anpr[j] - adapted[i + 1]) and abs(d) < abs(t_anpr[j + 1] - adapted[i]):
                if not used[i]:
                    used[i] = True
                    rows.append((j, i, d / ns))
                    break
    m = pd.DataFrame(rows, columns=["anpr_row", "lb_row", "residual_s"])
    out = pd.DataFrame({
        "pass_id": lb["pass_id"].to_numpy()[m["lb_row"]],
        "TUG_ID": anpr["TUG_ID"].to_numpy()[m["anpr_row"]],
        "anpr_index": anpr["anpr_index"].to_numpy()[m["anpr_row"]],
        "anpr_time": anpr["anpr_time"].to_numpy()[m["anpr_row"]],
        "match_residual_s": m["residual_s"].to_numpy(),
    })
    return out

OUTPUT_COLUMNS = ["pass_id", "er_day_index", "lb_time", "v", "a", "a_error_missing", "lb_length_m", "match_status",
                  "TUG_ID", "anpr_index", "anpr_time", "match_residual_s"]
def build_pass_matching(cfg: MatchingConfig = MILAN_CONFIG) -> pd.DataFrame:
    """One row per LB pass with the matched TUG_ID (match_status 'matched' / 'no ANPR match')."""
    lb, anpr = load_lb_passes(), filer_for_bazzoni_anpr_passes()
    matches = match_lb_anpr(lb, anpr, cfg)
    out = lb.merge(matches, on="pass_id", how="left", validate="1:1")
    out["match_status"] = np.where(out["TUG_ID"].isna(), "no ANPR match", "matched")
    return out.sort_values("pass_id")[OUTPUT_COLUMNS].reset_index(drop=True)

def dump(df, name: str, excel_de: bool = True) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    path = DEBUG_DIR / f"{name}.csv"
    df.to_csv(path, index=False,
              sep=";" if excel_de else ",",
              decimal="," if excel_de else ".",
              date_format="%Y-%m-%d %H:%M:%S.%f")

if __name__ == "__main__":
    #cfg = MatchingConfig()
    #lb = load_lb_passes()
    #dump(lb, "01_lb_passes")
    #anpr = filer_for_bazzoni_anpr_passes()
    #dump(anpr, "02_anpr_passes")
    #
    #matches = match_lb_anpr(lb, anpr, cfg)
    #dump(matches, "03_matches")
    Output.mkdir(exist_ok=True)
    mapping = build_pass_matching()
    mapping.to_csv(PASS_TUG_MAPPING_CSV, index=False)
    print("Done")