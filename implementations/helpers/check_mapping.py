from pathlib import Path

import pandas as pd

HELPERS_DIR = Path(__file__).resolve().parent
ROOT_DIR = HELPERS_DIR.parents[1]
MAPPING_CSV = HELPERS_DIR / "results" / "pass_tug_mapping.csv"
KNOLL_CSV = ROOT_DIR / "campaign_data" / "CARES_Milan_PS_TUG_EF_04112023.csv"
DEVIATIONS_CSV = HELPERS_DIR / "results"/ "debug" / "deviations.csv"


# Checks campaign_data/CARES_Milan_PS_TUG_EF_04112023.csv against results to validate the mapping
# Mapping Key is velocity and lb time -> minutes rounded
pf = pd.read_csv(MAPPING_CSV, parse_dates=["lb_time"])
pf["key_min"] = pf["lb_time"].dt.floor("min")
pf["key_v"] = pf["v"].round(6)

knoll = pd.read_csv(KNOLL_CSV, sep=";", decimal=",", encoding="latin-1", usecols=["TUG_ID", "Timestamp", "Velocity"])
knoll = knoll.rename(columns={"TUG_ID": "TUG_ID_knoll"})
knoll["key_min"] = pd.to_datetime(knoll["Timestamp"], format="%d.%m.%Y %H:%M")
knoll["key_v"] = knoll["Velocity"].round(6)

cmp = knoll.merge(pf[["pass_id", "lb_time", "key_min", "key_v", "TUG_ID", "match_status", "match_residual_s"]],
                  on=["key_min", "key_v"], how="inner")
cmp = cmp.drop_duplicates("pass_id", keep=False)

same = cmp["TUG_ID"] == cmp["TUG_ID_knoll"]
other = cmp["TUG_ID"].notna() & ~same
unmatched = cmp["TUG_ID"].isna()

print(f"Knoll rows: {len(knoll)}  paired with pf passes: {len(cmp)}")
print(f"  same TUG_ID      : {same.sum():>5}  ({same.mean():.2%})")
print(f"  div TUG_ID     : {other.sum():>5}")
print(f"  not matched by pf: {unmatched.sum():>5}")


# PRE MILAN_CONFIG = MatchingConfig(distance_lb_anpr_m=30.0, time_diff_lb_anpr_s=3.0) -> 98,08 Match Post -> 99,94 %