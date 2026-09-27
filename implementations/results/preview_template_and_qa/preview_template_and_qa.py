# This file contains the code that lead to the results in "Extrahierte Vorlagen und QA-Statistik"
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.shared_services.measurement_register import MeasurementRegister
from src.plume_template_extraction.extraction_config import ExtractionConfig, ChannelQAConfig
from src.plume_template_extraction.template_extraction import run_batch

# Data Loading
ROOT_DIR = Path(__file__).resolve().parents[3]
DATA = ROOT_DIR / "campaign_data" / "Merged_data"
PASS_TIMES = ROOT_DIR / "campaign_data" / "CARES_Milan_Madre_Cabrini_TUG_emission_ratios_allemissions_co2_4_80_ppm_3s.csv"

# Gets the LB-Triggers for the vehicles
pass_times = pd.read_csv(PASS_TIMES)
pass_times["Timestamp"] = pd.to_datetime(pass_times["Timestamp"])
available_days = pass_times["Timestamp"].dt.date.unique()
# Allows to exclude days that have a lot of faulty data for a channel
exclude_days = []

input_data = []
print(f"Days excluded from input data: {exclude_days}")
# Gathering all the input data
for day in available_days:
    day_data = pass_times[pass_times["Timestamp"].dt.date == day]
    data_path = DATA / f"CARES_Milan_MadreCabrini_instrument_data_time_aligned_{day}.csv"
    if not data_path.exists():
        print(f"Day {day} has no merged data")
        continue
    merged_data_of_day = pd.read_csv(data_path)
    input_data.append((day_data.reset_index(drop=True), merged_data_of_day))

# Building the MeasurementRegisters (skip_dt_validation=false would throw an error if the sampling times differ more than allowed)
registers = []
for day_lb_triggers, merged_df in input_data:
    lb_triggers = day_lb_triggers["Timestamp"].to_numpy(dtype="datetime64[ns]")
    register = MeasurementRegister.from_dataframe(merged_df, lb_triggers, skip_dt_validation=True)
    registers.append(register)

# Building Extraction Config (To show how the default values behave the config is not tuned)
# This config equals the default values, they are only assigned to show them, None -> Noise Based Estimation
preview_qa_config = ChannelQAConfig(
    band_before_co2_peak=np.timedelta64(2, "s"),
    band_after_co2_peak=np.timedelta64(4, "s"),
    smooth_window=np.timedelta64(2500, "ms"),
    smooth_polyorder=2,
    # These 2 values have to be adjusted for every channel, they define at what point a
    # signal is counted as faulty and should not be included in the extraction, they are
    # hand tuned by design
    min_physical_value=5.0,
    min_physical_run=np.timedelta64(2,"s"),

    min_peak_above_bg= None,
    peak_above_bg_sigma=3.0,
    min_prominence_ratio=0.3,
    min_prominence_floor=None,
    prominence_floor_sigma=3.0,
    tail_rise_ratio=None,
    tail_rise_abs=None,
    tail_percentile=95.0
)
# The Extraction Config should be kept the same for all channels to produce comparable results
preview_extraction_config = ExtractionConfig(
    co2_channel="Sample Co2 Channel",
    poll_channel="Sample Poll Channel",
    # For template extraction it makes sense to keep this in false, true drops co2 peaks with invalid pollutant peaks
    drop_co2_invalid_poll=False,
    min_gap=np.timedelta64(30, "s"),
    window_before=np.timedelta64(10, "s"),
    window_after=np.timedelta64(25, "s"),
    peak_search_after=np.timedelta64(15, "s"),
    window_before_peak=np.timedelta64(5, "s"),
    window_after_peak=np.timedelta64(20, "s"),
    bg_percentile=2.0,
    bg_rolling_window=np.timedelta64(100, "s"),
    co2_qa=ChannelQAConfig()
)