# Input data of the extraction pipeline

Everything `extraction_template.py` and `grouping_template.py` read. The CSV files in this folder are **synthetic
example data**: they only show the expected format and let the templates run. They do not validate the algorithm.

```
data_needed_for_extraction/
├── merged_data/                  instrument data, one CSV per measurement day
│   └── merged_<YYYY-MM-DD>.csv
└── metadata/
    ├── light_barrier_triggers.csv    one row per light barrier pass (all days)
    └── vehicle_mapping.csv           vehicle data per pass (only for grouping_template.py)
```

File name pattern and paths: `extraction_template.py`, section `paths` (`MERGED_FILE`, `PASS_TIMES_CSV`,
`VEHICLE_MAPPING_CSV`).

## merged_data/merged_\<day\>.csv

| Column | Required | Content |
|---|---|---|
| column 0 (any name, e.g. `timestamp`) | yes | sample time, parseable by `pandas.to_datetime`, ascending, (nearly) constant interval |
| one column per channel | yes for every channel in `PAIRS` | measured value, numeric (empty = NaN); the column name is the channel name used in `PAIRS` |

All columns after column 0 are read as channels, so they must be numeric; channels not used in `PAIRS` are loaded but
ignored. The day in the file name must match the days of the light barrier triggers.

## metadata/light_barrier_triggers.csv

| Column | Required | Content |
|---|---|---|
| `Timestamp` | yes | trigger time of the light barrier, same clock as the merged data |
| any other | no | ignored by the extraction |

The days to load come from this file. `pass_index` = row number of a pass within its day in this file (0, 1, …),
so the rows of a day must be in time order and the file must not be filtered after the vehicle mapping was built:
`pass_index` is the key of every result (`source_day`, `pass_index`).
Two rows with the same `Timestamp` (two passes in the same second) are reported by `load_registers`: their template
columns are identical, the fit can only merge them into one group (one area for both) and the extraction never uses
them.

## metadata/vehicle_mapping.csv

Only `grouping_template.py` (vehicle groups) and `fitting_template.SPLIT_KEY` use it.

| Column | Required | Content |
|---|---|---|
| `source_day` | yes | day of the pass, `YYYY-MM-DD` |
| `pass_index` | yes | row of the pass within its day in `light_barrier_triggers.csv` |
| `match_status` | no | if the column exists, only `"matched"` passes are grouped (also for `All_included`) |
| filter columns | as needed | every column a filter in `grouping_template.GROUPS` or `fitting_template.SPLIT_KEY` uses |

The filter columns depend on the vehicle data available, a filter `{"column": value}` only needs that column to exist.
Filter syntax: scalar = equal, list = one of, tuple = range (min, max).
