# PlumeFit

PlumeFit implements a template-based-fitting algorithm which functions as a post-processing method for vehicle emission data analysis. The algorithm was developed specifically for evaluating data gained trough roadside measurements using point sampling.  In point sampling, a small part of the diluted exhaust is drawn through an inlet next to the road and analyzed using various instruments, depending on the pollutants of interest. The measurement itself yields concentration time series that need to be evaluated via post-processing procedures to attribute the measured emissions to the relevant vehicle or vehicle group.

## Overview

 PlumeFit builds upon a two-step approach to evaluate the measurement data:

1. **Template extraction**

   Template extraction focuses on gaining prior knowledge required for signal separation using templates of isolated exhaust plumes. A template refers to the area-normalized, average plume shape of a vehicle group within a fixed time window around the peak. A vehicle group is a set of passes that share one template. Groups are formed from vehicle attributes available for each pass, such as vehicle category, fuel type, or exhaust position.

2. **Fitting**

   The second step fits these templates to every pass in the measurement time series. Overlaps are resolved as much as possible, and emission analysis is provided on per vehicle level.

**Input:** time-aligned instrument data and light barrier triggers, plus a vehicle mapping for vehicle groups.

**Output:** one .csv per day and channel with the fitted plume area of every pass.

## Background

PlumeFit was developed as part of my bachelor's thesis at TU Graz (Graz University of Technology) in 2026.

## Quick Start

Tested with Python 3.13 and 3.14.

```shell
pip install -r requirements.txt

python -m implementations.impl_templates.extraction_pipeline.extraction_template
python -m implementations.impl_templates.extraction_pipeline.grouping_template
python -m implementations.impl_templates.extraction_pipeline.copy_needed_data_to_fitting
python -m implementations.impl_templates.fitting_pipeline.fitting_template

# optional: figure of one pass (day, pass_index)
python -m implementations.impl_templates.fitting_pipeline.plot_pass 2024-05-13 12
```

The included data is **synthetic**. It shows the format and lets the templates run.

## Using your own data

1. **Provide the inputs**, see `data_needed_for_extraction/README.md`.
2. **Adjust the `SETTINGS` block** at the top of each implementation template script.

| Script                   | Adjust                                                       |
| ------------------------ | ------------------------------------------------------------ |
| `extraction_template.py` | file names, channel pairs (`PAIRS`), QA values, windows      |
| `grouping_template.py`   | vehicle groups (`GROUPS`)                                    |
| `fitting_template.py`    | days, template assignment, fit parameters, QA checks, diagnostics |

3. **Tune the extraction** with the console overview it prints: QA shares, template tail at the window end, peak positions. Each table names the parameters it helps to set.

Requirements of the data:

- (nearly) constant sampling interval, the same clock for instruments and light barrier

- the light barrier triggers before the plume is measured (the algorithm only check for peaks after the LB-Trigger)

For the algorithm to work properly there should be enough (around 30) passes per vehicle category (and channel) in interest that are isolated from other vehicle passes. By default the algorithm counts a spacing of 30 seconds or more to be sufficient. This is needed for template creation. 

## Output

```markdown
results/<run>/
├── run_config.json              # all settings of the run
├── peak_windows.csv             # peak window per channel in samples and s
├── passes/<day>/<channel>.csv   # the result, see below
└── diagnostics/<day>.zip        # <channel>_segments.csv.gz, <channel>_curves.csv.gz
```

### The fit in short

This section provides some background knowledge regarding the fit.

Every channel of a day is prepared as

$$r(t) = x(t) - \hat{b}(t) - m$$

with the measured signal $`x(t)`$, the background $`\hat{b}(t)`$ (centered rolling percentile) and the median $`m`$ of $`r`$ on
vehicle-free samples. In each segment $`r`$ is then described as

$$r(t) = \sum_j a_j s_j(t - \hat{\tau}_j) + \mathrm{baseline}(t) + e(t), \qquad \Delta t \sum_i s_j(t_i) = 1, \qquad a_j \ge 0$$

| Symbol                         | Meaning                                                      |
| ------------------------------ | ------------------------------------------------------------ |
| $`j`$, $`i`$                   | index of a vehicle pass and of a sample. Merged passes share one column and one area, see "Merged passes" |
| $`s_j`$                        | template with **unit area**, one column of the design matrix |
| $`a_j`$                        | fitted amplitude. Because $`s_j`$ has unit area, $`a_j`$ **is the plume area** (channel unit · s) |
| $`\tau^0_j`$, $`\hat{\tau}_j`$ | initial and refined peak position of pass $`j`$ in samples   |
| $`\Delta t`$                   | sampling interval in s                                       |
| $`A`$                          | design matrix of the segment: template columns and baseline columns |
| $`n`$, $`p`$                   | number of valid samples and of columns of the segment        |
| $`\hat{\sigma}`$               | noise level of the channel on this day, column `sigma`       |

The coefficients are found by bounded least squares (BVLS), the baseline coefficients stay free:

$$\min_x \lVert r - A x \rVert^2 \quad \text{subject to} \quad a_j \ge 0$$

### Columns of `passes/<day>/<channel>.csv`

Every light barrier pass has exactly one row per channel. Empty cells mean "not applicable". Numbers are stored with 6
significant digits.

**Identification**

| Column              | Meaning                                                      |
| ------------------- | ------------------------------------------------------------ |
| `source_day`        | day of the pass, `YYYY-MM-DD`                                |
| `pass_index`        | row of the pass within its day in `light_barrier_triggers.csv`, starting at 0. With `source_day` the key of a pass and of the vehicle mapping |
| `channel`           | channel name, so that several files can simply be concatenated |
| `role`              | `co2` or `pollutant`                                         |
| `reference_channel` | CO₂ channel the pollutant was coupled to, empty for CO₂      |

**Fit status**

| Column           | Value              | Meaning                                                      |
| ---------------- | ------------------ | ------------------------------------------------------------ |
| `status`         | `fitted`           | the pass has a fitted column, its own or one shared with a merged group |
|                  | `not_identifiable` | at most 50 % of the template area at $`\tau^0`$ lies on valid samples, the pass is left out of the fit |
|                  | `segment_failed`   | the segment could not be fitted: a column has at most 50 % of its area in the segment, or there are fewer valid samples than columns |
|                  | `outside_signal`   | $`\tau^0`$ lies outside the data of the day, the pass has no segment |
| `segment_status` | `fitted_linear`    | baseline $`\beta_0 + \beta_1 u(t)`$ with a centered ramp $`u(t)`$: anchor runs on both sides, segment longer than `t_min_linear` (60 s), at least 2 · `min_anchor_time` (10 s) of anchors |
|                  | `fitted_const`     | baseline $`\beta_0`$: otherwise, with at least `min_anchor_time` (5 s) of anchors |
|                  | `anchored_fixed`   | no free baseline: below `min_anchor_time`, or `force_anchored_fixed`. A polyline through the anchor levels is subtracted before the fit |
|                  | `failed`           | no fit in this segment                                       |
|                  | empty              | `outside_signal`, no segment                                 |

An *anchor run* is a stretch of at least `min_anchor_run` (5 s) of vehicle-free samples with $`\lvert r \rvert < k_3 \hat{\sigma}`$.
Anchor runs split a day into segments.

**Plume area and uncertainty**

| Column        | Meaning                                                      | Calculation                                                  |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `amplitude`   | plume area of the pass in channel unit · s. Empty if merged or not fitted | $`a_j`$ of its own column (BVLS)                             |
| `se`          | standard error of the column amplitude in channel unit · s, shared within a merged group. `inf` if undefined | $`\mathrm{se}_j = \sqrt{\hat{\sigma}_\mathrm{fit}^2 \left[(A^\top A)^{-1}\right]_{jj}}`$ with $`\hat{\sigma}_\mathrm{fit}^2 = \mathrm{SSE} / (n - p)`$ |
| `detected`    | the column amplitude is significant, shared within a merged group. `False` if not fitted | $`a_j \ge k_\mathrm{det} \, \mathrm{se}_j`$ with $`k_\mathrm{det}`$ = `detect_sigma` (default 2) |
| `replaced_by` | empty: `amplitude` is the fitted value. Otherwise the name of the non-detect policy that replaced it | default `keep_non_detects` never replaces. `non_detects_to_noise_level` sets $`a = k_\mathrm{det} \, \mathrm{se}`$ for not detected single passes and writes `noise_level` |

- **`se` is not `sigma`.** `se` comes from the fit residual of the segment ($`\hat{\sigma}_\mathrm{fit}`$), so a poor fit
  increases it. `sigma` is the noise on vehicle-free samples of the whole day.

- `se` follows the ordinary least squares formula: positions and templates count as known, and the bound $`a \ge 0`$ is not taken into account.

- The bound means no negative areas exist. A pass with $`a = 0`$ is reported as fitted and not detected.

**Merged passes**

Columns whose variance inflation factor exceeds `merge_vif_max` (default 10) cannot be separated. They are merged with their most similar neighbor into one column, the mean of the member templates. The fit then yields one area for the whole group.

| Column            | Meaning                                                      |
| ----------------- | ------------------------------------------------------------ |
| `group_size`      | number of passes sharing the column: 1 = separable, 0 = not fitted |
| `group_id`        | `pass_index` of the first member, `-1` if not fitted         |
| `group_amplitude` | total area of the group, only in the row of the first member. For `group_size = 1` equal to `amplitude` |

**Position**

| Column     | Meaning                                                      | Calculation                                                  |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `position` | peak position in samples of the day: $`\hat{\tau}`$ if fitted, $`\tau^0`$ otherwise | $`t \approx t_0 + \mathrm{position} \cdot \Delta t`$ with the first timestamp $`t_0`$ of the day |
| `shift_s`  | how far the refinement moved the template, in s              | $`(\hat{\tau} - \tau^0) \, \Delta t`$, 0 if not fitted       |

- **CO₂:** $`\tau^0 = \mathrm{trigger} + \mathrm{median\ delay}`$, search band $`\mathrm{trigger} + [q_5, q_{95}]`$ of the delays
  seen in the extraction.
- **Pollutant:** $`\tau^0 = \hat{\tau}_{\mathrm{CO}_2} + \mathrm{median\ offset}`$, search band
  $`\hat{\tau}_{\mathrm{CO}_2} + [q_5, q_{95}]`$ of the offsets. Without a CO₂ result the nominal CO₂ position is used.
- A move is only accepted if $`\Delta\mathrm{SSE} > \kappa \hat{\sigma}^2`$ with $`\kappa`$ = `refine_min_dsse` (default 8). A
  pollutant pass may only move if it is significant at its position ($`a \ge k_\mathrm{det} \, \mathrm{se}`$), which is
  tested again once after the first refinement.

**Quality**

| Column              | Meaning                                                      |
| ------------------- | ------------------------------------------------------------ |
| `qa_flags`          | flags of the QA checks, `;`-separated. They only inform, **no pass is removed** |
| `edge_extrapolated` | first column of a segment without a left anchor run, or last one without a right anchor run. A template cut at the signal border is rescaled to unit area, so part of the area is extrapolated |
| `sigma`             | noise level of the channel on this day, one value per file: $`\hat{\sigma} = \max\left(1.4826 \cdot \mathrm{MAD}(r_F), \; q / \sqrt{12}\right)`$, with $`r_F`$ = $`r`$ on vehicle-free samples (outside $`[\tau^0 - 5\,\mathrm{s}, \; \tau^0 + 15\,\mathrm{s}]`$ of every pass) and $`q`$ = smallest step of the raw signal |
| `segment_id`        | start sample of the segment, the key into the diagnostics. Empty without `SAVE_DIAGNOSTICS` |
| `template`          | origin of the template: `empirical:<channel>:<group>:<statistic>:n=<plumes>` (the statistic is omitted for the area-weighted mean) |

| Flag               | Set when                                                     | Default threshold                |
| ------------------ | ------------------------------------------------------------ | -------------------------------- |
| `residual_high`    | $`\sqrt{\mathrm{mean}\left[(r - \mathrm{model})^2\right]} > k_3 \hat{\sigma}`$ in the template window of the pass | $`k_3`$ = `noise_band_sigma` = 3 |
| `band_edge`        | $`\hat{\tau}`$ lies on the lower or upper end of the search band, a wider band might fit better | `DELAY_QUANTILES` = (5, 95)      |
| `co2_missing`      | pollutant only: the CO₂ pass is not fitted                   |                                  |
| `co2_not_detected` | pollutant only: the CO₂ pass is fitted but not detected      | `detect_sigma` = 2               |

**Pollutant / CO₂ coupling** (empty for CO₂ rows)

| Column              | Meaning                                                      | Calculation                                                  |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `coupled`           | the CO₂ pass of the reference channel is fitted              | `status` of the CO₂ row is `fitted`                          |
| `area_ratio`        | pollutant area / CO₂ area, **without unit conversion**. Empty for merged or not fitted passes, `inf` if $`a_{\mathrm{CO}_2} = 0`$ | $`a_\mathrm{poll} / a_{\mathrm{CO}_2}`$                      |
| `area_ratio_rel_se` | relative standard error of the ratio, assuming independent instruments (delta method). `inf` if one of the areas is 0 | $`\sqrt{(\mathrm{se}_\mathrm{poll} / a_\mathrm{poll})^2 + (\mathrm{se}_{\mathrm{CO}_2} / a_{\mathrm{CO}_2})^2}`$ |



### All thresholds

All in `fitting_template.py`, section `SETTINGS`, except where noted.

| Setting                                                      | Symbol                        | Default          | Used for                                                     |
| ------------------------------------------------------------ | ----------------------------- | ---------------- | ------------------------------------------------------------ |
| `detect_sigma`                                               | $`k_\mathrm{det}`$            | 2                | `detected`, significance test of the pollutant refinement    |
| `noise_band_sigma`                                           | $`k_3`$                       | 3                | anchor runs ($`\lvert r \rvert < k_3 \hat{\sigma}`$), flag `residual_high` |
| `refine_min_dsse`                                            | $`\kappa`$                    | 8                | a move needs $`\Delta\mathrm{SSE} > \kappa \hat{\sigma}^2`$  |
| `merge_vif_max`                                              | $`\mathrm{VIF}_\mathrm{max}`$ | 10               | columns above are merged                                     |
| `DELAY_QUANTILES`                                            | $`q_5, q_{95}`$               | (5, 95)          | search band of the refinement                                |
| `influence_before`, `influence_after`                        |                               | 5 s, 15 s        | what counts as vehicle-free for $`\hat{\sigma}`$ and anchors |
| `min_anchor_run`, `min_anchor_time`, `t_min_linear`          |                               | 5 s, 5 s, 60 s   | anchor runs and baseline mode                                |
| `bg_percentile`, `bg_rolling_window`                         |                               | 2, 100 s         | background $`\hat{b}(t)`$                                    |
| `min_physical_value` (extraction template), `min_physical_run` |                               | per channel, 2 s | faulty recordings become NaN                                 |
| `MIN_AREA_FRACTION` (constant in `mode_l_fitting.py`)        |                               | 0.5              | `not_identifiable`, `segment_failed`                         |

## Code structure

```
src/
├── shared_services/            # input structure, background and noise estimation
├── plume_template_extraction/
└── mode_linear_fitting/
implementations/
├── impl_templates/             # the scripts you run and adjust
└── helpers/                    # run folders, result files, console output (nothing to adjust)
```

| Module                                                       | Purpose                                                  |
| ------------------------------------------------------------ | -------------------------------------------------------- |
| `shared_services/measurement_register.py`                    | input data of one day, independent of the file format    |
| `shared_services/noise_and_background.py`                    | rolling background, noise estimate, vehicle-free samples |
| `plume_template_extraction/template_extraction.py`           | extraction flow                                          |
| `plume_template_extraction/extraction_quality.py`, `plume_status.py` | QA checks and status of the plumes                       |
| `plume_template_extraction/normalization.py`                 | zeroing, area normalization, mean shape                  |
| `plume_template_extraction/extraction_config.py`, `extraction_result.py` | configuration and results                                |
| `mode_linear_fitting/pair_fitting.py`                        | public API: fit CO₂ and the coupled pollutant            |
| `mode_linear_fitting/mode_l_fitting.py`                      | fit of one channel and day                               |
| `mode_linear_fitting/position_refinement.py`                 | where a template sits in its segment                     |
| `mode_linear_fitting/fitting_inputs.py`                      | shape templates and peak windows                         |
| `mode_linear_fitting/fit_checks.py`                          | QA checks and non-detect policies                        |
| `mode_linear_fitting/fitting_config.py`, `fitting_result.py` | configuration and results                                |

**Extending PlumeFit:**

- refinement strategy: `ChannelSetup.refinement`, see `position_refinement.py`
- template assignment: `ChannelSetup.templates`, see `GroupTemplates`
- QA checks and non-detect policy: `PairSetup`, see `fit_checks.py`

## AI assistance

Parts of the code and documentation were created with AI assistance. Individual uses are marked by `AI-Assisted` comments in the files. Example: # AI-Assisted: <Opus 5.5> ; (Extraction of relevant parameters and latex equations for README.md)

## License

MIT, see `LICENSE`.