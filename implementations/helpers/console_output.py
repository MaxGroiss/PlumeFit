"""Formatted console output of the templates.

Not a script and nothing to adjust: used by extraction_template.py and fitting_template.py.

print_extraction_overview: the starting point for tuning the extraction to a new campaign, three tables, each with the
parameters it helps to set:
    QA shares                QA thresholds (ChannelQAConfig)
    Tail at the window end   window_after_peak, min_gap
    Peak positions           peak_search_after, band_before_co2_peak, band_after_co2_peak
print_peak_windows: the peak windows the fit uses.
print_fit_overview: what happened to the passes of a fitting run, one row per channel.
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5.5> ; (Console output of the templates collected here for the public release)

from __future__ import annotations

import numpy as np
import pandas as pd

from src.plume_template_extraction.plume_status import PlumeStatus
from src.plume_template_extraction.extraction_result import BatchResult, CombinedResult

# Length of the window end over which the tail level is averaged -> used to calculate the average level of the
# plume tail above background at the end of the extraction window -> can be used for window tuning
TAIL_END = np.timedelta64(1, "s")
# Time of the tail used for exponential interpolation to estimate how much area is lost due to a too narrow cutout
TAIL_FIT = np.timedelta64(10, "s")
# Reference label of the CO₂ peak positions (relative to the light barrier trigger)
LB_REFERENCE = "light barrier"


def print_extraction_overview(batch: BatchResult, combined: dict[str, CombinedResult], pairs: dict[str, dict],
                              area_weighting: bool = True) -> None:
    """Prints all three tables of one extraction run.

    Args:
        batch: Result of the batch extraction.
        combined: {channel: CombinedResult}, e.g. batch.combined_by_channel().
        pairs: Instrument pairs, see extraction_template.PAIRS.
        area_weighting: Area-weighted mean template (True) or median template (False), see template_shape.
    """
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", "{:.3g}".format):
        print("\nQA shares of the base in % -> QA thresholds")
        print(qa_shares(batch))
        print("\nTemplate tail at the window end -> window_after_peak / min_gap")
        print(template_tail_summary(combined, area_weighting=area_weighting))
        print("\nPeak positions in s (CO₂ vs. light barrier, pollutant vs. CO₂ peak) "
              "-> peak_search_after / band_before/after_co2_peak")
        print(peak_delay_summary(combined, pairs))


def print_peak_windows(windows: pd.DataFrame) -> None:
    """Prints the peak windows of a fitting run (run_tables.write_peak_windows)."""
    print(windows.to_string(float_format="%.2f"))


def print_fit_overview(passes: pd.DataFrame) -> None:
    """Prints the fit overview of a fitting run with the meaning of its columns.

    Args:
        passes: Result table of all days of the run (run_tables.build_passes).
    """
    days = sorted(passes["source_day"].unique())
    print(f"\nFit overview, {len(days)} day(s): {', '.join(days)}\n"
          "  (all values in % unless 'passes')\n"
          "  with CO2 result      pollutant only: passes whose CO2 reference has a fit result (coupled)\n"
          "  fitted               passes with status fitted (own or merged column)\n"
          "  merged of fitted     fitted passes that shared one column with neighbors (not separable in time)\n"
          "  detected of fitted   fitted passes with amplitude >= detect_sigma * SE\n"
          "  detected of single   detection rate of passes fitted with their own column\n"
          "  single of detected   detections that belong to exactly one pass (usable for emission ratios)\n"
          "  QA flagged of fitted fitted passes with at least one QA flag (see fit_checks)")
    print(fit_overview(passes).to_string(float_format="%.1f"))


# AI-Assisted: <Opus 5.5> ; (Clearer fit overview)
def fit_overview(passes: pd.DataFrame) -> pd.DataFrame:
    """Fit overview of all passes of the run, one row per channel (all values in % unless 'passes')."""
    rows = []
    for channel, t in passes.groupby("channel", sort=False):
        fitted = t["status"] == "fitted"
        single = fitted & (t["group_size"] == 1)
        detected = t["detected"].astype(bool) & fitted
        flagged = t["qa_flags"].fillna("").astype(str) != ""
        pollutant = t["role"].iloc[0] == "pollutant"
        rows.append({"channel": channel, "role": t["role"].iloc[0], "passes": len(t),
                     "with CO2 result %": 100 * t["coupled"].astype(bool).mean() if pollutant else np.nan,
                     "fitted %": 100 * fitted.mean(),
                     "merged % of fitted": 100 * (fitted & ~single).sum() / fitted.sum(),
                     "detected % of fitted": 100 * detected.sum() / fitted.sum(),
                     "detected % of single": 100 * (detected & single).sum() / single.sum(),
                     "single % of detected": 100 * (detected & single).sum() / detected.sum(),
                     "QA flagged % of fitted": 100 * (flagged & fitted).sum() / fitted.sum()})
    return pd.DataFrame(rows).set_index("channel")


def template_shape(cr: CombinedResult, area_weighting: bool = True) -> np.ndarray:
    """Template of a channel with unit-area scaling: area-weighted pooled mean or pointwise median."""
    return cr.mean_shape if area_weighting else cr.median_shape


# ----------------------------------------------------------------------------- QA thresholds
def qa_shares(batch: BatchResult) -> pd.DataFrame:
    """Share of every QA status in %, one row per channel.

    Reference ("base") of the shares:
        CO₂: isolated passes with a complete light barrier window. Isolated passes whose light
             barrier window exceeds the segment bounds get no status -> n_isolated - base.
        Pollutant: passes that passed the first CO₂ QA stage. The pollutant QA runs on exactly
             these passes, independent of the second CO₂ stage -> pollutant VALID can be larger
             than n_template (n_template also needs a VALID CO₂ plume).
    """
    # AI-Assisted: <Opus 5.5> ; (Shares and reference of the qa table)
    statuses = [s.value for s in PlumeStatus]
    rows = [{"channel": channel, "reference": r.config.co2_channel if r.config.poll_channel == channel else "",
             **{s.value: r.qa_counts[s] for s in PlumeStatus}, "n_template": r.n_valid}
            for channel, results in batch.grouped_by_channel().items() for r in results]
    table = pd.DataFrame(rows).groupby(["channel", "reference"], as_index=False, sort=False).sum()
    table["base"] = table[statuses].sum(axis=1)
    pct = table[statuses].div(table["base"], axis=0).mul(100).add_suffix("_pct")
    return pd.concat([table[["channel", "reference", "base"]], pct, table["n_template"]], axis=1).set_index("channel")


# ----------------------------------------------------------------------------- window_after_peak, min_gap
def template_tail_summary(combined: dict[str, CombinedResult], tail_end: np.timedelta64 = TAIL_END,
                          tail_fit: np.timedelta64 = TAIL_FIT, area_weighting: bool = True) -> pd.DataFrame:
    """Level of every template at the end of the cutout window and the area lost behind it.

    If the plumes have not decayed at the window end, the unit area of the template refers to a
    truncated plume -> window_after_peak too short. The lost area is estimated by continuing the
    tail exponentially behind the window: A_lost = y_end · τ (template area inside the window = 1).

    Args:
        combined: {channel: CombinedResult}, e.g. batch.combined_by_channel().
        tail_end: Duration at the window end over which the template level is averaged.
        tail_fit: Duration at the window end over which the exponential (log-linear) is fitted.
        area_weighting: Area-weighted mean template (True) or median template (False), see template_shape.

    Returns:
        One row per channel:
            n: plumes in the template
            window_after_s: cutout window after the peak in s
            tail_end_rel: template level at the window end / template peak
            tau_tail_s: time constant of the tail in s (NaN if the tail does not decay or is ≤ 0)
            area_lost_pct: share of the (extrapolated) total area lying behind the window in %
    """
    # AI-Assisted: <Opus 5.5> ; (Tail level and exponential estimate of the lost area)
    rows = []
    for ch, cr in combined.items():
        t, y = cr.time_axis, template_shape(cr, area_weighting)
        n_end = max(1, int(round(tail_end / np.timedelta64(1, "s") / cr.dt)))
        y_end = y[-n_end:].mean()
        # Log-linear fit over the end of the window, only possible where the template is positive
        fit = t >= t[-1] - tail_fit / np.timedelta64(1, "s")
        tau = np.nan
        if np.all(y[fit] > 0):
            slope = np.polyfit(t[fit], np.log(y[fit]), 1)[0]
            tau = -1 / slope if slope < 0 else np.nan
        area_lost = y_end * tau
        rows.append({"channel": ch,
                     "n": cr.n_valid,
                     "window_after_s": t[-1] + cr.dt,
                     "tail_end_rel": y_end / y[cr.peak_index],
                     "tau_tail_s": tau,
                     "area_lost_pct": 100 * area_lost / (1 + area_lost)})
    return pd.DataFrame(rows).set_index("channel")


# ----------------------------------------------------------------------------- peak_search_after, band_*_co2_peak
def _delay_row(channel: str, reference: str, delays_s: np.ndarray) -> dict:
    """Percentiles q5/q25/q50/q75/q95 of the peak delays of one channel in s."""
    q05, q25, q50, q75, q95 = np.percentile(delays_s, [5, 25, 50, 75, 95])
    return dict(channel=channel, reference=reference, n=len(delays_s),
                q05=q05, q25=q25, q50=q50, q75=q75, q95=q95)


def peak_delay_summary(combined: dict[str, CombinedResult], pairs: dict[str, dict]) -> pd.DataFrame:
    """Distribution of the peak positions of all template plumes relative to their reference point.

    CO₂: peak relative to the light barrier trigger (trigger_delays).
    Pollutant: peak relative to the peak of its CO₂ reference (pollutant_offsets). Only plumes
    inside the accepted band are in the result, so this distribution is truncated at the band.

    Args:
        combined: {channel: CombinedResult}, e.g. batch.combined_by_channel().
        pairs: Instrument pairs, see extraction_template.PAIRS.

    Returns:
        One row per channel: reference, n, q05, q25, q50, q75, q95 (in s).
    """
    # AI-Assisted: <Opus 5.5> ; (Peak delay percentiles per channel)
    rows = []
    for ch in dict.fromkeys(p["co2"] for p in pairs.values()):
        cr = combined.get(ch)
        if cr is not None and cr.trigger_delays is not None:
            rows.append(_delay_row(ch, LB_REFERENCE, cr.trigger_delays * cr.dt))
    for p in pairs.values():
        cr = combined.get(p["poll"])
        if cr is not None and cr.pollutant_offsets is not None:
            rows.append(_delay_row(p["poll"], p["co2"], cr.pollutant_offsets * cr.dt))
    return pd.DataFrame(rows).set_index("channel")
