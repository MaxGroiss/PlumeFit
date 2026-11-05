"""Public API of the Mode L fitting: one instrument pair (CO2 + pollutant) per day, coupled.

Flow of fit_par_day (one MeasuremntRegister, one PairSetup):
    1. CO₂: τ_j⁽⁰⁾ = trigger_j + q50 of the trigger delays, search band trigger_j + [q5, q95], fitted with the CO₂ refinement strategy (fit_channel_day).
    2. Pollutant: the template peak of every pass is placed at its refined CO₂ peak τ̂_CO₂,j plus q50 of the pollutant offsets,
                  search band τ̂_CO₂,j + [q5, q95] of the offsets. Passes without a CO₂ result (failed segment, not identifiable)
                  are placed at their nominal CO₂ position, so their plume is not taken up by neighbors, the baseline;
                  they are reported with coupled = False.
    3. One PairDayResult: one row per light barrier pass with both channels (to_dataframe).

A PairSetup without pollutant fits only step 1 (CO₂ only).
fit_pairs_day runs several pairs on one day and fits a CO₂ channel shared by several pairs only once.

Inputs from the extraction results:
peak_windows_from_extraction: PeakWindow per channel from the extraction CSVs.
GroupTemplates.from_directory (fitting_config): template assignment per pass from grouped plume CSVs.
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Assistance with Docstring writing, latex equation to Unicode symbol equations)
# AI-Assisted: <Opus 5.5> ; (Assistance with optimizing the code structure for ease of use)

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.mode_linear_fitting.fitting_config import ChannelSetup, PairSetup
from src.mode_linear_fitting.fitting_inputs import PeakWindow
from src.mode_linear_fitting.fitting_result import ChannelDayResult, PairDayResult
from src.mode_linear_fitting.mode_l_fitting import fit_channel_day
from src.plume_template_extraction.extraction_result import CombinedResult
from src.shared_services.measurement_register import MeasurementRegister


def peak_windows_from_extraction(extraction_dir: Path, quantiles: tuple[float, float] = (5.0, 95.0)) -> dict[str, PeakWindow]:
    """PeakWindow for every channel in the extraction CSVs (extraction_dir/<channel>.csv).

    CO2 channels (trigger_delay column): Window relative to the LB-Trigger;
    pollutant channels (pollutant_offset column): window relative to the Co2 peak of the same pass.

    Args:
    extraction_dir: Folder with the per-channel extraction CSVs (CombinedResult.to_csv).
    quantiles: Search band quantiles. Defaults to (5, 95).

    Returns:
        {channel: PeakWindow}
    """
    windows = {}
    for path in sorted(Path(extraction_dir).glob("*.csv")):
        cr = CombinedResult.from_csv(path)
        windows[cr.channel] = PeakWindow.from_extraction(cr, quantiles)
    return windows


def trigger_samples(register: MeasurementRegister) -> np.ndarray:
    """Sample index of every light barrier trigger."""
    return np.searchsorted(register.timestamps, register.vehicle_pass_times)


# AI-Assisted: <Opus 5.5> ; (Check if window parameters came from the same extraction where the fit will be performed on)
def _check_window(window: PeakWindow, reference: str, register: MeasurementRegister) -> None:
    if window.reference != reference:
        raise ValueError(f"peak window relative to {window.reference!r}, expected {reference!r}")
    if not np.isclose(window.dt, register.dt):
        raise ValueError(f"peak window dt {window.dt} s differs from the register dt {register.dt} s")

def _band(anchor: np.ndarray, window: PeakWindow, setup: ChannelSetup) -> tuple[np.ndarray, np.ndarray] | None:
    """Search band anchor + [lo, hi], None without refinement strategy."""
    return None if setup.refinement is None else (anchor + window.lo, anchor + window.hi)


def _fit_co2_day(register: MeasurementRegister, setup: PairSetup, keep_diagnostics: bool = False) -> ChannelDayResult:
    """Performs the fitt for the CO2 channel of the pair"""
    window = setup.co2.window
    _check_window(window, "trigger", register)
    trigger = trigger_samples(register).astype(float) # LB-Trigger Indexes
    return fit_channel_day(register, setup, setup.co2, trigger + window.center, _band(trigger, window, setup.co2),
                           keep_diagnostics=keep_diagnostics)

def _fit_pollutant_day(register: MeasurementRegister, setup: PairSetup, co2: ChannelDayResult,
                      keep_diagnostics: bool = False) -> ChannelDayResult:
    """Pollutant channel of a pair, positions coupled to the CO₂ result of the same pass

    The CO₂ amplitude of every pass is handed to the refinement strategy as priority (column amplitude, 0 for
    passes without CO₂ result).
    !If you want to change the priority when using a different refinement strategy, do it here!
    The current refine was tested with refining the pollutants ordered by the significance of their co2 peak
    the results did not lead to an improvement so it was scrapped but as an example where to pass priorities the
    fill was kept.
    """
    if setup.pollutant is None:
        raise ValueError(f"pair {setup.name} has no pollutant channel (CO₂ only)")
    window = setup.pollutant.window
    _check_window(window, "co2_peak", register)
    if co2.channel != setup.co2.channel or co2.source_day != register.source_day:
        raise ValueError(f"CO₂ result {co2.channel} {co2.source_day} does not belong to this pair and register")
    # float is used because int cant handle possible nan values
    anchor = np.array([p.position for p in co2.passes],dtype=float)
    # Non valid co2 passes have nan amplitudes -> np.nan_to_num -> 0
    priority = np.nan_to_num(co2.column_amplitudes())
    return fit_channel_day(register, setup, setup.pollutant, anchor + window.center, _band(anchor, window, setup.pollutant),
                           priority=priority, reference=co2, keep_diagnostics=keep_diagnostics)


def fit_pair_day(register: MeasurementRegister, setup: PairSetup, co2: ChannelDayResult | None = None,
                  keep_diagnostics: bool = False) -> PairDayResult:
    """Fit one instrument pair on one day: CO2 first, then the coupled pollutant (if the pair has one)

    Args:
        register: Measurement data of the day.
        setup: The setup of the pair
        co2: Already fitted CO2 result of setup.co2 on this register (shared co2 channel). Defaults to None (Co2 is fitted here)
        keep_diagnostics: Keep residual and segment diagnostics. Defaults to False.
    Returns:
        One PairDayResult, one row per light barrier pass in to_dataframe(); pollutant None for Co2 only setup.

    """
    if co2 is None:
        co2 = _fit_co2_day(register, setup, keep_diagnostics=keep_diagnostics)
    if setup.pollutant is None:
        return PairDayResult(setup.name, co2)
    return PairDayResult(setup.name, co2, _fit_pollutant_day(register, setup, co2, keep_diagnostics))



def fit_pairs_day(register: MeasurementRegister, setups: dict[str, PairSetup],
                  keep_diagnostics: bool = False) -> dict[str, PairDayResult]:
    """Fit several pairs on one day; every CO₂ channel is fitted once, even if it is the reference of several pairs.

    Args:
        register: Measurement data of the day.
        setups: {pair name: PairSetup}. Pairs sharing a CO₂ channel must use the same ChannelSetup object and
            the same fit parameters.
        keep_diagnostics: See fit_pair_day. Defaults to False.

    Returns:
        {pair name: PairDayResult}.

    Raises:
        ValueError: If two pairs use different setups for the same CO₂ channel.
    """
    co2_fits: dict[str, tuple[PairSetup, ChannelDayResult]] = {}
    results = {}
    for name, setup in setups.items():
        # AI-Assisted: <Opus 5.5> ; (Check if co2 was already fitted and can be reused)
        known = co2_fits.get(setup.co2.channel)
        if known is not None and (known[0].co2 is not setup.co2 or known[0].fit != setup.fit
                                  or known[0].qa_checks != setup.qa_checks
                                  or known[0].non_detect is not setup.non_detect):
            raise ValueError(f"pairs {known[0].name} and {name} use different setups for {setup.co2.channel}")
        if known is None:
            known = co2_fits[setup.co2.channel] = (setup, _fit_co2_day(register, setup, keep_diagnostics))
        results[name] = fit_pair_day(register, setup, known[1], keep_diagnostics)
    return results



