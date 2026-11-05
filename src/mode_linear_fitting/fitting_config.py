"""Configuration of the Mode L fitting: everything needed to extend or run a fit.

Contents:
    ModeLConfig: Parameters of the day fit (windows, baseline rules, refinement, detection), shared by all channels.
    TemplateAssignment, Refinement: Signatures of the two per-channel extension points.
    GroupTemplates: Template assignment per pass, (source_day, pass_index) -> ShapeTemplate (vehicle groups).
    ChannelSetup: Everything channel specific: name, min_physical_value, peak window, templates, refinement.
    PairSetup: One instrument pair (CO₂ + optional pollutant channel) and the shared fit parameters, the only
        input of the fit (pair_fitting.fit_pair_day).
The inputs derived from the extraction (ShapeTemplate, PeakWindow) are in fitting_inputs, QA checks and
non-detect policies in fit_checks, refinement strategies in position_refinement.
implementations/impl_templates/fitting_pipeline/fitting_template.py sets all of it in one place.
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Assistance with Docstring writing / provenance parameter)
# AI-Assisted: <Opus 5.5> ; (Assistance with optimizing the code structure)


from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


from src.mode_linear_fitting.fit_checks import DEFAULT_QA_CHECKS, NonDetectPolicy, QACheck, keep_non_detects
from src.mode_linear_fitting.fitting_inputs import PeakWindow, ShapeTemplate, ShapeStatistic
from src.plume_template_extraction.extraction_result import CombinedResult
from src.plume_template_extraction.normalization import pooled_mean_shape

@dataclass(frozen=True)
class ModeLConfig:
    """Parameters of the Mode L day fit.

    Attributes:
        influence_before: Influence window before the expected peak position.
            Default 5 s, typical 3–10 s.
        influence_after: Influence window after the expected peak position.
            Default 15 s, typical 10–25 s.
        min_anchor_run: Minimum duration T_A of an anchor run. Default 5 s.
            Shorter gives more, smaller segments but less reliable anchors.
        t_min_linear: Minimum segment duration for a linear baseline. Default 60 s.
        min_anchor_time: Minimum total anchor duration N_anc,min · Δt of a segment for a fitted baseline, below
            the baseline is fixed (anchored_fixed); a linear baseline needs twice of it. Default 5 s
            (= 10 samples at Δt = 0.5 s, the former n_anchor_min). As a duration it does not depend on the
            sampling rate.
        merge_vif_max: VIF_max, template columns above are merged with a neighbor.
            Default 10, typical 5–10 (rule of thumb in Fahrmeir et al.).
        noise_band_sigma: Half width k₃ of the anchor noise band in σ. Default 3
        min_physical_run: Minimum duration below ChannelSetup.min_physical_value to count as faulty
            recording. Default 2 s.
        bg_percentile: Percentile q of the rolling background. Default 2.
        bg_rolling_window: Rolling background window T_RW. Default 100 s. Keep identical to the extraction.
        force_anchored_fixed: Use the fixed baseline in every segment. Default False.
        refine_sweeps: Maximum sweeps of the position refinement, stops early once a
            sweep moves nothing. Default 10, (needs to be high enough to give the methode the chance to converge)
        refine_min_dsse: κ, a shift is accepted only if the SSE drops by more than
            κ · σ̂². Default 8. Smaller lets templates move more easily.
        detect_sigma: k_det, a pass counts as detected if amplitude ≥ k_det · SE. Default 2.
    """

    influence_before: np.timedelta64 = np.timedelta64(5, "s")
    influence_after: np.timedelta64 = np.timedelta64(15, "s")
    min_anchor_run: np.timedelta64 = np.timedelta64(5, "s")

    t_min_linear: np.timedelta64 = np.timedelta64(60, "s")
    min_anchor_time: np.timedelta64 = np.timedelta64(5, "s")

    merge_vif_max: float = 10.0

    noise_band_sigma: float = 3.0

    min_physical_run: np.timedelta64 = np.timedelta64(2, "s")

    bg_percentile: float = 2.0
    bg_rolling_window: np.timedelta64 = np.timedelta64(100, "s")

    force_anchored_fixed: bool = False

    refine_sweeps: int = 10
    refine_min_dsse: float = 8.0
    detect_sigma: float = 2.0


# ----------------------------------------------------------------------------- channel and pair setup
# Pattern to assign a template to a pass its source day (string) and pass_index (int) -> ShapeTemplate
TemplateAssignment = Callable[[str, int], ShapeTemplate]

# The signature for position refinement strategies, take whatever as input and need to return a np.ndarray with the
# refined positions
Refinement = Callable[..., np.ndarray]


@dataclass(frozen=True)
class GroupTemplates:
    """ Template assignment by vehicle group: the template of the group of the pass, esle the fallback.

    Called as a function, (source_day, pass_index) -> ShapeTemplate. Any other callable with this signature can
    replace it in ChannelSetup.templates (e.g. a template chosen by vehicle speed).

    Attributes:
        fallback: Template of every pass wihout a group template (e.g. the All_included template)
        templates: Template per group value (e.g. {"left": template of LDV_Exhaust_left}).
        pass_values: Group value per pass, keyed by (source_day, pass_index); passes not listed, or with a value without
                     tempalte, get fallback.
    """

    fallback: ShapeTemplate
    templates: dict = field(default_factory=dict)
    pass_values: dict = field(default_factory=dict)

    def __call__(self, source_day: str, pass_index: int) -> ShapeTemplate:
        return self.templates.get(self.pass_values.get((source_day, pass_index)), self.fallback)

    @classmethod
    def from_directory(cls, template_dir: Path, channel: str, fallback_group: str, groups: dict | None = None,
                       pass_values: dict | None = None, min_n: int = 30,
                       shape: ShapeStatistic = pooled_mean_shape) -> GroupTemplates:
        """Build the assignment from grouped plume CSVs (template_dir/<group>/<channel>.csv)

        Args:
            template_dir: Folder with one subfolder per vehicle group (see extraction_pipeline/grouping_template.py)
            channel: Channel name (CSV file name).
            fallback_group: Group folder of the fallback template. (in template_dir the name of the folder (group))
            groups: Group folder per group value, e.g. {"left": "LDV_Exhaust_left"}. Defaults to None (no groups).
                    Means: The folder name LDV_Exhaust_left includes the grouped csvs of the template that should be
                           assigned to all vehicles where their pass_value (exhaust_side) equals left
            pass_values: Group value per (source_day, pass_index). Defaults to None. e.g. vehicle 100 of day 16.16.2016
                         has the value left, or right ( the keys in the groups dict)
            min_n: Minimum number of plumes of a group template:; smaller groups use the fallback. Default 30
            shape: Template statistic of every group and the fallback, e.g. median_shape (pointwise median, no area weighting).
                   Defaults to pooled_mean_shape (area-weighted mean).
        Returns:
            The assignment.
        Raises:
            ValueError: If the fallback CSV is missing or has fewer than min_n plumes.
        """

        path = Path(template_dir) / fallback_group / f"{channel}.csv"
        fallback = CombinedResult.from_csv(path) if path.exists() else None
        if fallback is None or fallback.n_valid < min_n:
            raise ValueError(f"{channel}: fallback {fallback_group} missing or below min_n = {min_n}")
        templates = {}
        for value, group in (groups or {}).items():
            path = Path(template_dir) / group / f"{channel}.csv"
            cr = CombinedResult.from_csv(path) if path.exists() else None
            if cr is not None and cr.n_valid >= min_n:
                templates[value] = ShapeTemplate.from_combined(cr, label=group, shape=shape)
        return cls(ShapeTemplate.from_combined(fallback, label=fallback_group, shape=shape), templates,
                   dict(pass_values or {}))



@dataclass(frozen=True)
class ChannelSetup:
    """Everything channel specific of ine fitted channel

    Attributes:
        channel: Channel name in the MeasurementRegister.
        min_physical_value: Values below it for at least ModeLConfig.min_physical_run count as faulty recording and
                            become NaN, in the channel unit.
        window: Peak window. CO2; relative to the light barrier trigger (PeakWindow.from_trigger_delays);
                             pollutant: relative to the CO2 peak of the same pass (PeakWindow.from_pollutant_offsets).
        templates: Template assignment (source_day, pass_idex) -> ShapeTemplate, e.g. GroupTemplates
        refinement: Position refinement strategy (see position_refinement), None = blind placement at tau 0
    """

    channel: str
    min_physical_value: float
    window: PeakWindow
    templates: TemplateAssignment
    refinement: Refinement | None = None


@dataclass(frozen=True)
class PairSetup:
    """One instrument pair: CO₂ reference channel, pollutant channel and the shared fit parameters.

    A CO2 ChannelSetup used by several pairs (e.g. one CO2 channel for BC and PN) should be the same object, then the pipeline
    fits it only once per day.

    Without pollutant only the CO2 channel is fitted.

    Attributes:
        name: Pair name, e.g. "NO2".
        co2: CO2 channel (positions relative to the light barrier trigger)
        pollutant: Pollutant channel (positions coupled to the CO₂ result of the same pass). Defaults to None -> CO2 only
        fit: Fit parameters shared by both channels. Defaults to ModeLConfig().
        qa_checks: QA check functions applied to every pass of both channels (extension point, see fit_checks).
                   Defaults to DEFAULT_QA_CHECKS (residual_above_noise, position_at_band_edge, co2_reference).
        non_detect: Non-detect policy applied to every pass of both channels after the checks (extension point, see fit_checks).
                    Defaults to keep_non_detects (fitted value stays).
    """

    name: str
    co2: ChannelSetup
    pollutant: ChannelSetup | None = None
    fit: ModeLConfig = ModeLConfig()
    qa_checks: tuple[QACheck, ...] = DEFAULT_QA_CHECKS
    non_detect: NonDetectPolicy = keep_non_detects