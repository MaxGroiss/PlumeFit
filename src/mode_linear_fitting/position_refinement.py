"""Position refinement strategies: where the template of every pass sits in its segment.

Build in a way so future implementations (strategies) can be adapted if they use the same signature
the refinement methode can be selected per channel in ChannelSetup.refinement.

strategy(y, templates, tau0, lo, hi, dt, baseline, noise_var, cfg, priority) -> tau_hat
    y          segment signal
    templates  template per pass
    tau0       initial positions τ⁽⁰⁾ relative to the segment start in samples
    lo, hi     search band C_j = [τ_j^min, τ_j^max] per pass relative to the segment start in samples
    dt         sampling interval in s
    baseline   baseline model of the segment (None, "const", "linear")
    noise_var  σ̂² of the vehicle-free residual
    cfg        ModeLConfig (refine_sweeps, refine_min_dsse = κ, …)
    priority   optional per-pass priority (pollutant fit: CO₂ amplitude of the same pass), For possible prior of fast moving vehicles for example
    Returns
    tau_hat    refined positions τ̂ in samples, same order as tau0

All strategies so they fit in the pipeline should minimize SSE(τ) inside the band and accept moves only if it lowers the
sse with positive amplitude and not noise driven. The strategy has to preserve the arrival order of passes.

Implemented Strategies:
refine_positions: cyclic coordinate descent with the OLS lower bound prefilter (see thesis appendix),
                  processing order by decreasing initial amplitude. Used for CO₂. (basically just calls coordinate_descent)
refine_pollutant_positions: the same descent for pollutants, but only passes with a significant amplitude at
	        their blind position (a_j ≥ k_det · se_j) may move; the others keep τ⁽⁰⁾ = τ̂_CO₂ + median offset, unless
	        they become significant after the first descent (one re-check).
coordinate_descent: the descent itself, with a free processing order and an optional mask of movable passes.

New strategies can be added by a new function with this signature, set in the ChannelSetup.
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5> ; (Assistance with Docstring writing, latex equation to Unicode symbol equations)
# AI-Assisted: <Opus 5.5> ; (Assistance with more modular strategy design)


from __future__ import annotations

import numpy as np

from src.mode_linear_fitting.fitting_config import ModeLConfig
from src.mode_linear_fitting.fitting_inputs import ShapeTemplate
from src.mode_linear_fitting.mode_l_fitting import (template_column, baseline_columns, design_matrix, bvls,
                                                    segment_sse, fit_mode_l, criteria_of_detection)

def neighbor_passes(tau0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Previous and next pass (by arrival) ov every pass, -1 if no pass (edges)

    The position refinement keeps the order of occurrence dictated from the Licht Barrier Passes intact.
    Vehicle A passes the LB before vehicle B after the refine_positions vehicle B cannot be in front of vehicle A.
    """

    rank = np.argsort(tau0, kind="stable") # so equal positions keep their input order
    prev_pass = np.full(tau0.size, -1)
    next_pass = np.full(tau0.size, -1)
    prev_pass[rank[1:]] = rank[:-1]
    next_pass[rank[:-1]] = rank[1:]
    # tau0 can be not in order of arrival so tau0 = [30, 12, 30, 5] says that the first pass is at position 30
    # when ordering by arrival -> [5,12,30,30] so prev_pass[0] references the indexes of tau0 and results 1 because
    # in the sorted array the tau0 with value 12 is at array positon 1 wich comes before the 0 index pass of tau0
    return prev_pass, next_pass

def search_band(j: int, tau: np.ndarray, lo: np.ndarray, hi: np.ndarray, n: int, prev_pass: np.ndarray,
                next_pass: np.ndarray) -> tuple[int, int]:
    """Search band C_j = [τ_min, τ_max] of pass j, clipped to the segment bounds and the current neighbor positions.

    The shift is guarded by lo and hi, bounds that prevent a template from traveling away from its plume. It also
    prevents that candidates sitting near the segment edges travel out of it.
    """
    p_lo = max(int(lo[j]), 0)
    p_hi = min(int(hi[j]), n - 1)
    if prev_pass[j] >= 0: # >= 0 because the prev/next is guarded by -1 if pass j is the first/last pass in the segment
        p_lo = max(p_lo, int(tau[prev_pass[j]])) #If a pass is not the first one in the segment it cannot shift past its left neighbor
    if next_pass[j] >= 0:
        p_hi = min(p_hi, int(tau[next_pass[j]])) #If a pass is not the last one in the segment it cannot shift past its right neighbor
    return p_lo, p_hi



def refine_positions(signal: np.ndarray, templates: list[ShapeTemplate], positions: np.ndarray,
                     lo: np.ndarray, hi: np.ndarray, dt: float, baseline: str | None, noise_var: float,
                     cfg: ModeLConfig = ModeLConfig(), priority: np.ndarray | None = None) -> np.ndarray:
    """Refine the template positions of one segment by minimizing the SSE (thesis appendix, used for CO₂).

    Coordinate descent (see coordinate_descent) in order of decreasing initial amplitude.

    Args:
        signal: Segment signal, as for fit_mode_l.
        templates: Template per pass.
        positions: Initial peak positions τ⁽⁰⁾ (trigger + median delay) relative to
            the segment start in samples.
        lo: Lowest allowed position per pass, relative to the segment start in samples.
        hi: Highest allowed position per pass, relative to the segment start in samples.
        dt: Sampling interval in s.
        baseline: Baseline model, see baseline_columns.
        noise_var: Noise variance σ̂² of the vehicle-free residual.
        cfg: refine_sweeps (maximum number of sweeps, default 10) and refine_min_dsse (κ of the
            acceptance threshold κ · σ̂², default 8). Defaults to ModeLConfig().
        priority: Not used (the order follows the initial amplitude), see the module docstring.

    Returns:
        Refined positions τ̂ [samples].
    """
    return coordinate_descent(signal, templates, positions, lo, hi, dt, baseline, noise_var, cfg)


def coordinate_descent(signal: np.ndarray, templates: list[ShapeTemplate], positions: np.ndarray,
	                       lo: np.ndarray, hi: np.ndarray, dt: float, baseline: str | None, noise_var: float,
	                       cfg: ModeLConfig = ModeLConfig(), order: np.ndarray | None = None,
	                       movable: np.ndarray | None = None) -> np.ndarray:
    """Cyclic coordinate descent on SSE(τ) inside the search bands

        One pass at a time, in the fixed processing order, is moved to the candidate position τ' with the lowest
	    BVLS SSE. Candidates are pre-sorted by a cheap lower bound of their SSE (one least-squares solve for all candidates)
	    BVLS only runs while the bond can still beat the best SSE. A move is accepted only if it lowers the SSE by more than
	    κ ⋅ noise_var. Sweeps repeat until nothing moves. The arrival order of the passes is preserved.

	Args:
	    signal, templates, positions, lo, hi, dt, baseline, noise_var, cfg: See refine_positions.
	    order: Processing order (indicies of the passes).
	    movable: Bool per pass, False keeps the pass at its initial position (its column stays in the fit).
	             None: every pass may be moved.

	Returns:
	    Refined positions τ̂ [samples].
	Raises:
	    SegmentNotIdentifiable: If the initial positions give a template column with too little area.

    """
    # Parameter preparation
    tau = np.array(positions, dtype=int)
    n, valid = signal.size, np.isfinite(signal)
    valid_signal = signal[valid]
    threshold = cfg.refine_min_dsse * noise_var

    # Initial fit: SSE reference and processing order (largest amplitude first, so weak
    # plumes cannot settle inside a strong plume before the strong one is placed)
    x0, curr_sse, _ = bvls(design_matrix(n, templates, tau, dt, baseline), signal, len(templates))
    # Orders by amplitude if no other ordering is given (priority ordering)
    amp_order = np.argsort(-x0[:len(templates)]) if order is None else np.asarray(order)
    if movable is not None:
        amp_order = [j for j in amp_order if movable[j]]
    # Design matrix at the current positions (valid rows), column j = pass j, baseline last.
    # A move only replaces the column of the moved pass.
    A = np.column_stack([template_column(t, p, n, dt) for t, p in zip(templates, tau)] + baseline_columns(n, baseline))[valid]
    prev_pass, next_pass = neighbor_passes(np.asarray(positions))
    # The refine positions loop
    for _ in range(cfg.refine_sweeps):
        moved = False
        # Highest Amplitudes get shifted first trying to prevent low emitters hiding in high emitter plumes instead of
        # their own
        for j in amp_order:
            # For the current j pass all possible positions are collected and the template centeres are shifted
            p_lo, p_hi = search_band(j, tau, lo, hi, n, prev_pass, next_pass)
            candidates, cols = [], []
            for p in range (p_lo, p_hi + 1):
                col = template_column(templates[j], p, n, dt) if p != tau[j] else None
                if col is not None:
                    # Gets all the columns with the template of pass j positioned at the available points in the band
                    candidates.append(p)
                    cols.append(col[valid])
            if not candidates:
                # No candidates available for this pass in this sweep
                continue
            # Lower bound of the SSE per candidate: project signal and candidate columns onto the
            # orthogonal complement of the other columns (one lstsq for all), then the best
            # non-negative amplitude of each candidate on the remaining residual
            other_passes = np.delete(A, j, axis=1)
            targets = np.column_stack([valid_signal] + cols)
            # np.linalg.lstsq(other_passes, targets, rcond=None)[0] = Residual Maker in thesis
            residual = targets - other_passes @ np.linalg.lstsq(other_passes, targets, rcond=None)[0]
            y_perp, s_perp = residual[:, 0], residual[:, 1:]
            sse_without = float(y_perp @ y_perp)
            num = s_perp.T @ y_perp
            denom = (s_perp ** 2).sum(axis=0)
            # A negative optimal amplitude is clipped to 0 by BVLS -> no gain
            gain = np.divide(num ** 2, denom, out=np.zeros_like(num), where= num>0)
            lower = sse_without - gain
            best_sse, best_position = curr_sse - threshold, tau[j]
            # Verify candidates by BVLS in order of their bound; stop once the bound
            # cannot beat the best SSE found so far
            for c in np.argsort(lower):
                if lower[c] >= best_sse:
                    # No need to validate a candidate that could not beat the current best sse even with its lower bound
                    break
                trial = tau.copy()
                trial[j] = candidates[c]
                sse = segment_sse(signal, templates, trial, dt, baseline)
                if sse < best_sse:
                    best_sse, best_position = sse, candidates[c]
            if best_position != tau[j]:
                # If the new best positon has changed moved is marked so the loop continues till everything is stationary
                tau[j], curr_sse, moved = best_position, best_sse, True
                A[:, j] = template_column(templates[j], best_position, n, dt)[valid]
        if not moved:
            break
    return tau



# AI-Assisted: <Opus 5.5> ; (Rework: pollutant refinement)
def refine_pollutant_positions(signal: np.ndarray, templates: list[ShapeTemplate], positions: np.ndarray,
                               lo: np.ndarray, hi: np.ndarray, dt: float, baseline: str | None, noise_var: float,
                               cfg: ModeLConfig = ModeLConfig(), priority: np.ndarray | None = None) -> np.ndarray:
    """Pollutant refinement: the coordinate descent of refine_positions, moving only significant passes.

    A pass whose amplitude at its blind position τ⁽⁰⁾ = τ̂_CO₂ + median offset is not significant
    (a_j < k_det · se_j) carries no information about its position. Moving it
    could only fit the noise, a noise bump inside the band then looks like a plume.
    Such a pass keeps τ⁽⁰⁾, its column stays in the fit. All other passes are refined exactly
    like refine_positions (band, κσ̂² acceptance, arrival order, order of decreasing initial amplitude).

    # AI-Assisted: <Opus 5.5> ; (Addition of the re check)
    Re-check: a real plume placed far off at τ⁽⁰⁾ can look insignificant only because its neighbors are still
    misplaced. After the first descent the significance is therefore tested once more, at the refined positions,
    and the passes that became significant are refined in a second descent (the others stay where they are).

    Args:
        signal, templates, positions, lo, hi, dt, baseline, noise_var: See refine_positions.
        cfg: refine_sweeps, refine_min_dsse (κ) and detect_sigma (k_det). Defaults to ModeLConfig().
        priority: Not used (CO₂ amplitude order brought no gain, see the report).

    Returns:
        Refined positions τ̂ [samples].
    """
    def significant(tau: np.ndarray) -> np.ndarray:
        "Performs the initial fit to decide what a significant pass is."
        fit = fit_mode_l(signal, templates, np.asarray(tau), dt, baseline)
        return criteria_of_detection(fit.amplitudes, fit.amplitudes_se, cfg)

    movable = significant(positions)
    tau = coordinate_descent(signal, templates, positions, lo, hi, dt, baseline, noise_var, cfg, movable=movable)
    # Is the peak now significant after the first move
    newly = significant(tau) & ~movable
    if newly.any():
        tau = coordinate_descent(signal, templates, tau, lo, hi, dt, baseline, noise_var, cfg, movable=newly)
    return tau









