"""Track confirmation by M-of-N logic: what the tracker does for Pd and Pfa.

A single detection can be wrong; a sequence is much less likely to be. This lets the
per-look threshold be lowered. A lower threshold raises per-look Pd and false alarms,
and the M-of-N test then removes those false alarms. The gain is the difference between
the SNR needed for one-look detection and the SNR needed for confirmed tracks at the
same false-track rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom

from ..target.detection import Fluctuation, required_snr_db


def confirm_probability(p: float, m: int, n: int) -> float:
    """P(at least m detections in n looks)."""
    return float(binom.sf(m - 1, n, p))


def scans_to_confirm(p: float, m: int, n: int, horizon: int = 400) -> tuple[float, float]:
    """Sliding-window M-of-N as a Markov chain on the last n outcomes.

    Returns the mean scans to confirm (from no history) and the chance of confirming by
    scan n. Mean time counts all scans until the first time m hits lie in the last n.
    """
    states = 2**n
    popcount = np.array([bin(i).count("1") for i in range(states)])
    mask = states - 1
    state_prob = np.zeros(states)
    state_prob[0] = 1.0
    survival_sum = 0.0
    by_n = 0.0
    for scan in range(1, horizon + 1):
        next_prob = np.zeros(states)
        for state in np.nonzero(state_prob)[0]:
            weight = state_prob[state]
            miss = (state << 1) & mask
            hit = miss | 1
            next_prob[miss] += weight * (1.0 - p)
            next_prob[hit] += weight * p
        next_prob[popcount >= m] = 0.0  # absorbed: confirmed
        survival_sum += state_prob.sum()
        state_prob = next_prob
        if scan == n:
            by_n = 1.0 - state_prob.sum()
        if state_prob.sum() < 1e-10:
            break
    return float(survival_sum), float(by_n)


def false_track_rate_per_hour(pfa_cell: float, n_cells: float, gate_cells: float, m: int, n: int, scan_s: float) -> float:
    """False tracks per hour: starts from any false alarm, then m-1 of n-1 hits inside the gate."""
    starts_per_scan = pfa_cell * n_cells
    p_gate = -np.expm1(gate_cells * np.log1p(-min(pfa_cell, 0.5)))
    follow = float(binom.sf(m - 2, n - 1, p_gate)) if m >= 2 else 1.0
    return float(starts_per_scan * follow * 3600.0 / scan_s)


def required_pd_for_confirmation(target_probability: float, m: int, n: int) -> float:
    return float(brentq(lambda p: confirm_probability(p, m, n) - target_probability, 1e-6, 1.0 - 1e-9))


@dataclass(frozen=True)
class ThresholdTrade:
    pfa_grid: np.ndarray
    required_snr_db: np.ndarray
    false_tracks_per_hour: np.ndarray
    per_look_pd: float
    best_pfa: float
    best_snr_db: float
    best_false_tracks_per_hour: float
    single_look_snr_db: float
    gain_db: float


def threshold_trade(
    single_look_pfa: float, target_probability: float, m: int, n: int, looks: int, model: Fluctuation,
    n_cells: float, gate_cells: float, scan_s: float, allowed_false_tracks_per_hour: float,
) -> ThresholdTrade:
    """Find the lowest per-look SNR that still meets the false-track limit."""
    per_look_pd = required_pd_for_confirmation(target_probability, m, n)
    pfa_grid = np.logspace(-10, -1, 37)
    snr = np.array([required_snr_db(per_look_pd, p, looks, model) for p in pfa_grid])
    tracks = np.array([false_track_rate_per_hour(p, n_cells, gate_cells, m, n, scan_s) for p in pfa_grid])
    single = required_snr_db(target_probability, single_look_pfa, looks, model)

    feasible = tracks <= allowed_false_tracks_per_hour
    best = int(np.argmin(np.where(feasible, snr, np.inf))) if feasible.any() else 0
    return ThresholdTrade(
        pfa_grid=pfa_grid,
        required_snr_db=snr,
        false_tracks_per_hour=tracks,
        per_look_pd=per_look_pd,
        best_pfa=float(pfa_grid[best]),
        best_snr_db=float(snr[best]),
        best_false_tracks_per_hour=float(tracks[best]),
        single_look_snr_db=float(single),
        gain_db=float(single - snr[best]),
    )
