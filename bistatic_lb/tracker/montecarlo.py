"""Stage 4, simulation: run the track filters on detections drawn from the link budget.

A target flies a straight line. At every scan the link budget gives the SINR, and so the
detection probability and measurement accuracy, at that point. Each run draws detections
and false alarms from those numbers, and passes them through the association and filters.
Results are compared with the single-look values and the PCRLB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..config import Scenario
from ..geometry import build_geometry
from ..target.estimation import compute_estimation
from ..target.link import TargetAnalyzer
from ..units import C_M_S
from .bounds import constant_velocity_model, information_reduction_factor, pcrlb_series
from .filters import GaussianTrack


@dataclass(frozen=True)
class TruthAtScan:
    position: np.ndarray
    range_m: float
    rate_m_s: float
    azimuth: float
    pd: float
    sigma_range_m: float
    sigma_rate_m_s: float
    sigma_azimuth: float
    single_look_position_m: float


@dataclass(frozen=True)
class TrackerMonteCarloResult:
    time_s: np.ndarray
    rmse_range_m: np.ndarray
    rmse_rate_m_s: np.ndarray
    bound_range_m: np.ndarray
    bound_rate_m_s: np.ndarray
    single_look_range_m: np.ndarray
    single_look_rate_m_s: np.ndarray
    rmse_position_m: np.ndarray
    single_look_position_m: np.ndarray
    per_scan_pd: np.ndarray
    continuity: float
    lost_runs: int
    runs: int
    seed: int
    pfa_per_cell: float


def build_truth(analyzer: TargetAnalyzer, scans: int, pfa: float) -> list[TruthAtScan]:
    s: Scenario = analyzer.s
    g0 = build_geometry(s)
    dt = s.tracker.update_interval_s
    n_cells = analyzer.number_of_search_cells()
    truth = []
    for k in range(scans):
        link = analyzer.link(g0.target + g0.velocity * k * dt)
        g = link.geometry
        est = compute_estimation(s, analyzer.array, analyzer.waveform, link, n_cells, None)
        truth.append(
            TruthAtScan(
                position=g.target,
                range_m=g.bistatic_range_m,
                rate_m_s=g.bistatic_range_rate_m_s,
                azimuth=g.azimuth_at_rx_rad,
                pd=analyzer.detection_at(link.sinr_db, pfa),
                sigma_range_m=est.sigma_bistatic_range_m,
                sigma_rate_m_s=est.sigma_range_rate_m_s,
                sigma_azimuth=np.deg2rad(est.sigma_azimuth_deg),
                single_look_position_m=est.position_rms_m,
            )
        )
    return truth


def _cartesian_measure(tx: np.ndarray, rx: np.ndarray, altitude: float, baseline: float) -> Callable:
    """Bi-static range, range rate and azimuth from state (x, y, vx, vy) at known altitude."""

    def measure(state: np.ndarray) -> np.ndarray:
        p = np.array([state[0], state[1], altitude])
        v = np.array([state[2], state[3], 0.0])
        rel_t, rel_r = p - tx, p - rx
        rt, rr = np.linalg.norm(rel_t), np.linalg.norm(rel_r)
        rate = v @ (rel_t / rt + rel_r / rr)
        return np.array([rt + rr - baseline, rate, np.arctan2(rel_r[1], rel_r[0])])

    return measure


def _draw_measurements(rng, truth_z: np.ndarray, sigmas: np.ndarray, pd: float, track, r, gate_sigma, density, pfa) -> np.ndarray:
    """One detection (if not missed) plus Poisson false alarms around the track's prediction."""
    z = []
    if rng.random() < pd:
        z.append(truth_z + rng.normal(0.0, sigmas))
    predicted = track.measure(track.x)
    h = track.jacobian()
    half = gate_sigma * np.sqrt(np.diag(h @ track.p @ h.T + r))
    # The detector resolves no finer than one cell, so a small gate still sees one cell's false alarms.
    expected = max(density * np.prod(2.0 * half), pfa)
    for _ in range(rng.poisson(min(expected, 200.0))):
        z.append(predicted + rng.uniform(-half, half))
    return np.array(z).reshape(-1, truth_z.size)


def _update(track: GaussianTrack, z: np.ndarray, r, gate_sigma, method, pd, density) -> bool:
    innovations, h, s = track.gate(r, z, gate_sigma)
    if method == "nearest":
        return track.update_nearest(innovations, h, s)
    return track.update_pda(innovations, h, s, pd, density)


def run_tracker_monte_carlo(analyzer: TargetAnalyzer, pfa: float, on_progress: Callable[[float], None] | None = None) -> TrackerMonteCarloResult:
    s = analyzer.s
    cfg, mc = s.tracker, s.monte_carlo
    dt, scans = cfg.update_interval_s, mc.scans
    truth = build_truth(analyzer, scans, pfa)
    wf = analyzer.waveform
    range_cell = C_M_S / wf.bandwidth_hz
    rate_cell = analyzer.lam * wf.doppler_resolution_hz
    density_rd = pfa / (range_cell * rate_cell)  # false alarms per m x m/s, per scan
    g0 = build_geometry(s)
    measure_xy = _cartesian_measure(g0.tx, g0.rx, g0.target[2], g0.baseline_m)

    f1, q1 = constant_velocity_model(dt, cfg.process_accel_m_s2)
    f2 = np.block([[np.eye(2), dt * np.eye(2)], [np.zeros((2, 2)), np.eye(2)]])
    a2 = cfg.process_accel_m_s2**2
    q2 = np.block(
        [[dt**4 / 4 * a2 * np.eye(2), dt**3 / 2 * a2 * np.eye(2)], [dt**3 / 2 * a2 * np.eye(2), dt**2 * a2 * np.eye(2)]]
    )

    rng = np.random.default_rng(mc.seed)
    err_range, err_rate, err_pos = np.full((mc.runs, scans), np.nan), np.full((mc.runs, scans), np.nan), np.full((mc.runs, scans), np.nan)
    lost = np.zeros(mc.runs, bool)

    for run in range(mc.runs):
        first = truth[0]
        r0 = np.diag([first.sigma_range_m**2, first.sigma_rate_m_s**2])
        x0 = np.array([first.range_m, first.rate_m_s]) + rng.normal(0.0, [first.sigma_range_m, first.sigma_rate_m_s])
        rd = GaussianTrack(x0, r0 * 4.0, f1, q1, lambda state: state)

        vxy = g0.velocity[:2]
        p0 = np.diag([(2.0 * first.single_look_position_m) ** 2] * 2 + [(0.2 * max(np.linalg.norm(vxy), 10.0)) ** 2] * 2)
        xy0 = np.r_[first.position[:2], vxy] + rng.normal(0.0, np.sqrt(np.diag(p0)))
        cart = GaussianTrack(xy0, p0, f2, q2, measure_xy, wrap_dims=(2,))

        misses = 0
        for k in range(scans):
            tr = truth[k]
            if k > 0:
                rd.predict()
                cart.predict()
            r_rd = np.diag([tr.sigma_range_m**2, tr.sigma_rate_m_s**2])
            r_xy = np.diag([tr.sigma_range_m**2, tr.sigma_rate_m_s**2, tr.sigma_azimuth**2])

            z_rd = _draw_measurements(rng, np.array([tr.range_m, tr.rate_m_s]), np.array([tr.sigma_range_m, tr.sigma_rate_m_s]),
                                      tr.pd, rd, r_rd, cfg.gate_sigma, density_rd, pfa)
            used = _update(rd, z_rd, r_rd, cfg.gate_sigma, cfg.association, tr.pd, density_rd)
            misses = 0 if used else misses + 1

            # The Cartesian filter shares the range-Doppler measurement; azimuth noise is added.
            if len(z_rd) > 0:
                azimuth = tr.azimuth + rng.normal(0.0, tr.sigma_azimuth, len(z_rd))
                z_xy = np.c_[z_rd, azimuth]
                density_xy = density_rd / (2.0 * np.pi)
                _update(cart, z_xy, r_xy, cfg.gate_sigma, cfg.association, tr.pd, density_xy)

            err_range[run, k] = rd.x[0] - tr.range_m
            err_rate[run, k] = rd.x[1] - tr.rate_m_s
            err_pos[run, k] = np.linalg.norm(cart.x[:2] - tr.position[:2])

            if misses > cfg.confirm_n or abs(err_range[run, k]) > 10.0 * np.sqrt(rd.p[0, 0] + r_rd[0, 0]):
                lost[run] = True
        if on_progress:
            on_progress((run + 1) / mc.runs)

    # Bound along the trajectory, reducing information by Pd and the clutter density.
    mean_pd = float(np.mean([t.pd for t in truth]))
    sig_r = float(np.mean([t.sigma_range_m for t in truth]))
    density_norm = pfa * sig_r / range_cell
    reduction = max(information_reduction_factor(mean_pd, density_norm), 1e-3)
    r_mean = np.diag([sig_r**2, float(np.mean([t.sigma_rate_m_s for t in truth])) ** 2])
    bound = pcrlb_series(f1, q1, np.eye(2), r_mean, reduction, r_mean * 4.0, scans)

    good = ~lost if (~lost).any() else np.ones(mc.runs, bool)
    return TrackerMonteCarloResult(
        time_s=np.arange(scans) * dt,
        rmse_range_m=np.sqrt(np.mean(err_range[good] ** 2, axis=0)),
        rmse_rate_m_s=np.sqrt(np.mean(err_rate[good] ** 2, axis=0)),
        bound_range_m=np.sqrt(bound[:, 0, 0]),
        bound_rate_m_s=np.sqrt(bound[:, 1, 1]),
        single_look_range_m=np.array([t.sigma_range_m for t in truth]),
        single_look_rate_m_s=np.array([t.sigma_rate_m_s for t in truth]),
        rmse_position_m=np.sqrt(np.mean(err_pos[good] ** 2, axis=0)),
        single_look_position_m=np.array([t.single_look_position_m for t in truth]),
        per_scan_pd=np.array([t.pd for t in truth]),
        continuity=float(np.mean(~lost)),
        lost_runs=int(lost.sum()),
        runs=mc.runs,
        seed=mc.seed,
        pfa_per_cell=pfa,
    )
