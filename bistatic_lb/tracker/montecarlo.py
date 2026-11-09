"""Stage 4, simulation: run the track filters on detections drawn from the link budget.

A target flies a straight line, optionally starting a turn halfway through. At every scan
the link budget gives the SINR, and so the detection probability and measurement accuracy,
at that point. Each run draws detections and false alarms from those numbers and passes
them through the association and filters.

Track initiation. By default a track is not handed the truth: it starts at the target's
first detection, from that one measurement, and must then collect M hits within N scans
before it counts as confirmed. A tentative track that sees nothing for N scans is dropped
and the next detection starts a new one. Errors are recorded only while a track is
confirmed. ``initiation = "on_truth"`` instead starts every track confirmed on the true
state, which isolates filter accuracy. Tracks started by false alarms are not simulated;
their rate comes from the analytic false-track calculation.

Altitude. When the array cannot measure elevation, the Cartesian filter needs an assumed
altitude. Each run offsets it by a draw from the stated altitude uncertainty.

Results are compared with the single-look values and the PCRLB. The PCRLB assumes a track
since the first scan, so it sits below what a late-starting track can reach early on.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from ..config import Scenario
from ..geometry import BistaticGeometry, build_geometry
from ..target.estimation import compute_estimation
from ..target.link import TargetAnalyzer
from ..units import C_M_S
from .bounds import constant_velocity_model, information_reduction_factor, pcrlb_series
from .filters import GaussianTrack

MAX_POSITION_SIGMA_M = 1e6  # stands in for "not resolved" so the filters stay finite
MIN_RUNS_FOR_RMSE = 5  # scans with fewer confirmed runs than this report no error


@dataclass(frozen=True)
class TruthAtScan:
    position: np.ndarray
    velocity: np.ndarray
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
    rmse_range_m: np.ndarray  # NaN at scans with too few confirmed tracks
    rmse_rate_m_s: np.ndarray
    bound_range_m: np.ndarray
    bound_rate_m_s: np.ndarray
    single_look_range_m: np.ndarray
    single_look_rate_m_s: np.ndarray
    rmse_position_m: np.ndarray
    single_look_position_m: np.ndarray
    per_scan_pd: np.ndarray
    continuity: float  # confirmed tracks that were never lost
    lost_runs: int
    runs: int
    seed: int
    pfa_per_cell: float
    initiation: str = "on_truth"
    confirmed_runs: int = 0
    mean_scans_to_confirm: float = float("nan")  # counted from the first scan, over the runs that confirmed
    tracked_runs: np.ndarray | None = None  # confirmed, unlost tracks at each scan

    @property
    def confirmed_fraction(self) -> float:
        return self.confirmed_runs / self.runs


def build_truth(analyzer: TargetAnalyzer, scans: int, pfa: float) -> list[TruthAtScan]:
    """The target's path and what the link budget gives at each scan."""
    s: Scenario = analyzer.s
    g0 = build_geometry(s)
    dt = s.tracker.update_interval_s
    turn = s.monte_carlo.manoeuvre_accel_m_s2
    n_cells = analyzer.number_of_search_cells()
    position, velocity = g0.target.copy(), g0.velocity.copy()
    speed = float(np.linalg.norm(velocity))
    truth = []
    for k in range(scans):
        g = BistaticGeometry(tx=g0.tx, rx=g0.rx, target=position.copy(), velocity=velocity.copy())
        # The link takes its Doppler from the scenario heading, so pass the true one for a turning target.
        link = analyzer.link(position, doppler_hz=g.doppler_hz(analyzer.lam))
        est = compute_estimation(s, analyzer.array, analyzer.waveform, link, n_cells, None)
        truth.append(
            TruthAtScan(
                position=position.copy(),
                velocity=velocity.copy(),
                range_m=g.bistatic_range_m,
                rate_m_s=g.bistatic_range_rate_m_s,
                azimuth=g.azimuth_at_rx_rad,
                pd=analyzer.detection_at(link.sinr_db, pfa),
                sigma_range_m=est.sigma_bistatic_range_m,
                sigma_rate_m_s=est.sigma_range_rate_m_s,
                # An unmeasurable azimuth is as good as uniform over the circle.
                sigma_azimuth=min(np.deg2rad(est.sigma_azimuth_deg), np.pi),
                single_look_position_m=min(est.position_rms_m, MAX_POSITION_SIGMA_M),
            )
        )
        if turn > 0.0 and speed > 0.0 and k >= scans // 2:
            # A level turn at constant speed: the velocity rotates at accel / speed.
            angle = turn / speed * dt
            c, sn = np.cos(angle), np.sin(angle)
            turned = np.array([c * velocity[0] - sn * velocity[1], sn * velocity[0] + c * velocity[1], velocity[2]])
            position = position + 0.5 * (velocity + turned) * dt
            velocity = turned
        else:
            position = position + velocity * dt
    return truth


def _cartesian_measure(tx: np.ndarray, rx: np.ndarray, altitude: float, baseline: float) -> Callable:
    """Bi-static range, range rate and azimuth from state (x, y, vx, vy) at an assumed altitude."""

    def measure(state: np.ndarray) -> np.ndarray:
        p = np.array([state[0], state[1], altitude])
        v = np.array([state[2], state[3], 0.0])
        rel_t, rel_r = p - tx, p - rx
        rt, rr = np.linalg.norm(rel_t), np.linalg.norm(rel_r)
        rate = v @ (rel_t / rt + rel_r / rr)
        return np.array([rt + rr - baseline, rate, np.arctan2(rel_r[1], rel_r[0])])

    return measure


def position_from_measurement(tx: np.ndarray, rx: np.ndarray, altitude: float, baseline: float, range_m: float, azimuth: float) -> np.ndarray:
    """The point (x, y) at this altitude where the bi-static range and the azimuth from the receiver meet."""
    direction = np.array([np.cos(azimuth), np.sin(azimuth)])

    def excess(ground_range: float) -> float:
        p = np.array([rx[0] + ground_range * direction[0], rx[1] + ground_range * direction[1], altitude])
        return np.linalg.norm(p - tx) + np.linalg.norm(p - rx) - baseline - range_m

    reach = range_m + baseline + abs(altitude)
    if excess(0.0) >= 0.0:  # the measured range is shorter than any point on this bearing allows
        return rx[:2].copy()
    return rx[:2] + brentq(excess, 0.0, reach) * direction


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


def _associated(track: GaussianTrack, z: np.ndarray, r, gate_sigma) -> int | None:
    """Index of the measurement closest to the track's prediction inside its gate, or None."""
    if len(z) == 0:
        return None
    h = track.jacobian()
    s_inverse = np.linalg.inv(h @ track.p @ h.T + r)
    nu = z - track.measure(track.x)
    d2 = np.einsum("ni,ij,nj->n", nu, s_inverse, nu)
    best = int(np.argmin(d2))
    return best if d2[best] <= gate_sigma**2 else None


def _rms(errors: np.ndarray, use: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RMS over the runs marked in ``use`` at each scan, and how many runs that was."""
    masked = np.where(use, errors, np.nan)
    count = np.sum(np.isfinite(masked), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # scans where no run has a confirmed track
        rms = np.sqrt(np.nanmean(masked**2, axis=0))
    # Too few tracks make a noisy average, unless the whole simulation is that small.
    rms[count < max(min(MIN_RUNS_FOR_RMSE, int(use[:, 0].sum())), 1)] = np.nan
    return rms, count


def run_tracker_monte_carlo(analyzer: TargetAnalyzer, pfa: float, on_progress: Callable[[float], None] | None = None) -> TrackerMonteCarloResult:
    s = analyzer.s
    cfg, mc = s.tracker, s.monte_carlo
    dt, scans = cfg.update_interval_s, mc.scans
    m_hits, n_window = cfg.confirm_m, cfg.confirm_n
    from_detections = mc.initiation == "from_detections"
    truth = build_truth(analyzer, scans, pfa)
    wf = analyzer.waveform
    range_cell = C_M_S / wf.bandwidth_hz
    rate_cell = analyzer.lam * wf.doppler_resolution_hz
    density_rd = pfa / (range_cell * rate_cell)  # false alarms per m x m/s, per scan

    g0 = build_geometry(s)
    altitude_sigma = 0.0 if analyzer.array.has_elevation else s.geometry.target_altitude_sigma_m
    f1, q1 = constant_velocity_model(dt, cfg.process_accel_m_s2)
    f2 = np.block([[np.eye(2), dt * np.eye(2)], [np.zeros((2, 2)), np.eye(2)]])
    a2 = cfg.process_accel_m_s2**2
    q2 = np.block(
        [[dt**4 / 4 * a2 * np.eye(2), dt**3 / 2 * a2 * np.eye(2)], [dt**3 / 2 * a2 * np.eye(2), dt**2 * a2 * np.eye(2)]]
    )

    rng = np.random.default_rng(mc.seed)
    shape = (mc.runs, scans)
    err_range, err_rate, err_pos = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
    confirmed_at = np.full(mc.runs, np.nan)
    lost = np.zeros(mc.runs, bool)

    for run in range(mc.runs):
        assumed_altitude = g0.target[2] + (rng.normal(0.0, altitude_sigma) if altitude_sigma > 0.0 else 0.0)
        measure_xy = _cartesian_measure(g0.tx, g0.rx, assumed_altitude, g0.baseline_m)
        rd = cart = None
        hits: list[bool] = []
        confirmed = False
        misses = 0

        for k in range(scans):
            tr = truth[k]
            r_rd = np.diag([tr.sigma_range_m**2, tr.sigma_rate_m_s**2])
            r_xy = np.diag([tr.sigma_range_m**2, tr.sigma_rate_m_s**2, tr.sigma_azimuth**2])
            sigmas = np.array([tr.sigma_range_m, tr.sigma_rate_m_s])
            truth_rd = np.array([tr.range_m, tr.rate_m_s])
            position_sigma = 2.0 * tr.single_look_position_m

            if rd is None and not from_detections:
                # Best case: a confirmed track on the true state, with single-look uncertainty.
                rd = GaussianTrack(truth_rd + rng.normal(0.0, sigmas), r_rd * 4.0, f1, q1, lambda state: state)
                vxy = tr.velocity[:2]
                p0 = np.diag([position_sigma**2] * 2 + [(0.2 * max(np.linalg.norm(vxy), 10.0)) ** 2] * 2)
                cart = GaussianTrack(np.r_[tr.position[:2], vxy] + rng.normal(0.0, np.sqrt(np.diag(p0))), p0, f2, q2, measure_xy, wrap_dims=(2,))
                confirmed, confirmed_at[run] = True, 1.0
                used = True
            elif rd is None:
                # No track yet: the target's next detection starts one, from that measurement alone.
                if rng.random() >= tr.pd:
                    continue
                z0 = truth_rd + rng.normal(0.0, sigmas)
                az0 = tr.azimuth + rng.normal(0.0, tr.sigma_azimuth)
                rd = GaussianTrack(z0, r_rd.copy(), f1, q1, lambda state: state)
                xy0 = position_from_measurement(g0.tx, g0.rx, assumed_altitude, g0.baseline_m, z0[0], az0)
                p0 = np.diag([position_sigma**2] * 2 + [s.detection.max_speed_m_s**2] * 2)
                cart = GaussianTrack(np.r_[xy0, 0.0, 0.0], p0, f2, q2, measure_xy, wrap_dims=(2,))
                # The range rate measured with it already says something about the velocity.
                cart.update_unscented(np.array([z0[0], z0[1], az0]), r_xy)
                hits, used = [True], True
            else:
                rd.predict()
                cart.predict()
                z_rd = _draw_measurements(rng, truth_rd, sigmas, tr.pd, rd, r_rd, cfg.gate_sigma, density_rd, pfa)
                # Association is decided in range-Doppler, where the track is sharp. The Cartesian
                # filter then converts that one measurement, with its azimuth, into a position. It
                # uses an unscented update: range and range rate are far sharper than the azimuth.
                chosen = _associated(rd, z_rd, r_rd, cfg.gate_sigma)
                used = _update(rd, z_rd, r_rd, cfg.gate_sigma, cfg.association, tr.pd, density_rd)
                if chosen is not None:
                    azimuth = tr.azimuth + rng.normal(0.0, tr.sigma_azimuth)
                    cart.update_unscented(np.array([z_rd[chosen, 0], z_rd[chosen, 1], azimuth]), r_xy)
                hits.append(used)

            if not confirmed:
                recent = hits[-n_window:]
                if sum(recent) >= m_hits:
                    confirmed, confirmed_at[run] = True, k + 1.0
                elif len(recent) == n_window and not any(recent):
                    rd = cart = None  # nothing for a whole window: drop it and wait for the next detection
                    hits = []
                    continue
            if not confirmed:
                continue

            misses = 0 if used else misses + 1
            err_range[run, k] = rd.x[0] - tr.range_m
            err_rate[run, k] = rd.x[1] - tr.rate_m_s
            err_pos[run, k] = np.linalg.norm(cart.x[:2] - tr.position[:2])
            if misses > n_window or abs(err_range[run, k]) > 10.0 * np.sqrt(rd.p[0, 0] + r_rd[0, 0]):
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

    was_confirmed = np.isfinite(confirmed_at)
    good = was_confirmed & ~lost
    use = np.broadcast_to((good if good.any() else was_confirmed)[:, None], shape)
    rmse_range, tracked = _rms(err_range, use)
    return TrackerMonteCarloResult(
        time_s=np.arange(scans) * dt,
        rmse_range_m=rmse_range,
        rmse_rate_m_s=_rms(err_rate, use)[0],
        bound_range_m=np.sqrt(bound[:, 0, 0]),
        bound_rate_m_s=np.sqrt(bound[:, 1, 1]),
        single_look_range_m=np.array([t.sigma_range_m for t in truth]),
        single_look_rate_m_s=np.array([t.sigma_rate_m_s for t in truth]),
        rmse_position_m=_rms(err_pos, use)[0],
        single_look_position_m=np.array([t.single_look_position_m for t in truth]),
        per_scan_pd=np.array([t.pd for t in truth]),
        continuity=float(np.mean(~lost[was_confirmed])) if was_confirmed.any() else 0.0,
        lost_runs=int(lost.sum()),
        runs=mc.runs,
        seed=mc.seed,
        pfa_per_cell=pfa,
        initiation=mc.initiation,
        confirmed_runs=int(was_confirmed.sum()),
        mean_scans_to_confirm=float(np.mean(confirmed_at[was_confirmed])) if was_confirmed.any() else float("nan"),
        tracked_runs=tracked,
    )
