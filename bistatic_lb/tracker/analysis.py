"""Stage 4, analytic part: tracked performance against single-look performance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Scenario
from ..target.detection import Fluctuation
from ..target.estimation import EstimationResult
from ..target.link import DetectionSummary
from ..units import C_M_S
from ..waveforms.base import Waveform
from .bounds import alpha_beta_gains, constant_velocity_model, information_reduction_factor, steady_state_covariance
from .initiation import ThresholdTrade, confirm_probability, false_track_rate_per_hour, scans_to_confirm, threshold_trade


@dataclass(frozen=True)
class TrackerResult:
    single_look_pd: float
    single_look_pfa: float
    track_confirm_probability: float
    confirm_probability_at_relaxed_threshold: float
    relaxed_pd: float
    mean_scans_to_confirm: float
    mean_time_to_confirm_s: float
    false_tracks_per_hour: float
    single_look_false_alarms_per_hour: float
    gate_cells: float
    trade: ThresholdTrade
    information_reduction: float
    single_look_sigma_range_m: float
    tracked_sigma_range_m: float
    single_look_sigma_rate_m_s: float
    tracked_sigma_rate_m_s: float
    range_improvement_db: float
    rate_improvement_db: float
    tracking_index: float
    alpha: float
    beta: float
    latency_note: str


def gate_size_cells(s: Scenario, waveform: Waveform, estimation: EstimationResult, wavelength_m: float) -> float:
    """Range-Doppler cells inside a track's gate at the first follow-up scan."""
    t = s.tracker
    scan = t.update_interval_s
    range_cell = C_M_S / waveform.bandwidth_hz
    rate_cell = wavelength_m * waveform.doppler_resolution_hz
    sigma_range = np.hypot(estimation.sigma_bistatic_range_m, estimation.sigma_range_rate_m_s * scan)
    sigma_rate = np.hypot(estimation.sigma_range_rate_m_s, t.process_accel_m_s2 * scan)
    return float((2.0 * t.gate_sigma * sigma_range / range_cell + 1.0) * (2.0 * t.gate_sigma * sigma_rate / rate_cell + 1.0))


def compute_tracker(
    s: Scenario,
    waveform: Waveform,
    detection: DetectionSummary,
    estimation: EstimationResult,
    fluctuation: Fluctuation,
    pd_at_pfa,
    wavelength_m: float,
) -> TrackerResult:
    t = s.tracker
    m, n = t.confirm_m, t.confirm_n
    if m > n:
        raise ValueError("Track confirmation needs M <= N.")
    gate = gate_size_cells(s, waveform, estimation, wavelength_m)

    trade = threshold_trade(
        detection.pfa, t.target_confirm_probability, m, n, s.detection.noncoherent_looks, fluctuation,
        detection.n_cells, gate, t.update_interval_s, t.allowed_false_tracks_per_hour,
    )
    relaxed_pd = pd_at_pfa(estimation.snr_db, trade.best_pfa)
    confirm_at_design = confirm_probability(detection.pd, m, n)
    confirm_relaxed = confirm_probability(relaxed_pd, m, n)
    mean_scans, _ = scans_to_confirm(relaxed_pd, m, n) if relaxed_pd > 1e-3 else (np.inf, 0.0)

    # Tracking accuracy with the measurement noise at the achieved SNR.
    f, q = constant_velocity_model(t.update_interval_s, t.process_accel_m_s2)
    h = np.eye(2)
    r = np.diag([estimation.sigma_bistatic_range_m**2, estimation.sigma_range_rate_m_s**2])

    rate_cell = wavelength_m * waveform.doppler_resolution_hz
    range_cell = C_M_S / waveform.bandwidth_hz
    density = trade.best_pfa * estimation.sigma_bistatic_range_m / range_cell
    density = max(density, trade.best_pfa * estimation.sigma_range_rate_m_s / rate_cell)
    reduction = information_reduction_factor(relaxed_pd, density)
    reduction = max(reduction, 1e-3)
    _, updated = steady_state_covariance(f, q, h, r / reduction)

    index, alpha, beta = alpha_beta_gains(estimation.sigma_bistatic_range_m, t.process_accel_m_s2, t.update_interval_s)
    sigma_range = float(np.sqrt(updated[0, 0]))
    sigma_rate = float(np.sqrt(updated[1, 1]))
    frame = waveform.cpi_s * s.detection.noncoherent_looks
    return TrackerResult(
        single_look_pd=detection.pd,
        single_look_pfa=detection.pfa,
        track_confirm_probability=confirm_at_design,
        confirm_probability_at_relaxed_threshold=confirm_relaxed,
        relaxed_pd=relaxed_pd,
        mean_scans_to_confirm=mean_scans,
        mean_time_to_confirm_s=mean_scans * t.update_interval_s,
        false_tracks_per_hour=trade.best_false_tracks_per_hour,
        single_look_false_alarms_per_hour=detection.pfa * detection.n_cells / frame * 3600.0,
        gate_cells=gate,
        trade=trade,
        information_reduction=reduction,
        single_look_sigma_range_m=estimation.sigma_bistatic_range_m,
        tracked_sigma_range_m=sigma_range,
        single_look_sigma_rate_m_s=estimation.sigma_range_rate_m_s,
        tracked_sigma_rate_m_s=sigma_rate,
        range_improvement_db=float(20.0 * np.log10(estimation.sigma_bistatic_range_m / sigma_range)),
        rate_improvement_db=float(20.0 * np.log10(estimation.sigma_range_rate_m_s / sigma_rate)),
        tracking_index=index,
        alpha=alpha,
        beta=beta,
        latency_note=f"Confirmation takes about {mean_scans:.1f} scans ({mean_scans * t.update_interval_s:.1f} s). This is the price of the gain.",
    )


__all__ = ["TrackerResult", "compute_tracker", "false_track_rate_per_hour", "gate_size_cells"]
