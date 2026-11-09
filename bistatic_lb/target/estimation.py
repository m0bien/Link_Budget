"""Stage 3b: how accurately target parameters can be estimated at the achieved SINR.

Cramer-Rao bounds with E/N0 as the SNR (Van Trees; Richards, ch. 7):
    sigma_tau = 1 / (2 pi B_rms sqrt(2 SNR))
    sigma_f   = 1 / (2 pi T_rms sqrt(2 SNR))
Angle bounds come from the array's Fisher information. Position follows from the
geometry Jacobian, which shows the degradation near the baseline.

The bounds only hold above the threshold SNR. Below it, noise peaks elsewhere in the
search space produce large outliers. ``outlier_threshold_snr_db`` gives a union-bound
estimate of that point, and estimation_mc.py shows it by simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..antenna.arrays import ReceiveArray
from ..config import Scenario
from ..units import C_M_S, undb, wavelength_m
from ..waveforms.base import Waveform
from .link import TargetLink


@dataclass(frozen=True)
class EstimationResult:
    snr_db: float
    bound_valid: bool
    threshold_snr_db: float
    sigma_delay_s: float
    sigma_bistatic_range_m: float
    sigma_doppler_hz: float
    sigma_range_rate_m_s: float
    sigma_azimuth_deg: float
    sigma_elevation_deg: float | None
    sigma_x_m: float
    sigma_y_m: float
    sigma_z_m: float | None
    position_rms_m: float
    rcs_measurement_std_db: float
    rcs_total_std_db: float
    range_resolution_m: float
    doppler_resolution_hz: float


def delay_crlb_s(snr_linear: float, rms_bandwidth_hz: float) -> float:
    return 1.0 / (2.0 * np.pi * rms_bandwidth_hz * np.sqrt(2.0 * snr_linear))


def doppler_crlb_hz(snr_linear: float, rms_duration_s: float) -> float:
    return 1.0 / (2.0 * np.pi * rms_duration_s * np.sqrt(2.0 * snr_linear))


def outlier_threshold_snr_db(n_cells: float, outlier_probability: float = 0.01) -> float:
    """SNR where the union bound (M/2) exp(-SNR/2) reaches the given outlier probability."""
    return float(10.0 * np.log10(2.0 * np.log(max(n_cells, 2.0) / (2.0 * outlier_probability))))


def _measurements(xyz: np.ndarray, tx: np.ndarray, rx: np.ndarray, use_elevation: bool) -> np.ndarray:
    rel_rx = xyz - rx
    rr = np.linalg.norm(rel_rx)
    rb = np.linalg.norm(xyz - tx) + rr - np.linalg.norm(tx - rx)
    az = np.arctan2(rel_rx[1], rel_rx[0])
    return np.array([rb, az, np.arcsin(rel_rx[2] / rr)]) if use_elevation else np.array([rb, az])


def position_covariance(
    xyz: np.ndarray, tx: np.ndarray, rx: np.ndarray, sigma_range_m: float, sigma_az_rad: float, sigma_el_rad: float | None
) -> np.ndarray:
    """Position covariance from (range, azimuth[, elevation]). Without elevation, altitude is known.

    Infinite on the diagonal when an angle cannot be measured, so the position is not resolved.
    """
    use_el = sigma_el_rad is not None
    n_state = 3 if use_el else 2
    jacobian = np.zeros((n_state, n_state))
    step = 1.0
    for i in range(n_state):
        delta = np.zeros(3)
        delta[i] = step
        jacobian[:, i] = (_measurements(xyz + delta, tx, rx, use_el) - _measurements(xyz - delta, tx, rx, use_el)) / (2 * step)
    variance = np.array([sigma_range_m**2, sigma_az_rad**2] + ([sigma_el_rad**2] if use_el else []))
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        # An unmeasured quantity leaves the position unresolved: range alone only gives an ellipse.
        return np.diag(np.full(n_state, np.inf))
    r_inverse = np.diag(1.0 / variance)
    return np.linalg.pinv(jacobian.T @ r_inverse @ jacobian)


def rcs_error_db(snr_linear: float, fluctuation_m: float | None, looks: int = 1) -> tuple[float, float]:
    """Standard deviation of an RCS estimate: from noise alone, and with target fluctuation.

    Noise part: var(P)/P^2 is about (2/SNR + 1/SNR^2) per look. Fluctuation part: a
    chi-square target with parameter m adds 1/m. Converted to dB with 4.34 x relative std.
    """
    noise_relative = np.sqrt((2.0 / snr_linear + 1.0 / snr_linear**2) / looks)
    fluctuation_relative = 0.0 if fluctuation_m is None else np.sqrt(1.0 / (fluctuation_m * looks))
    return float(4.34 * noise_relative), float(4.34 * np.sqrt(noise_relative**2 + fluctuation_relative**2))


def compute_estimation(
    s: Scenario, array: ReceiveArray, waveform: Waveform, link: TargetLink, n_cells: float, fluctuation_m: float | None
) -> EstimationResult:
    snr = float(undb(link.sinr_db))
    lam = wavelength_m(s.frequency_hz)
    sigma_tau = delay_crlb_s(snr, waveform.rms_bandwidth_hz)
    sigma_f = doppler_crlb_hz(snr, waveform.rms_duration_s)
    g = link.geometry

    # Angle: the bound needs E/N0 for one isotropic channel, so remove the array gain.
    gain_linear = float(undb(link.rx_gain_dbi))
    sigma_az, sigma_el = array.angle_crlb_rad(snr / gain_linear, g.azimuth_at_rx_rad, g.elevation_at_rx_rad)

    sigma_range = C_M_S * sigma_tau
    cov = position_covariance(g.target, g.tx, g.rx, sigma_range, sigma_az, sigma_el)
    threshold = outlier_threshold_snr_db(n_cells)
    noise_part, total_part = rcs_error_db(snr, fluctuation_m, s.detection.noncoherent_looks)
    return EstimationResult(
        snr_db=link.sinr_db,
        bound_valid=link.sinr_db >= threshold,
        threshold_snr_db=threshold,
        sigma_delay_s=sigma_tau,
        sigma_bistatic_range_m=sigma_range,
        sigma_doppler_hz=sigma_f,
        sigma_range_rate_m_s=sigma_f * lam,
        sigma_azimuth_deg=float(np.rad2deg(sigma_az)),
        sigma_elevation_deg=None if sigma_el is None else float(np.rad2deg(sigma_el)),
        sigma_x_m=float(np.sqrt(cov[0, 0])),
        sigma_y_m=float(np.sqrt(cov[1, 1])),
        sigma_z_m=float(np.sqrt(cov[2, 2])) if cov.shape[0] == 3 else None,
        position_rms_m=float(np.sqrt(np.trace(cov))),
        rcs_measurement_std_db=noise_part,
        rcs_total_std_db=total_part,
        range_resolution_m=g.range_resolution_m(waveform.bandwidth_hz),
        doppler_resolution_hz=waveform.doppler_resolution_hz,
    )
