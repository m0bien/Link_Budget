"""Diffraction from Rec. ITU-R P.526-16 (equation numbers refer to that edition).

* spherical_earth_loss_db: sec. 3.2, any distance, frequencies of 10 MHz and above
* knife_edge_j_db: eq. 31, plus the exact Fresnel form as a cross-check
* bullington_loss_db: sec. 4.5.1, eq. 49-57, for a terrain profile
* general_path_loss_db: sec. 4.5.2 combination, simplified as noted on the function
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import fresnel

from ..units import wavelength_m
from .surface import SurfaceConstants


@dataclass(frozen=True)
class SphericalEarthResult:
    loss_db: float
    clearance_m: float
    required_clearance_m: float
    los_distance_km: float
    line_of_sight: bool
    case: str


def _surface_admittance(f_mhz: float, a_km: float, surface: SurfaceConstants, vertical: bool) -> float:
    """Normalized surface admittance K, eq. 11a and 12a."""
    term = (18000.0 * surface.conductivity_s_m / f_mhz) ** 2
    k_h = 0.36 * (a_km * f_mhz) ** (-1.0 / 3.0) * ((surface.permittivity - 1.0) ** 2 + term) ** (-0.25)
    return float(k_h * np.sqrt(surface.permittivity**2 + term)) if vertical else float(k_h)


def _beta(k: float, f_mhz: float, surface: SurfaceConstants, vertical: bool) -> float:
    """Eq. 16. May be taken as 1 for horizontal polarization, and for vertical above 20 MHz over land or 300 MHz over sea."""
    limit_mhz = 300.0 if surface.is_sea else 20.0
    if not vertical or f_mhz > limit_mhz:
        return 1.0
    return float((1.0 + 1.6 * k**2 + 0.67 * k**4) / (1.0 + 4.5 * k**2 + 1.53 * k**4))


def _height_gain_db(y: float, beta: float, k: float) -> float:
    """G(Y), eq. 18, 18a, 18b with the lower limit 2 + 20 log K."""
    b = beta * y
    g = (
        17.6 * np.sqrt(b - 1.1) - 5.0 * np.log10(b - 1.1) - 8.0
        if b > 2.0
        else 20.0 * np.log10(b + 0.1 * b**3)
    )
    return float(max(g, 2.0 + 20.0 * np.log10(k)))


def _distance_term_db(x: float) -> float:
    """F(X), eq. 17a and 17b."""
    if x >= 1.6:
        return float(11.0 + 10.0 * np.log10(x) - 17.6 * x)
    return float(-20.0 * np.log10(x) - 5.6488 * x**1.425)


def _field_relative_db(d_km, h1_m, h2_m, f_mhz, a_km, surface, vertical) -> float:
    """20 log(E/E0) = F(X) + G(Y1) + G(Y2), eq. 13, 14a, 15a."""
    k = _surface_admittance(f_mhz, a_km, surface, vertical)
    beta = _beta(k, f_mhz, surface, vertical)
    x = 2.188 * beta * f_mhz ** (1.0 / 3.0) * a_km ** (-2.0 / 3.0) * d_km
    y_factor = 9.575e-3 * beta * f_mhz ** (2.0 / 3.0) * a_km ** (-1.0 / 3.0)
    return _distance_term_db(x) + _height_gain_db(y_factor * h1_m, beta, k) + _height_gain_db(y_factor * h2_m, beta, k)


def reflection_point_distances_m(d_m: float, h1_m: float, h2_m: float, a_m: float) -> tuple[float, float]:
    """Distances to the lowest-clearance point on a spherical Earth, eq. 22a-22e."""
    m = d_m**2 / (4.0 * a_m * (h1_m + h2_m))
    c = (h1_m - h2_m) / (h1_m + h2_m)
    argument = np.clip(1.5 * c * np.sqrt(3.0 * m / (m + 1.0) ** 3), -1.0, 1.0)
    b = 2.0 * np.sqrt((m + 1.0) / (3.0 * m)) * np.cos(np.pi / 3.0 + np.arccos(argument) / 3.0)
    d1 = d_m / 2.0 * (1.0 + b)
    return float(d1), float(d_m - d1)


def spherical_earth_loss_db(
    d_km: float, h1_m: float, h2_m: float, frequency_hz: float, a_km: float, surface: SurfaceConstants, vertical: bool = True
) -> SphericalEarthResult:
    """Diffraction loss for any distance, sec. 3.2. Heights are above the local surface."""
    f_mhz = frequency_hz / 1e6
    lam = wavelength_m(frequency_hz)
    d, a = d_km * 1e3, a_km * 1e3
    h1, h2 = max(h1_m, 0.01), max(h2_m, 0.01)

    d_los = np.sqrt(2.0 * a) * (np.sqrt(h1) + np.sqrt(h2))  # eq. 21
    if d >= d_los:
        loss = max(-_field_relative_db(d_km, h1, h2, f_mhz, a_km, surface, vertical), 0.0)
        return SphericalEarthResult(loss, 0.0, np.nan, d_los / 1e3, False, "beyond the radio horizon")

    d1, d2 = reflection_point_distances_m(d, h1, h2, a)
    clearance = ((h1 - d1**2 / (2.0 * a)) * d2 + (h2 - d2**2 / (2.0 * a)) * d1) / d  # eq. 22
    h_req = 0.552 * np.sqrt(d1 * d2 * lam / d)  # eq. 23
    if clearance > h_req:
        return SphericalEarthResult(0.0, clearance, h_req, d_los / 1e3, True, "clear line of sight")

    a_em = 0.5 * (d / (np.sqrt(h1) + np.sqrt(h2))) ** 2  # eq. 24
    a_h = -_field_relative_db(d_km, h1, h2, f_mhz, a_em / 1e3, surface, vertical)
    if a_h < 0.0:
        return SphericalEarthResult(0.0, clearance, h_req, d_los / 1e3, True, "marginal, but no net diffraction loss")
    loss = (1.0 - clearance / h_req) * a_h  # eq. 25
    return SphericalEarthResult(float(loss), clearance, h_req, d_los / 1e3, True, "line of sight with too little clearance")


def knife_edge_j_db(nu):
    """J(nu), eq. 31, zero below nu = -0.78."""
    nu = np.asarray(nu, float)
    term = nu - 0.1
    return np.where(nu > -0.78, 6.9 + 20.0 * np.log10(np.sqrt(term**2 + 1.0) + term), 0.0)


def knife_edge_exact_db(nu):
    """Exact knife-edge loss from Fresnel integrals, for checking eq. 31."""
    s, c = fresnel(np.asarray(nu, float))
    return -20.0 * np.log10(np.sqrt((1.0 - c - s) ** 2 + (c - s) ** 2) / 2.0)


def bullington_loss_db(
    distances_km: np.ndarray, heights_m: np.ndarray, h_ts_m: float, h_rs_m: float, frequency_hz: float, a_km: float
) -> float:
    """Bullington loss, eq. 49-57. Intermediate profile points only; heights above sea level."""
    d = float(distances_km[-1])
    di, hi = np.asarray(distances_km[1:-1], float), np.asarray(heights_m[1:-1], float)
    if di.size == 0:
        return 0.0
    lam = wavelength_m(frequency_hz)
    ce = 1.0 / a_km
    bulge = hi + 500.0 * ce * di * (d - di)

    s_tim = np.max((bulge - h_ts_m) / di)  # eq. 49
    s_tr = (h_rs_m - h_ts_m) / d  # eq. 50
    if s_tim < s_tr:  # line of sight
        nu = (bulge - (h_ts_m * (d - di) + h_rs_m * di) / d) * np.sqrt(0.002 * d / (lam * di * (d - di)))  # eq. 51
        l_uc = float(knife_edge_j_db(np.max(nu)))  # eq. 52
    else:  # trans-horizon
        s_rim = np.max((bulge - h_rs_m) / (d - di))  # eq. 53
        d_b = (h_rs_m - h_ts_m + s_rim * d) / (s_tim + s_rim)  # eq. 54
        nu_b = (h_ts_m + s_tim * d_b - (h_ts_m * (d - d_b) + h_rs_m * d_b) / d) * np.sqrt(
            0.002 * d / (lam * d_b * (d - d_b))
        )  # eq. 55
        l_uc = float(knife_edge_j_db(nu_b))  # eq. 56
    return float(l_uc + (1.0 - np.exp(-l_uc / 6.0)) * (10.0 + 0.02 * d))  # eq. 57


def general_path_loss_db(
    distances_km, heights_m, h_tx_agl_m, h_rx_agl_m, frequency_hz, a_km, surface, vertical=True
) -> tuple[float, SphericalEarthResult]:
    """Combine the Bullington and spherical losses, following sec. 4.5.2.

    L = L_ba + max(L_sph - L_bs, 0). Simplification: effective heights for the
    spherical part are the heights above the mean line between the terminals' ground
    levels, not the Recommendation's least-squares smooth-surface fit.
    """
    distances_km = np.asarray(distances_km, float)
    heights_m = np.asarray(heights_m, float)
    d = float(distances_km[-1])
    h_ts, h_rs = heights_m[0] + h_tx_agl_m, heights_m[-1] + h_rx_agl_m
    l_ba = bullington_loss_db(distances_km, heights_m, h_ts, h_rs, frequency_hz, a_km)

    mean_ground = float(np.mean([heights_m[0], heights_m[-1]]))
    h_te, h_re = max(h_ts - mean_ground, 1.0), max(h_rs - mean_ground, 1.0)
    smooth_profile = np.zeros_like(heights_m)
    l_bs = bullington_loss_db(distances_km, smooth_profile, h_te, h_re, frequency_hz, a_km)
    sph = spherical_earth_loss_db(d, h_te, h_re, frequency_hz, a_km, surface, vertical)
    return float(l_ba + max(sph.loss_db - l_bs, 0.0)), sph
