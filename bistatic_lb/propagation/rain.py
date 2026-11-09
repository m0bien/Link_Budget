"""Rain attenuation: specific attenuation from P.838-3, path statistics from P.530-19 sec. 2.4.1."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

P838_MIN_GHZ = 1.0


@lru_cache(maxsize=512)
def _coefficients(f_ghz: float, vertical: bool) -> tuple[float, float]:
    from itur.models import itu838

    k, alpha = itu838.rain_specific_attenuation_coefficients(f_ghz, 0.0, 90.0 if vertical else 0.0)
    return float(k), float(alpha)


def rain_specific_attenuation_db_km(frequency_hz: float, rain_rate_mm_h: float, vertical: bool = True) -> float:
    """gamma_R = k R^alpha; zero below the 1 GHz start of P.838."""
    f_ghz = round(frequency_hz / 1e9, 4)
    if rain_rate_mm_h <= 0.0 or f_ghz < P838_MIN_GHZ:
        return 0.0
    k, alpha = _coefficients(f_ghz, vertical)
    return k * rain_rate_mm_h**alpha


def rain_attenuation_db(
    frequency_hz: float, r001_mm_h: float, distance_km: float, time_percent: float, vertical: bool = True
) -> float:
    """Path attenuation exceeded for time_percent of the time, P.530-19 eq. 32-35."""
    f_ghz = frequency_hz / 1e9
    if r001_mm_h <= 0.0 or f_ghz < P838_MIN_GHZ or distance_km <= 0.0:
        return 0.0
    _, alpha = _coefficients(round(f_ghz, 4), vertical)
    gamma = rain_specific_attenuation_db_km(frequency_hz, r001_mm_h, vertical)

    # Eq. 32, with the Recommendation's stated limits on f, d and R inside r only.
    f_r, d_r, r_r = max(f_ghz, 6.0), min(distance_km, 30.0), max(r001_mm_h, 3.0)
    r = 1.0 / (0.477 * d_r**0.633 * r_r ** (0.073 * alpha) * f_r**0.123 - 10.579 * (1.0 - np.exp(-0.024 * d_r)))
    a_001 = gamma * distance_km * r  # eq. 33

    p = float(np.clip(time_percent, 0.001, 1.0))
    c0 = 0.12 + 0.4 * np.log10(f_ghz / 10.0) ** 0.8 if f_ghz >= 10.0 else 0.12
    c1 = 0.07**c0 * 0.12 ** (1.0 - c0)
    c2 = 0.855 * c0 + 0.546 * (1.0 - c0)
    c3 = 0.139 * c0 + 0.043 * (1.0 - c0)
    return float(a_001 * c1 * p ** (-(c2 + c3 * np.log10(p))))  # eq. 34-35
