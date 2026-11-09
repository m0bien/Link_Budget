"""Tropospheric refraction: effective Earth radius and a ducting check (P.453).

k = 157 / (157 + dN/dh). A gradient below -157 N/km bends rays faster than the Earth
curves, which is the condition for a duct and can carry signals well past the horizon.
"""

from __future__ import annotations

EARTH_RADIUS_KM = 6371.0
DUCT_GRADIENT_N_PER_KM = -157.0
MAX_K_FACTOR = 50.0  # k grows without limit near the duct threshold


def k_factor(gradient_n_per_km: float) -> float:
    if gradient_n_per_km <= DUCT_GRADIENT_N_PER_KM + 157.0 / MAX_K_FACTOR:
        return MAX_K_FACTOR
    return 157.0 / (157.0 + gradient_n_per_km)


def effective_earth_radius_km(gradient_n_per_km: float) -> float:
    return k_factor(gradient_n_per_km) * EARTH_RADIUS_KM


def is_ducting(gradient_n_per_km: float) -> bool:
    return gradient_n_per_km <= DUCT_GRADIENT_N_PER_KM
