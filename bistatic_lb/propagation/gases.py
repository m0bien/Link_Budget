"""Gaseous attenuation, Rec. ITU-R P.676, through the ITU-Rpy package.

P.676 starts at 1 GHz. Below that the value at 1 GHz is used: it is an upper bound,
and it is tiny either way. Zenith attenuation uses equivalent heights of 6 km for
oxygen and 2 km for water vapour, a simplification of the P.676 slant-path method.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import EnvironmentConfig

P676_MIN_GHZ = 1.0
OXYGEN_HEIGHT_KM = 6.0
WATER_HEIGHT_KM = 2.0


@lru_cache(maxsize=512)
def _specific_db_km(f_ghz: float, pressure_hpa: float, rho: float, temp_k: float) -> tuple[float, float]:
    from itur.models import itu676  # imported here because the package loads slowly

    f = max(f_ghz, P676_MIN_GHZ)
    oxygen = itu676.gamma0_exact(f, pressure_hpa, rho, temp_k).value
    water = itu676.gammaw_exact(f, pressure_hpa, rho, temp_k).value
    return float(oxygen), float(water)


def gas_specific_attenuation_db_km(frequency_hz: float, env: EnvironmentConfig) -> float:
    oxygen, water = _specific_db_km(
        round(frequency_hz / 1e9, 4), env.pressure_hpa, env.water_vapour_g_m3, env.surface_temperature_k
    )
    return oxygen + water


def zenith_gas_attenuation_db(frequency_hz: float, env: EnvironmentConfig) -> float:
    oxygen, water = _specific_db_km(
        round(frequency_hz / 1e9, 4), env.pressure_hpa, env.water_vapour_g_m3, env.surface_temperature_k
    )
    return oxygen * OXYGEN_HEIGHT_KM + water * WATER_HEIGHT_KM
