"""Ground reflection (two-ray over a spherical Earth) and clear-air fading (P.530-19)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import wavelength_m
from .diffraction import reflection_point_distances_m
from .surface import SurfaceConstants


@dataclass(frozen=True)
class TwoRayResult:
    grazing_deg: float
    reflection_magnitude: float  # includes divergence and roughness
    nominal_db: float  # with the actual phase at this geometry; can be up or down
    mean_db: float  # power average over phase, for a target that moves through the lobes
    maximum_db: float
    minimum_db: float


def fresnel_reflection(grazing_rad: float, surface: SurfaceConstants, frequency_hz: float, vertical: bool) -> complex:
    """Plane-wave reflection coefficient of a smooth ground with complex permittivity."""
    eps = surface.permittivity - 1j * 60.0 * wavelength_m(frequency_hz) * surface.conductivity_s_m
    s = np.sin(grazing_rad)
    root = np.sqrt(eps - np.cos(grazing_rad) ** 2)
    if vertical:
        return (eps * s - root) / (eps * s + root)
    return (s - root) / (s + root)


def two_ray(
    d_km: float, h1_m: float, h2_m: float, frequency_hz: float, a_km: float, surface: SurfaceConstants,
    roughness_rms_m: float, vertical: bool = True,
) -> TwoRayResult:
    """Direct ray plus one specular ground ray, using effective heights at the reflection point."""
    lam = wavelength_m(frequency_hz)
    d, a = d_km * 1e3, a_km * 1e3
    d1, d2 = reflection_point_distances_m(d, h1_m, h2_m, a)
    h1e, h2e = h1_m - d1**2 / (2.0 * a), h2_m - d2**2 / (2.0 * a)
    if h1e <= 0.0 or h2e <= 0.0:
        return TwoRayResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    grazing = np.arctan((h1e + h2e) / d)
    divergence = (1.0 + 2.0 * d1 * d2 / (a * d * np.tan(grazing))) ** -0.5
    roughness = np.exp(-2.0 * (2.0 * np.pi * roughness_rms_m * np.sin(grazing) / lam) ** 2)  # Ament
    rho = fresnel_reflection(grazing, surface, frequency_hz, vertical) * divergence * roughness
    magnitude = float(abs(rho))

    path_difference = 2.0 * h1e * h2e / d
    total = 1.0 + rho * np.exp(-2j * np.pi * path_difference / lam)
    floor = 1e-3
    return TwoRayResult(
        grazing_deg=float(np.rad2deg(grazing)),
        reflection_magnitude=magnitude,
        nominal_db=float(20.0 * np.log10(max(abs(total), floor))),
        mean_db=float(10.0 * np.log10(1.0 + magnitude**2)),
        maximum_db=float(20.0 * np.log10(1.0 + magnitude)),
        minimum_db=float(20.0 * np.log10(max(1.0 - magnitude, floor))),
    )


def clear_air_fade_depth_db(
    d_km: float, f_ghz: float, h1_m: float, h2_m: float, log10_k: float, dn75: float, time_percent: float,
    mean_terrain_m: float = 0.0,
) -> float:
    """Fade depth A exceeded for time_percent of the worst month: P.530-19 eq. 5-9 solved for A."""
    he, hr = max(h1_m, h2_m), min(h1_m, h2_m)
    epsilon = abs(hr - he) / d_km  # eq. 5, mrad
    hc = (hr + he) / 2.0 - d_km**2 / 102.0 - mean_terrain_m  # eq. 6
    limit = dn75 * d_km**1.5 * f_ghz**0.5 / 24730.0  # eq. 9
    v_sr = min((dn75 / 50.0) ** 1.8 * np.exp(-hc / (2.5 * np.sqrt(d_km))), limit)  # eq. 8
    h_l = hr
    log_pw_at_0db = (
        log10_k + 3.51 * np.log10(d_km) + 0.447 * np.log10(f_ghz**2 + 13.0)
        - 0.376 * np.tanh((hc - 147.0) / 125.0) - 0.334 * epsilon**0.39 - 0.00027 * h_l + 17.85 * v_sr
    )  # eq. 7
    return float(max(10.0 * (log_pw_at_0db - np.log10(time_percent)), 0.0))


def clear_air_fade_applicability(d_km: float, f_ghz: float, line_of_sight: bool, h1_m: float, h2_m: float) -> tuple[bool, str]:
    """P.530-19 sec. 2.3.1 limits: over 5 km, f >= 15/d GHz, grazing LoS, inclination to 37 mrad."""
    if not line_of_sight:
        return False, "Rejected: the P.530 fit requires at least a grazing line of sight."
    if d_km <= 5.0:
        return False, f"Rejected: P.530 sets clear-air fading to zero for paths of 5 km or less ({d_km:.1f} km here)."
    inclination = abs(h1_m - h2_m) / d_km
    if inclination > 37.0:
        return False, f"Rejected: path inclination {inclination:.0f} mrad is beyond the 37 mrad range of the P.530 fit (an elevated path)."
    f_min = 15.0 / d_km
    if f_ghz < f_min:
        return False, f"Rejected: {f_ghz:.2f} GHz is below the P.530 lower limit of about {f_min:.2f} GHz for {d_km:.0f} km."
    return True, f"Line of sight, over 5 km, and above the P.530 lower limit of {f_min:.2f} GHz for this path."
