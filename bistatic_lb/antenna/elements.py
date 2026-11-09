"""Antenna element patterns.

Half-wave dipole: F(theta) = cos((pi/2) cos(theta)) / sin(theta), theta measured from
the dipole axis (Balanis, Antenna Theory, ch. 4). A folded dipole has the same
pattern; it differs in feed impedance (about 4x) and bandwidth (lower Q).

Quarter-wave monopole over an ideal ground plane: the dipole pattern in the upper
hemisphere with twice the gain (about 5.15 dBi).

Off-tune behaviour uses a single-resonance impedance model. It is a planning
approximation, not an EM result, and it does not change the pattern shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Scenario

ELEMENT_Q = {"dipole": 8.0, "folded_dipole": 4.0, "monopole": 8.0}

# 2 / integral(F^2 sin(theta) dtheta) over 0..pi for the half-wave dipole.
_THETA = np.linspace(1e-6, np.pi - 1e-6, 20001)
_F2 = (np.cos(np.pi / 2 * np.cos(_THETA)) / np.sin(_THETA)) ** 2
DIPOLE_DIRECTIVITY = float(2.0 / np.trapezoid(_F2 * np.sin(_THETA), _THETA))


def dipole_gain(cos_theta):
    """Linear power gain of a half-wave dipole, given cos(angle from its axis)."""
    c = np.clip(cos_theta, -1.0, 1.0)
    s2 = np.maximum(1.0 - c**2, 1e-12)
    return DIPOLE_DIRECTIVITY * np.cos(np.pi / 2.0 * c) ** 2 / s2


def mismatch_loss_db(frequency_hz: float, design_hz: float, element: str) -> float:
    """Reflection loss away from resonance, for a series RLC element matched at design_hz."""
    q = ELEMENT_Q[element]
    x = q * (frequency_hz / design_hz - design_hz / frequency_hz)
    gamma2 = x**2 / (4.0 + x**2)  # |Gamma|^2 for Z = R(1 + jx)
    return float(-10.0 * np.log10(1.0 - gamma2))


def usable_band_hz(design_hz: float, element: str, vswr: float = 2.0) -> tuple[float, float]:
    """Frequencies between which VSWR stays below the limit."""
    gamma = (vswr - 1.0) / (vswr + 1.0)
    x = 2.0 * gamma / np.sqrt(1.0 - gamma**2)
    delta = x / ELEMENT_Q[element]  # f/f0 - f0/f
    upper = (delta + np.sqrt(delta**2 + 4.0)) / 2.0
    return design_hz / upper, design_hz * upper


@dataclass(frozen=True)
class MonopoleTx:
    """Vertical monopole: omnidirectional in azimuth, vertically polarized.

    Elevation angles below the horizon use the same gain as the matching angle above
    it. Rays toward a lower receiver leave at a small negative angle, and the ideal
    ground-plane image applies near the horizon.
    """

    finite_ground: bool = False
    tilt_rad: float = np.deg2rad(30.0)
    horizon_loss_db: float = 6.0

    def gain_linear(self, elevation_rad: float) -> float:
        el = abs(elevation_rad)
        gain = 2.0 * float(dipole_gain(np.sin(el)))
        if self.finite_ground:
            # Empirical: full loss at the horizon, falling linearly in dB to none at the lobe peak.
            fraction = max(0.0, 1.0 - el / self.tilt_rad)
            gain *= 10.0 ** (-self.horizon_loss_db * fraction / 10.0)
        return gain

    def gain_dbi(self, elevation_rad: float) -> float:
        return float(10.0 * np.log10(max(self.gain_linear(elevation_rad), 1e-12)))


def monopole_from_scenario(s: Scenario) -> MonopoleTx:
    t = s.transmitter
    return MonopoleTx(
        finite_ground=t.ground_plane == "finite",
        tilt_rad=np.deg2rad(t.finite_ground_tilt_deg),
        horizon_loss_db=t.finite_ground_horizon_loss_db,
    )
