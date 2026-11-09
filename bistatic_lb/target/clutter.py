"""Ground clutter that shares the target's range cell.

A target at bi-static range Rb (extra path over the direct signal) competes with every
ground patch whose own extra path is the same, to within one range cell c/B. Those patches
lie on a ring around the transmitter and the receiver: an ellipse with the two sites as
foci and semi-major axis a = (L + Rb) / 2. The ring passes much closer to the two sites
than the ground under an airborne target does, so it is where the strong clutter is.

The ring is cut into patches. Each one returns

    dP = EIRP(toward it) G_rx(toward it) sigma0 dA lambda^2 / ((4 pi)^3 rt^2 rr^2 L_t L_r)

    sigma0 = gamma sqrt(sin(psi_t) sin(psi_r))     bi-static constant-gamma (Willis, ch. 9)
    dA     = (b cos^2(t) + (a^2 / b) sin^2(t)) da dt,   da = c / (2 B)

with psi the grazing angles, G_rx the beamformed gain (so sidelobes and the null count),
and L the propagation loss beyond free space on each leg, which shadows patches past the
radio horizon. The sum is the clutter at zero Doppler in the target's range cell, before
Doppler filtering and cancellation.

Limits: flat ground geometry with the smooth-Earth propagation loss, one reflectivity for
all ground, no discrete scatterers, no wind-driven spectral spread, and no range-ambiguous
clutter for pulsed waveforms.
"""

from __future__ import annotations

import numpy as np

from ..units import C_M_S

RING_PATCHES = 180
MIN_PATCH_RANGE_M = 10.0
LOSS_GRID_KM = np.geomspace(0.02, 600.0, 72)


class LegLoss:
    """Propagation loss beyond free space against ground distance, tabulated once for one leg."""

    def __init__(self, excess_loss_db_at_km):
        self._log_km = np.log(LOSS_GRID_KM)
        self._loss_db = np.array([excess_loss_db_at_km(float(d)) for d in LOSS_GRID_KM])

    def __call__(self, distance_m: np.ndarray) -> np.ndarray:
        km = np.clip(np.asarray(distance_m) / 1e3, LOSS_GRID_KM[0], LOSS_GRID_KM[-1])
        return np.interp(np.log(km), self._log_km, self._loss_db)


def ring_patches(baseline_m: float, bistatic_range_m: float, bandwidth_hz: float, n: int = RING_PATCHES):
    """Patch centres (x, y) and areas on the ground ring, with the receiver at the origin and the transmitter on +x."""
    a = (baseline_m + max(bistatic_range_m, 0.0)) / 2.0
    c = baseline_m / 2.0
    da = C_M_S / (2.0 * bandwidth_hz)
    # Closer to the baseline than one cell, the ring collapses onto it; keep one cell of width.
    b = np.sqrt(max(a**2 - c**2, (a + da) ** 2 - a**2))
    t = (np.arange(n) + 0.5) * 2.0 * np.pi / n
    x, y = c + a * np.cos(t), b * np.sin(t)
    area = (b * np.cos(t) ** 2 + a**2 / b * np.sin(t) ** 2) * da * (2.0 * np.pi / n)
    return x, y, area


def ring_clutter_dbm(
    baseline_m: float,
    bistatic_range_m: float,
    bandwidth_hz: float,
    wavelength_m: float,
    tx_height_m: float,
    rx_height_m: float,
    gamma_db: float,
    eirp_w_toward,
    rx_gain_toward,
    tx_leg_loss: LegLoss,
    rx_leg_loss: LegLoss,
) -> float:
    """Clutter power (dBm) at zero Doppler in the range cell at this bi-static range.

    eirp_w_toward(grazing_rad) gives the EIRP in watts toward a ground patch seen at that
    depression angle; rx_gain_toward(az_rad, el_rad) gives the beamformed linear gain.
    """
    x, y, area = ring_patches(baseline_m, bistatic_range_m, bandwidth_hz)
    ground_rx = np.maximum(np.hypot(x, y), MIN_PATCH_RANGE_M)
    ground_tx = np.maximum(np.hypot(x - baseline_m, y), MIN_PATCH_RANGE_M)
    psi_t, psi_r = np.arctan2(tx_height_m, ground_tx), np.arctan2(rx_height_m, ground_rx)
    rt, rr = np.hypot(ground_tx, tx_height_m), np.hypot(ground_rx, rx_height_m)
    sigma0 = 10.0 ** (gamma_db / 10.0) * np.sqrt(np.sin(psi_t) * np.sin(psi_r))
    loss = 10.0 ** (-(tx_leg_loss(ground_tx) + rx_leg_loss(ground_rx)) / 10.0)
    power_w = (
        eirp_w_toward(psi_t) * rx_gain_toward(np.arctan2(y, x), -psi_r) * sigma0 * area * wavelength_m**2
        / ((4.0 * np.pi) ** 3 * rt**2 * rr**2) * loss
    )
    return float(10.0 * np.log10(max(np.sum(power_w), 1e-33)) + 30.0)
