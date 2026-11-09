"""Loss when the target moves through range or Doppler cells during one coherent interval.

Coherent integration assumes the echo stays in one range-Doppler cell. A moving target
breaks that in two ways, and each has a closed form for an unweighted, flat-spectrum
waveform when the detector reads the cell at the middle of the walk.

Range walk. The bi-static path changes by |range rate| x T. With a range response sinc(x),
x in cells of c/B, a walk of M cells leaves the amplitude

    A = (1/M) * integral of sinc(x) over -M/2..M/2 = 2 Si(pi M / 2) / (pi M)

which tends to 1/M: only the time spent in the cell integrates coherently, while noise
integrates over the whole interval.

Doppler walk. The Doppler changes by (range acceleration / wavelength) x T. A walk of N
Doppler cells (each 1/T) is a quadratic phase pi N u^2 over the interval, leaving

    A = | integral of exp(j pi N u^2) over -1/2..1/2 | = |C(x) + j S(x)| / x,  x = sqrt(N/2)

with C and S the Fresnel integrals.

The range acceleration here is the geometric one for a target flying straight at constant
speed. A manoeuvring target adds up to twice its own acceleration to it, which is not
included. Processing that follows the target (keystone or stretch processing, Doppler-rate
search) removes these losses; ``compensation`` says which of them the processor removes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import fresnel, sici

from ..units import C_M_S


@dataclass(frozen=True)
class Migration:
    range_cells: float  # bi-static range cells crossed in one coherent interval
    doppler_cells: float  # Doppler cells crossed in one coherent interval
    range_loss_db: float  # after any compensation
    doppler_loss_db: float

    @property
    def loss_db(self) -> float:
        return self.range_loss_db + self.doppler_loss_db


def range_walk_loss_db(cells: float) -> float:
    """SNR loss for an echo that crosses this many range cells during the interval."""
    m = abs(cells)
    if m < 1e-6:
        return 0.0
    amplitude = 2.0 * sici(np.pi * m / 2.0)[0] / (np.pi * m)
    return float(-20.0 * np.log10(amplitude))


def doppler_walk_loss_db(cells: float) -> float:
    """SNR loss for an echo whose Doppler crosses this many Doppler cells during the interval."""
    n = abs(cells)
    if n < 1e-6:
        return 0.0
    x = np.sqrt(n / 2.0)
    s, c = fresnel(x)
    return float(-20.0 * np.log10(np.hypot(c, s) / x))


def cells_for_loss(loss, loss_db: float = 1.0) -> float:
    """Cells crossed at which ``loss`` (one of the two functions above) reaches loss_db."""
    return float(brentq(lambda cells: loss(cells) - loss_db, 1e-3, 50.0))


def migration(
    range_rate_m_s: float,
    range_accel_m_s2: float,
    cpi_s: float,
    bandwidth_hz: float,
    wavelength_m: float,
    compensation: str = "none",
) -> Migration:
    """Cells crossed, and the loss left after the stated compensation ('none', 'range' or 'range_doppler')."""
    range_cells = abs(range_rate_m_s) * cpi_s / (C_M_S / bandwidth_hz)
    doppler_cells = abs(range_accel_m_s2) / wavelength_m * cpi_s**2
    return Migration(
        range_cells=float(range_cells),
        doppler_cells=float(doppler_cells),
        range_loss_db=range_walk_loss_db(range_cells) if compensation == "none" else 0.0,
        doppler_loss_db=doppler_walk_loss_db(doppler_cells) if compensation != "range_doppler" else 0.0,
    )


def longest_cpi_s(range_rate_m_s: float, range_accel_m_s2: float, bandwidth_hz: float, wavelength_m: float, compensation: str = "none", loss_db: float = 1.0) -> float:
    """Longest coherent interval that keeps each uncompensated migration loss under loss_db."""
    limit = np.inf
    if compensation == "none" and abs(range_rate_m_s) > 0.0:
        limit = min(limit, cells_for_loss(range_walk_loss_db, loss_db) * (C_M_S / bandwidth_hz) / abs(range_rate_m_s))
    if compensation != "range_doppler" and abs(range_accel_m_s2) > 0.0:
        limit = min(limit, np.sqrt(cells_for_loss(doppler_walk_loss_db, loss_db) * wavelength_m / abs(range_accel_m_s2)))
    return float(limit)
