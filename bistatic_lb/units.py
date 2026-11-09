"""Physical constants and decibel helpers used throughout the package."""

from __future__ import annotations

import numpy as np

C_M_S = 299_792_458.0  # speed of light
K_BOLTZMANN = 1.380649e-23  # J/K
T0_K = 290.0  # IEEE reference noise temperature
T_CMB_K = 2.73  # cosmic background (ITU-R P.372-18 eq. 10)


def db(x):
    """Power ratio to decibels. Zero maps to a very low level, not an error."""
    return 10.0 * np.log10(np.maximum(x, 1e-300))


def undb(x_db):
    """Decibels to power ratio."""
    return 10.0 ** (np.asarray(x_db, dtype=float) / 10.0)


def wavelength_m(frequency_hz: float) -> float:
    return C_M_S / frequency_hz


def dbm_to_w(p_dbm):
    return undb(p_dbm) * 1e-3


def w_to_dbm(p_w):
    return db(p_w) + 30.0


def power_sum_db(*levels_db: float) -> float:
    """Add powers expressed in dB (non-coherent sum)."""
    return float(db(sum(undb(level) for level in levels_db)))
