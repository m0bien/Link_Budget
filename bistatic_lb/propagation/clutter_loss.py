"""Terminal clutter loss, Rec. ITU-R P.2108-1 sec. 3.2 (0.5 to 67 GHz, urban and suburban)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

P2108_MIN_GHZ = 0.5
P2108_MAX_GHZ = 67.0


def _median_and_spread(f_ghz: float, d_km: float, location_percent: float) -> float:
    ll = -2.0 * np.log10(10.0 ** (-5.0 * np.log10(f_ghz) - 12.5) + 10.0**-16.5)  # eq. 4a
    ls = 32.98 + 23.9 * np.log10(d_km) + 3.0 * np.log10(f_ghz)  # eq. 5a
    a, b = 10.0 ** (-0.2 * ll), 10.0 ** (-0.2 * ls)
    sigma = np.sqrt((16.0 * a + 36.0 * b) / (a + b))  # eq. 3b with sigma_l = 4, sigma_s = 6
    return float(-5.0 * np.log10(a + b) - sigma * norm.isf(location_percent / 100.0))  # eq. 3a


def terrestrial_clutter_loss_db(f_ghz: float, d_km: float, location_percent: float = 50.0) -> float:
    """Loss at one end of the path, limited to its value at 2 km (eq. 6)."""
    return min(
        _median_and_spread(f_ghz, d_km, location_percent),
        _median_and_spread(f_ghz, 2.0, location_percent),
    )
