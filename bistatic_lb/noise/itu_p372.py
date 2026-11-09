"""External noise factors from Rec. ITU-R P.372-18 (09/2026).

    eq (17), Table 1, Fig. 39   Fam = c - d log10(f_MHz). City, residential and rural run
                                from 0.3 to 250 MHz. Quiet rural (curve D) stops at 30 MHz:
                                it is ionospherically reflected noise, seen only below fxF2.
    eq (15), Fig. 3 curve B     galactic Fam = 52 - 23 log10(f_MHz). The equation is stated up
                                to about 100 MHz; curve B continues to about 5 GHz with close
                                to the same slope (it reads within about 3 dB).
    Fig. 3 curve A              estimated city man-made noise to about 900 MHz. It flattens
                                to about -11.5 dB per decade above 175 MHz.
    Fig. 3 curve D              quiet Sun.
    eq (10), eq (11)            Tb = 2.73 x 10^(-A/10) + Tmr (1 - 10^(-A/10)). Tmr = 275 K for
                                rain. For clear air, Tmr comes from Tmr_approx.txt, a separate
                                ITU file that is not bundled, so 275 K is used there too.

The factors refer to a short vertical lossless grounded monopole (Fig. 39 title).
Values marked "Fig." are read from the published curve, to about +/-1 dB.
"""

from __future__ import annotations

import numpy as np

from ..units import T0_K

MAN_MADE_COEFFICIENTS = {
    "city": (76.8, 27.7),
    "residential": (72.5, 27.7),
    "rural": (67.2, 27.7),
    "quiet_rural": (53.6, 28.6),
}
MAN_MADE_MAX_VALID_HZ = 250e6
QUIET_RURAL_MAX_VALID_HZ = 30e6
GALACTIC_STATED_MAX_HZ = 100e6
ABOVE_LIMIT_SLOPE_DB_PER_DECADE = 11.5  # Fig. 3 curve A: 13 dB at 300 MHz to 7.5 dB at 900 MHz
MEAN_RADIATING_TEMPERATURE_K = 275.0
REFERENCE_MONOPOLE_GAIN_DBI = 4.77  # short monopole over a perfect ground plane: directivity 3

# Fig. 3 curve D, read from the figure: (frequency in GHz, Fa in dB).
_QUIET_SUN_FA_DB = (
    (0.1, 34.0), (0.2, 32.0), (0.5, 29.0), (1.0, 27.0), (2.0, 23.5),
    (5.0, 19.0), (10.0, 15.0), (20.0, 13.0),
)


def man_made_fam_db(frequency_hz, category: str):
    """Median man-made noise figure: eq. 17, then the flatter Fig. 3 slope above 250 MHz.

    Above 250 MHz is an estimate. Curve A is for city noise only, and stops near 900 MHz.
    Using its slope for other categories and at higher frequencies is an assumption of
    this tool, though it matches outdoor measurements at 322-425 MHz much better than
    continuing eq. 17 (P.372-13, Tables 3 and 4).
    """
    c, d = MAN_MADE_COEFFICIENTS[category]
    f = np.asarray(frequency_hz, float)
    limit = MAN_MADE_MAX_VALID_HZ
    eq17 = c - d * np.log10(np.minimum(f, limit) / 1e6)
    return eq17 - ABOVE_LIMIT_SLOPE_DB_PER_DECADE * np.log10(np.maximum(f, limit) / limit)


def man_made_applies(frequency_hz: float, category: str) -> bool:
    """False where the category has no curve: quiet rural stops at 30 MHz (Fig. 39)."""
    return not (category == "quiet_rural" and frequency_hz > QUIET_RURAL_MAX_VALID_HZ)


def galactic_fam_db(frequency_hz):
    return 52.0 - 23.0 * np.log10(np.asarray(frequency_hz, float) / 1e6)


def fa_to_temperature_k(fa_db):
    return T0_K * 10.0 ** (np.asarray(fa_db, float) / 10.0)


def brightness_through_absorption_k(attenuation_db, background_k, medium_k=MEAN_RADIATING_TEMPERATURE_K):
    """Emission of an absorbing medium plus the attenuated background behind it."""
    transmission = 10.0 ** (-np.asarray(attenuation_db, float) / 10.0)
    return medium_k * (1.0 - transmission) + background_k * transmission


def quiet_sun_brightness_k(frequency_hz: float) -> float:
    """Quiet-Sun temperature from Fig. 3 curve D, held flat outside its plotted range here."""
    points = np.array(_QUIET_SUN_FA_DB)
    fa = np.interp(np.log10(frequency_hz / 1e9), np.log10(points[:, 0]), points[:, 1])
    return float(fa_to_temperature_k(fa))
