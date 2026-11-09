"""How much noise can vary: Rec. ITU-R P.372-18 Part 7, eq. 31-39, and Table 2.

External noise is not one number. P.372-18 gives each source a median and decile deviations,
modelled as two half-normal distributions in dB, and a way to combine the sources.

Two choices here go beyond the Recommendation's text:

* Part 7 is applied to the external random sources only (man-made and galactic). Ground,
  atmosphere and receiver noise are treated as constant and added afterwards. Moment-matching
  a sum that includes a large constant gives a poor result, and can even put the median below
  that constant, which is impossible.
* Eq. 31 does not say which deviations to use for the median. The upper-decile set is used,
  since this is what sets the design margin. The lower decile uses the lower-decile set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import T0_K

C_DB = 10.0 / np.log(10.0)  # eq. 33
DECILE_FACTOR = 1.282
GALACTIC_SIGMA_DB = 1.56  # 2 dB decile deviation / 1.282
RESTRICTION_DECILE_DB = 12.0

# Table 2: (upper with time, lower with time, upper with location, lower with location), in dB.
MAN_MADE_DECILES = {
    "city": (11.0, 6.7, 8.4, 8.4),
    "residential": (10.6, 5.3, 5.8, 5.8),
    "rural": (9.2, 4.6, 6.8, 6.8),
}


@dataclass(frozen=True)
class NoiseStatistics:
    """Noise temperatures at the array output. 'System' includes the constant sources."""

    external_median_k: float
    external_upper_k: float
    external_lower_k: float
    constant_k: float
    system_sum_of_medians_k: float
    upper_margin_db: float
    man_made_upper_deviation_db: float
    man_made_lower_deviation_db: float
    combined_upper_deviation_db: float
    combined_lower_deviation_db: float
    note: str

    @property
    def system_median_k(self) -> float:
        return self.external_median_k + self.constant_k

    @property
    def system_upper_k(self) -> float:
        return self.external_upper_k + self.constant_k

    @property
    def system_lower_k(self) -> float:
        return self.external_lower_k + self.constant_k


def man_made_decile_deviations_db(category: str, include_location: bool = True) -> tuple[float, float]:
    """Upper and lower deviations. Time and location are uncorrelated (Part 6), so they add in power."""
    if category not in MAN_MADE_DECILES:
        return 0.0, 0.0
    upper_time, lower_time, upper_location, lower_location = MAN_MADE_DECILES[category]
    if not include_location:
        return upper_time, lower_time
    return float(np.hypot(upper_time, upper_location)), float(np.hypot(lower_time, lower_location))


def combine_noise_figures(medians_db, sigmas_db) -> tuple[float, float]:
    """Median and standard deviation of a sum of noise processes: eq. 31, 32, 34, 35, 38, 39."""
    medians = np.asarray(medians_db, float)
    sigmas = np.asarray(sigmas_db, float)
    alpha = np.exp(medians / C_DB + sigmas**2 / (2.0 * C_DB**2))  # eq. 34
    alpha_total = float(alpha.sum())
    beta_total = float(np.sum(alpha**2 * (np.exp(sigmas**2 / C_DB**2) - 1.0)))  # eq. 35
    sigma_total = C_DB * np.sqrt(np.log(1.0 + beta_total / alpha_total**2))  # eq. 32
    if np.max(DECILE_FACTOR * sigmas) > RESTRICTION_DECILE_DB:
        gamma_total = float(np.sum(np.exp(medians / C_DB)))  # eq. 39
        sigma_total = min(sigma_total, C_DB * np.sqrt(2.0 * np.log(alpha_total / gamma_total)))  # eq. 38
    median = C_DB * (np.log(alpha_total) - sigma_total**2 / (2.0 * C_DB**2))  # eq. 31
    return float(median), float(sigma_total)


def _k_to_db(temperature_k) -> np.ndarray:
    return 10.0 * np.log10(np.asarray(temperature_k, float) / T0_K)


def _db_to_k(value_db: float) -> float:
    return float(T0_K * 10.0 ** (value_db / 10.0))


def noise_statistics(
    man_made_k: float, galactic_k: float, constant_k: float, upper_deviation_db: float, lower_deviation_db: float, note: str = ""
) -> NoiseStatistics:
    """Combine the random sources with Part 7, then add the constant ones."""
    temperatures, upper_sigmas, lower_sigmas = [], [], []
    if man_made_k > 0.0:
        temperatures.append(man_made_k)
        upper_sigmas.append(upper_deviation_db / DECILE_FACTOR)
        lower_sigmas.append(lower_deviation_db / DECILE_FACTOR)
    if galactic_k > 0.0:
        temperatures.append(galactic_k)
        upper_sigmas.append(GALACTIC_SIGMA_DB)
        lower_sigmas.append(GALACTIC_SIGMA_DB)

    if temperatures:
        medians_db = _k_to_db(temperatures)
        median_u, sigma_u = combine_noise_figures(medians_db, upper_sigmas)
        median_l, sigma_l = combine_noise_figures(medians_db, lower_sigmas)
        external_median = _db_to_k(median_u)
        external_upper = _db_to_k(median_u + DECILE_FACTOR * sigma_u)
        external_lower = min(_db_to_k(median_l - DECILE_FACTOR * sigma_l), external_median)
    else:
        external_median = external_upper = external_lower = 0.0
        sigma_u = sigma_l = 0.0

    sum_of_medians = man_made_k + galactic_k + constant_k
    return NoiseStatistics(
        external_median_k=external_median,
        external_upper_k=external_upper,
        external_lower_k=external_lower,
        constant_k=constant_k,
        system_sum_of_medians_k=sum_of_medians,
        upper_margin_db=float(10.0 * np.log10((external_upper + constant_k) / sum_of_medians)),
        man_made_upper_deviation_db=upper_deviation_db if man_made_k > 0.0 else 0.0,
        man_made_lower_deviation_db=lower_deviation_db if man_made_k > 0.0 else 0.0,
        combined_upper_deviation_db=float(DECILE_FACTOR * sigma_u),
        combined_lower_deviation_db=float(DECILE_FACTOR * sigma_l),
        note=note,
    )
