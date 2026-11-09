"""Detection probability for a square-law detector with non-coherent integration.

Noise-normalized statistic: Y = sum over n looks of |s_i + w_i|^2, with E|w|^2 = 1. Under
noise alone Y is Gamma(n, 1), so the threshold is exact for any Pfa. Target models differ
only in how the per-look SNR x fluctuates:

    Swerling 0   steady
    Swerling 1   chi-square m = 1, constant over the n looks (scan-to-scan)
    Swerling 2   chi-square m = 1, independent per look (pulse-to-pulse)
    Swerling 3   chi-square m = 2, constant over the looks
    Swerling 4   chi-square m = 2, independent per look
    chi2_scan    chi-square with any m (Weinstock), constant over the looks
    rician       dominant scatterer plus diffuse part, constant over the looks

Add a model by writing a new entry for FLUCTUATION_MODELS in ``make_fluctuation``.
Albersheim and Shnidman closed forms are here as cross-checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaincc, gammainccinv
from scipy.stats import gamma as gamma_dist
from scipy.stats import ncx2

from ..config import TargetConfig
from ..units import undb

_QUANTILES = (np.arange(400) + 0.5) / 400.0


@dataclass(frozen=True)
class Fluctuation:
    """How the target's SNR fluctuates."""

    name: str
    kind: str  # 'steady', 'scan', 'pulse' or 'rician'
    m: float = 1.0
    rician_k: float = 0.0


FLUCTUATION_MODELS = {
    "swerling0": Fluctuation("Swerling 0 (steady)", "steady"),
    "swerling1": Fluctuation("Swerling 1", "scan", 1.0),
    "swerling2": Fluctuation("Swerling 2", "pulse", 1.0),
    "swerling3": Fluctuation("Swerling 3", "scan", 2.0),
    "swerling4": Fluctuation("Swerling 4", "pulse", 2.0),
}


def make_fluctuation(cfg: TargetConfig) -> Fluctuation:
    if cfg.fluctuation == "chi2_scan":
        return Fluctuation(f"Chi-square, m = {cfg.chi2_dof_m:g}", "scan", cfg.chi2_dof_m)
    if cfg.fluctuation == "rician":
        return Fluctuation(f"Rician, K = {cfg.rician_k_db:g} dB", "rician", rician_k=float(undb(cfg.rician_k_db)))
    return FLUCTUATION_MODELS[cfg.fluctuation]


def threshold(pfa: float, n: int = 1) -> float:
    """Exact threshold on Y for this false-alarm probability."""
    return float(gammainccinv(n, pfa))


def _pd_given_x(x_per_look, t: float, n: int):
    """Pd for a steady SNR x per look: Marcum Q via the non-central chi-square."""
    return ncx2.sf(2.0 * t, 2 * n, 2.0 * n * np.asarray(x_per_look))


def detection_probability(snr_linear: float, pfa: float, n: int, model: Fluctuation) -> float:
    """Pd for mean SNR per look. Exact to the numerical quadrature used."""
    t = threshold(pfa, n)
    if model.kind == "steady":
        return float(_pd_given_x(snr_linear, t, n))
    if model.kind == "pulse":
        if model.m == 1.0:
            return float(gammaincc(n, t / (1.0 + snr_linear)))
        # Y depends on the looks only through their total power, which is Gamma(n m, S / m).
        total = gamma_dist.ppf(_QUANTILES, n * model.m, scale=snr_linear / model.m)
        return float(np.mean(_pd_given_x(total / n, t, n)))
    if model.kind == "scan":
        if model.m == 1.0 and n == 1:
            return float(pfa ** (1.0 / (1.0 + snr_linear)))
        x = gamma_dist.ppf(_QUANTILES, model.m, scale=snr_linear / model.m)
    else:  # Rician power: a non-central chi-square with 2 degrees of freedom
        scale = snr_linear / (1.0 + model.rician_k) / 2.0
        x = ncx2.ppf(_QUANTILES, 2, 2.0 * model.rician_k) * scale
    return float(np.mean(_pd_given_x(x, t, n)))


def pd_curve(snr_db: np.ndarray, pfa: float, n: int, model: Fluctuation) -> np.ndarray:
    return np.array([detection_probability(float(undb(v)), pfa, n, model) for v in np.atleast_1d(snr_db)])


def required_snr_db(pd: float, pfa: float, n: int, model: Fluctuation) -> float:
    """Mean SNR per look needed to reach pd."""
    return float(brentq(lambda v: detection_probability(float(undb(v)), pfa, n, model) - pd, -40.0, 80.0, xtol=1e-3))


def albersheim_snr_db(pd: float, pfa: float, n: int = 1) -> float:
    """Albersheim (1981), steady target, for 0.1 <= Pd <= 0.9 and 1e-7 <= Pfa <= 1e-3."""
    a = np.log(0.62 / pfa)
    b = np.log(pd / (1.0 - pd))
    term = a + 0.12 * a * b + 1.7 * b
    return float(-5.0 * np.log10(n) + (6.2 + 4.54 / np.sqrt(n + 0.44)) * np.log10(term))


def shnidman_snr_db(pd: float, pfa: float, n: int, swerling: int) -> float:
    """Shnidman (2002), for Swerling 0-4; about 1 dB accuracy over 0.1 <= Pd <= 0.99."""
    k = {0: np.inf, 1: 1.0, 2: float(n), 3: 2.0, 4: 2.0 * n}[swerling]
    alpha = 0.25 if n >= 40 else 0.0
    eta = np.sqrt(-0.8 * np.log(4.0 * pfa * (1.0 - pfa))) + np.sign(pd - 0.5) * np.sqrt(-0.8 * np.log(4.0 * pd * (1.0 - pd)))
    x_inf = eta * (eta + 2.0 * np.sqrt(n / 2.0 + (alpha - 0.25)))
    c1 = (((17.7006 * pd - 18.4496) * pd + 14.5339) * pd - 3.525) / k
    c2 = (np.exp(27.31 * pd - 25.14) + (pd - 0.8) * (0.7 * np.log(1e-5 / pfa) + (2.0 * n - 20.0) / 80.0)) / k
    c_db = c1 if pd <= 0.872 else c1 + c2
    return float(10.0 * np.log10(x_inf / n) + c_db)


def cfar_loss_db(pd: float, pfa: float, reference_cells: int) -> float:
    """CA-CFAR loss from the exact Swerling 1, single-look result (Gandhi and Kassam)."""
    if reference_cells <= 0:
        return 0.0
    n = reference_cells
    with_cfar = (pfa ** (-1.0 / n) - 1.0) / (pd ** (-1.0 / n) - 1.0)
    fixed = np.log(pfa) / np.log(pd)
    return float(10.0 * np.log10(with_cfar / fixed))
