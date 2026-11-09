"""Bi-static radar cross-section options.

constant:        the user's value is taken as the bi-static RCS at this geometry.
equivalence:     monostatic-bistatic equivalence: the bi-static RCS is about the monostatic
                 RCS on the bisector. Valid for smooth targets, up to moderate beta.
forward_scatter: adds the shadow lobe near beta = 180 deg, sigma_fs = 4 pi A^2 / lambda^2,
                 with main-lobe width about lambda / sqrt(A). Approximate, for simple shapes.
(Willis, Bistatic Radar, ch. 8; Griffiths and Baker, ch. 2.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import TargetConfig
from ..results import Contribution
from ..units import db, undb

EQUIVALENCE_VALID_BETA_DEG = 60.0
FORWARD_SCATTER_REGION_BETA_DEG = 140.0


@dataclass(frozen=True)
class BistaticRcs:
    value_dbsm: float
    contribution: Contribution


def bistatic_rcs(cfg: TargetConfig, beta_rad: float, wavelength_m: float) -> BistaticRcs:
    beta_deg = float(np.rad2deg(beta_rad))
    base = cfg.rcs_dbsm

    if cfg.rcs_model == "constant":
        note = ""
        if beta_deg > FORWARD_SCATTER_REGION_BETA_DEG:
            note = f"beta = {beta_deg:.0f} deg is close to forward scatter; consider the forward-scatter model."
        return BistaticRcs(base, Contribution(
            "Target RCS", base, "dBsm", True, "Entered directly by the user for this geometry.", "Input", note))

    if cfg.rcs_model == "equivalence":
        valid = beta_deg <= EQUIVALENCE_VALID_BETA_DEG
        note = "" if valid else f"beta = {beta_deg:.0f} deg is outside the {EQUIVALENCE_VALID_BETA_DEG:.0f} deg validity guide."
        return BistaticRcs(base, Contribution(
            "Target RCS", base, "dBsm", True,
            "Monostatic-bistatic equivalence: use the monostatic RCS at the bisector.", "Willis ch. 8", note))

    area = cfg.forward_scatter_area_m2
    peak = 4.0 * np.pi * area**2 / wavelength_m**2
    width = wavelength_m / np.sqrt(area)
    offset = np.pi - beta_rad
    lobe = peak * np.exp(-((offset / width) ** 2))
    total = float(db(undb(base) + lobe))
    return BistaticRcs(total, Contribution(
        "Target RCS", total, "dBsm", True,
        f"Input RCS plus a forward-scatter lobe: peak {db(peak):.1f} dBsm, width {np.rad2deg(width):.1f} deg, "
        f"{np.rad2deg(offset):.1f} deg away from exact forward scatter.",
        "Willis ch. 8", "Approximate; applies to compact shapes, and forward-scatter Doppler and range resolution are poor."))
