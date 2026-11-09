"""Receiver line-up: cascaded noise figure, gain and third-order intercept (Friis)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import FrontEndConfig
from ..units import T0_K, db, undb


@dataclass(frozen=True)
class Stage:
    name: str
    gain_db: float
    nf_db: float
    iip3_dbm: float = np.inf


@dataclass(frozen=True)
class Lineup:
    stages: tuple[Stage, ...]

    @property
    def gain_db(self) -> float:
        return sum(s.gain_db for s in self.stages)

    @property
    def noise_figure_db(self) -> float:
        total, gain = 0.0, 1.0
        for i, s in enumerate(self.stages):
            f = undb(s.nf_db)
            total += f if i == 0 else (f - 1.0) / gain
            gain *= undb(s.gain_db)
        return float(db(total))

    @property
    def noise_temperature_k(self) -> float:
        return float(T0_K * (undb(self.noise_figure_db) - 1.0))

    @property
    def iip3_dbm(self) -> float:
        inverse, gain = 0.0, 1.0
        for s in self.stages:
            if np.isfinite(s.iip3_dbm):
                inverse += gain / undb(s.iip3_dbm)
            gain *= undb(s.gain_db)
        return float(db(1.0 / inverse)) if inverse > 0 else np.inf

    def gain_before_db(self, stage_name: str) -> float:
        total = 0.0
        for s in self.stages:
            if s.name == stage_name:
                return total
            total += s.gain_db
        raise KeyError(stage_name)

    def table(self) -> list[dict]:
        rows, running_gain, running = [], 0.0, []
        for s in self.stages:
            running.append(s)
            running_gain += s.gain_db
            rows.append(
                {
                    "Stage": s.name,
                    "Gain (dB)": round(s.gain_db, 2),
                    "NF (dB)": round(s.nf_db, 2),
                    "IIP3 (dBm)": "-" if not np.isfinite(s.iip3_dbm) else f"{s.iip3_dbm:.1f}",
                    "Cumulative gain (dB)": round(running_gain, 2),
                    "Cumulative NF (dB)": round(Lineup(tuple(running)).noise_figure_db, 2),
                }
            )
        return rows


def build_lineup(f: FrontEndConfig) -> Lineup:
    """Passive stages have NF equal to their loss, at the 290 K reference."""
    return Lineup(
        (
            Stage("Feed", -f.feed_loss_db, f.feed_loss_db),
            Stage("Limiter", -f.limiter_loss_db, f.limiter_loss_db),
            Stage("RF filter", -f.rf_filter_loss_db, f.rf_filter_loss_db),
            Stage("LNA", f.lna_gain_db, f.lna_nf_db, f.lna_iip3_dbm),
            Stage("Mixer", -f.mixer_loss_db, f.mixer_loss_db, f.mixer_iip3_dbm),
            Stage("IF amplifier", f.if_gain_db, f.if_nf_db, f.if_iip3_dbm),
        )
    )
