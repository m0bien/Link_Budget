"""Continuous noise-like codes: random phase coding and Zadoff-Chu.

Random phase code: a long random code's range-Doppler floor sits near 1/(BT) of the
peak at every delay and Doppler. So the direct path's floor can mask weak targets
unless it is cancelled.

Zadoff-Chu: constant amplitude and an ideally zero periodic autocorrelation, so a
stationary direct path leaves no range sidelobes. The price is range ambiguity at the
code period, and a Doppler mismatch loss if the code period is too long for the target.
``cfg.pulse_width_s`` sets the code period.
"""

from __future__ import annotations

import numpy as np

from ..units import C_M_S
from .base import HARDWARE_FLOOR_DB, Waveform

PHASE_CODE_FILTERED_PAPR_DB = 3.0  # practical band-limited transmitter


class PhaseCoded(Waveform):
    name = "Phase-coded (noise-like)"

    @property
    def bandwidth_hz(self) -> float:
        return self.cfg.bandwidth_hz

    @property
    def papr_db(self) -> float:
        return PHASE_CODE_FILTERED_PAPR_DB

    @property
    def duty_cycle(self) -> float:
        return 1.0

    def direct_leakage_db(self, delay_s: float, doppler_hz: float) -> float:
        return float(max(-10.0 * np.log10(self.time_bandwidth), HARDWARE_FLOOR_DB))

    def notes(self) -> list[str]:
        return [f"Direct-path floor near {-10.0 * np.log10(self.time_bandwidth):.0f} dB of its own peak in every cell."]

    def generate(self, fs_hz: float, duration_s: float, rng: np.random.Generator) -> np.ndarray:
        chips = rng.choice([-1.0, 1.0], size=int(duration_s * self.cfg.bandwidth_hz) + 1)
        index = (np.arange(int(duration_s * fs_hz)) * self.cfg.bandwidth_hz / fs_hz).astype(int)
        return chips[index].astype(complex)


class ZadoffChu(Waveform):
    name = "Zadoff-Chu"

    @property
    def bandwidth_hz(self) -> float:
        return self.cfg.bandwidth_hz

    @property
    def papr_db(self) -> float:
        return 0.0

    @property
    def duty_cycle(self) -> float:
        return 1.0

    @property
    def period_s(self) -> float:
        return self.cfg.pulse_width_s

    @property
    def ambiguous_doppler_hz(self) -> float:
        return 1.0 / self.period_s

    @property
    def ambiguous_range_m(self) -> float:
        return C_M_S * self.period_s

    def direct_leakage_db(self, delay_s: float, doppler_hz: float) -> float:
        # Range sidelobes vanish for a stationary signal; only Doppler rejection remains.
        return float(max(self.doppler_rejection_db(doppler_hz), HARDWARE_FLOOR_DB))

    def signal_loss_db(self, delay_s: float, doppler_hz: float, blind_extra_s: float = 0.0) -> float:
        """Correlating a Doppler-shifted code loses energy as sinc^2(fd * period)."""
        value = np.sinc(doppler_hz * self.period_s)
        return float(-20.0 * np.log10(max(abs(value), 1e-3)))

    def notes(self) -> list[str]:
        return [
            f"Zero periodic range sidelobes, but range repeats every {self.ambiguous_range_m / 1e3:.0f} km of path.",
            "The hardware floor, not the code, limits direct-path rejection in practice.",
        ]

    def generate(self, fs_hz: float, duration_s: float, rng: np.random.Generator) -> np.ndarray:
        n_code = int(round(self.period_s * self.cfg.bandwidth_hz)) | 1  # odd length
        k = np.arange(n_code)
        code = np.exp(-1j * np.pi * k * (k + 1) / n_code)
        index = (np.arange(int(duration_s * fs_hz)) * self.cfg.bandwidth_hz / fs_hz).astype(int) % n_code
        return code[index]
