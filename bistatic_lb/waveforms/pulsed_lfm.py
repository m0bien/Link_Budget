"""Pulsed linear FM: constant envelope, with the direct path separated in time.

A receiver that hears the transmitter pulse directly is effectively blanked while that
pulse lasts, unless the front-end can keep it linear. A target whose delay (modulo the
pulse repetition interval) falls inside the pulse is therefore partly eclipsed. With gated
gain, the receiver also stays blind while the gain switch settles after the pulse.
"""

from __future__ import annotations

import numpy as np

from ..units import C_M_S
from .base import HARDWARE_FLOOR_DB, Waveform


class PulsedLFM(Waveform):
    name = "Pulsed LFM"
    is_pulsed = True

    @property
    def bandwidth_hz(self) -> float:
        return self.cfg.bandwidth_hz

    @property
    def papr_db(self) -> float:
        return 0.0

    @property
    def duty_cycle(self) -> float:
        return min(self.cfg.pulse_width_s * self.cfg.prf_hz, 1.0)

    @property
    def rms_duration_s(self) -> float:
        return self.cpi_s / np.sqrt(12.0)

    @property
    def ambiguous_doppler_hz(self) -> float:
        return self.cfg.prf_hz

    @property
    def ambiguous_range_m(self) -> float:
        return C_M_S / self.cfg.prf_hz

    def _folded_delay_s(self, delay_s: float) -> float:
        return delay_s % (1.0 / self.cfg.prf_hz)

    def direct_leakage_db(self, delay_s: float, doppler_hz: float) -> float:
        tau = self.cfg.pulse_width_s
        range_part = self.cfg.window_sidelobe_db if self._folded_delay_s(delay_s) < 2.0 * tau else HARDWARE_FLOOR_DB
        return max(range_part + self.doppler_rejection_db(doppler_hz), HARDWARE_FLOOR_DB)

    def signal_loss_db(self, delay_s: float, doppler_hz: float, blind_extra_s: float = 0.0) -> float:
        """The echo lasts one pulse width; the part arriving while the receiver is blind is lost."""
        tau = self.cfg.pulse_width_s
        visible = np.clip((self._folded_delay_s(delay_s) - blind_extra_s) / tau, 1e-3, 1.0)
        return float(-10.0 * np.log10(visible))

    def notes(self) -> list[str]:
        return [
            f"Range repeats every {self.ambiguous_range_m / 1e3:.0f} km of bi-static path; "
            f"Doppler repeats every {self.cfg.prf_hz:.0f} Hz.",
            f"Time-bandwidth of one pulse: {self.bandwidth_hz * self.cfg.pulse_width_s:.0f}.",
        ]

    def generate(self, fs_hz: float, duration_s: float, rng: np.random.Generator) -> np.ndarray:
        n = int(duration_s * fs_hz)
        t = np.arange(n) / fs_hz
        pri = 1.0 / self.cfg.prf_hz
        local = t % pri
        slope = self.cfg.bandwidth_hz / self.cfg.pulse_width_s
        on = local < self.cfg.pulse_width_s
        phase = np.pi * slope * (local - self.cfg.pulse_width_s / 2.0) ** 2
        return on * np.exp(1j * phase)
