"""The interface every waveform implements.

To add a waveform: subclass Waveform in a new file, then add one line to
``WAVEFORMS`` in waveforms/__init__.py and one option to WaveformConfig.kind.

Levels returned by ``direct_leakage_db`` are relative to the matched-filter peak the
same signal would produce in its own cell. That makes them directly comparable with
the target's power and the post-integration noise level.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import WaveformConfig
from ..units import C_M_S

# Below this, hardware limits (spurs, drift, phase noise) dominate the ideal waveform.
HARDWARE_FLOOR_DB = -90.0
MAIN_LOBE_3DB_HALF_WIDTH_CELLS = 0.65  # Doppler window main lobe, in cells of 1 / CPI


class Waveform(ABC):
    name = "waveform"
    is_pulsed = False  # True when the direct signal is absent between pulses, so gain can be switched

    def __init__(self, cfg: WaveformConfig):
        self.cfg = cfg

    # --- Properties each waveform must define ---------------------------------------
    @property
    @abstractmethod
    def bandwidth_hz(self) -> float:
        """Occupied bandwidth; sets the ADC rate and the receiver noise bandwidth."""

    @property
    @abstractmethod
    def papr_db(self) -> float:
        """Peak-to-mean power that the PA and the ADC must both accommodate."""

    @property
    @abstractmethod
    def duty_cycle(self) -> float:
        """Fraction of time the transmitter is on."""

    @abstractmethod
    def direct_leakage_db(self, delay_s: float, doppler_hz: float) -> float:
        """Direct-path level in a target's range-Doppler cell, relative to its own peak."""

    @abstractmethod
    def generate(self, fs_hz: float, duration_s: float, rng: np.random.Generator) -> np.ndarray:
        """Complex baseband samples, used to check the model values against a real signal."""

    # --- Shared behaviour; override where the waveform differs --------------------------
    @property
    def rms_bandwidth_hz(self) -> float:
        """RMS bandwidth for the delay bound. A flat spectrum gives B / sqrt(12)."""
        return self.bandwidth_hz / np.sqrt(12.0)

    @property
    def cpi_s(self) -> float:
        return self.cfg.cpi_s

    @property
    def rms_duration_s(self) -> float:
        """RMS duration for the Doppler bound. A uniform CPI gives T / sqrt(12)."""
        return self.cpi_s / np.sqrt(12.0)

    @property
    def integration_time_s(self) -> float:
        """Time the transmitter delivers energy during one CPI."""
        return self.cpi_s * self.duty_cycle

    @property
    def time_bandwidth(self) -> float:
        return self.bandwidth_hz * self.integration_time_s

    @property
    def processing_gain_db(self) -> float:
        """Matched-filter SNR gain over the raw signal bandwidth, including window loss."""
        return float(10.0 * np.log10(self.time_bandwidth) - self.cfg.window_loss_db)

    @property
    def doppler_resolution_hz(self) -> float:
        return 1.0 / self.cpi_s

    @property
    def ambiguous_doppler_hz(self) -> float | None:
        """Doppler repeat interval; None for continuous waveforms."""
        return None

    @property
    def ambiguous_range_m(self) -> float | None:
        """Bi-static path-difference repeat distance; None if there is no strong repeat."""
        return None

    def doppler_cells_from_zero(self, doppler_hz: float) -> float:
        """Distance of this Doppler from the nearest zero-Doppler line, in Doppler cells of 1 / CPI."""
        folded = abs(doppler_hz)
        if self.ambiguous_doppler_hz:
            prf = self.ambiguous_doppler_hz
            folded = abs((doppler_hz + prf / 2.0) % prf - prf / 2.0)
        return float(folded / self.doppler_resolution_hz)

    def doppler_rejection_db(self, doppler_hz: float) -> float:
        """Response of the Doppler filter at this target Doppler to a stationary return (clutter, direct path).

        A Gaussian main lobe, 3 dB down at 0.65 cells as for windows of about -35 to -45 dB,
        falling until it meets the window's sidelobe level, which then holds.
        """
        cells = self.doppler_cells_from_zero(doppler_hz)
        return float(max(-3.0 * (cells / MAIN_LOBE_3DB_HALF_WIDTH_CELLS) ** 2, self.cfg.window_sidelobe_db))

    def signal_loss_db(self, delay_s: float, doppler_hz: float, blind_extra_s: float = 0.0) -> float:
        """Waveform-specific target loss, such as eclipsing or Doppler mismatch.

        blind_extra_s is how long the receiver stays blind after a direct pulse ends, for pulsed waveforms.
        """
        return 0.0

    def number_of_range_doppler_cells(self, max_range_m: float, max_doppler_hz: float) -> float:
        range_cells = max(max_range_m / (C_M_S / self.bandwidth_hz), 1.0)
        doppler_span = 2.0 * max_doppler_hz
        if self.ambiguous_doppler_hz:
            doppler_span = min(doppler_span, self.ambiguous_doppler_hz)
        return range_cells * max(doppler_span * self.cpi_s, 1.0)

    def notes(self) -> list[str]:
        return []
