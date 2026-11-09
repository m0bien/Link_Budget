"""Standard broadcast waveforms: FM, DAB and DVB-T.

These assume the signal is known to the receiver, either because the system transmits
it or through a reference channel. Parameters: DAB Mode I from ETSI EN 300 401, DVB-T
8k mode from ETSI EN 300 744, and FM behaviour from Griffiths and Baker, Passive Radar.

Zero-Doppler range sidelobes are structured (programme content for FM, pilots and guard
intervals for OFDM). Away from zero Doppler, the level falls to the 1/(BT) noise floor.
"""

from __future__ import annotations

import numpy as np

from ..units import C_M_S
from .base import HARDWARE_FLOOR_DB, Waveform

OFDM_PAPR_DB = 10.5  # power exceeded about 1e-5 of the time (Rayleigh envelope: P(x) = exp(-x))

# Effective bandwidth and near-zero-Doppler sidelobe level depend on programme content.
FM_PRESETS = {
    "low": {"bandwidth_hz": 15e3, "structure_db": -10.0},
    "typical": {"bandwidth_hz": 50e3, "structure_db": -20.0},
    "high": {"bandwidth_hz": 100e3, "structure_db": -25.0},
}
DVBT_BANDWIDTH_HZ = {6: 5.71e6, 7: 6.66e6, 8: 7.61e6}
DVBT_USEFUL_SYMBOL_S = {6: 1194.667e-6, 7: 1024e-6, 8: 896e-6}
DVBT_CARRIERS = 6817
DAB_BANDWIDTH_HZ = 1.536e6
DAB_USEFUL_SYMBOL_S = 1e-3
DAB_GUARD_S = 246e-6
DAB_NULL_S = 1.297e-3
DAB_FRAME_S = 96e-3
DAB_CARRIERS = 1536


def _continuous_leakage(wave: Waveform, doppler_hz: float, structure_db: float) -> float:
    """Structured zero-Doppler sidelobes, falling with the Doppler filter's response to the 1/(BT) floor."""
    noise_floor = max(-10.0 * np.log10(wave.time_bandwidth), HARDWARE_FLOOR_DB)
    return float(max(structure_db + wave.doppler_rejection_db(doppler_hz), noise_floor))


def _ofdm_samples(n_carriers: int, useful_s: float, guard_s: float, fs_hz: float, duration_s: float, rng) -> np.ndarray:
    """Random QPSK OFDM with a cyclic prefix, sampled at fs_hz."""
    n_fft = int(round(useful_s * fs_hz))
    n_guard = int(round(guard_s * fs_hz))
    n_total = int(duration_s * fs_hz)
    out = []
    while sum(map(len, out)) < n_total:
        spectrum = np.zeros(n_fft, complex)
        bins = np.arange(-n_carriers // 2, n_carriers // 2 + 1)
        symbols = np.exp(1j * (np.pi / 4.0 + np.pi / 2.0 * rng.integers(0, 4, bins.size)))
        spectrum[bins % n_fft] = symbols
        time = np.fft.ifft(spectrum) * np.sqrt(n_fft)
        out.append(np.r_[time[-n_guard:] if n_guard else time[:0], time])
    return np.concatenate(out)[:n_total]


class FMBroadcast(Waveform):
    name = "FM broadcast"

    @property
    def _preset(self) -> dict:
        return FM_PRESETS[self.cfg.fm_programme]

    @property
    def bandwidth_hz(self) -> float:
        return self._preset["bandwidth_hz"]

    @property
    def papr_db(self) -> float:
        return 0.0

    @property
    def duty_cycle(self) -> float:
        return 1.0

    def direct_leakage_db(self, delay_s: float, doppler_hz: float) -> float:
        return _continuous_leakage(self, doppler_hz, self._preset["structure_db"])

    def notes(self) -> list[str]:
        return [
            f"Effective bandwidth assumed {self.bandwidth_hz / 1e3:.0f} kHz for '{self.cfg.fm_programme}' programme content.",
            "Quiet programme content shrinks bandwidth sharply, degrading both resolution and processing gain.",
        ]

    def generate(self, fs_hz: float, duration_s: float, rng: np.random.Generator) -> np.ndarray:
        """Frequency modulation by Gaussian noise whose RMS deviation matches the preset."""
        n = int(duration_s * fs_hz)
        spectrum = np.fft.rfft(rng.standard_normal(n))
        freqs = np.fft.rfftfreq(n, 1.0 / fs_hz)
        spectrum[freqs > self.bandwidth_hz / 2.0] = 0.0
        message = np.fft.irfft(spectrum, n)
        message *= self.rms_bandwidth_hz / np.std(message)
        return np.exp(2j * np.pi * np.cumsum(message) / fs_hz)


class DAB(Waveform):
    name = "DAB (Mode I)"

    @property
    def bandwidth_hz(self) -> float:
        return DAB_BANDWIDTH_HZ

    @property
    def papr_db(self) -> float:
        return OFDM_PAPR_DB

    @property
    def duty_cycle(self) -> float:
        return 1.0 - DAB_NULL_S / DAB_FRAME_S

    @property
    def ambiguous_range_m(self) -> float:
        return C_M_S * DAB_USEFUL_SYMBOL_S

    def direct_leakage_db(self, delay_s: float, doppler_hz: float) -> float:
        return _continuous_leakage(self, doppler_hz, -25.0)

    def notes(self) -> list[str]:
        return [
            f"Guard interval gives an ambiguity near {self.ambiguous_range_m / 1e3:.0f} km of path; "
            "remove by reference-signal conditioning.",
            "The null symbol at each frame start causes a small repeating sidelobe.",
        ]

    def generate(self, fs_hz: float, duration_s: float, rng: np.random.Generator) -> np.ndarray:
        return _ofdm_samples(DAB_CARRIERS, DAB_USEFUL_SYMBOL_S, DAB_GUARD_S, fs_hz, duration_s, rng)


class DVBT(Waveform):
    name = "DVB-T (8k)"

    @property
    def bandwidth_hz(self) -> float:
        return DVBT_BANDWIDTH_HZ[self.cfg.dvbt_channel_mhz]

    @property
    def papr_db(self) -> float:
        return OFDM_PAPR_DB

    @property
    def duty_cycle(self) -> float:
        return 1.0

    @property
    def useful_symbol_s(self) -> float:
        return DVBT_USEFUL_SYMBOL_S[self.cfg.dvbt_channel_mhz]

    @property
    def ambiguous_range_m(self) -> float:
        return C_M_S * self.useful_symbol_s / 3.0

    def direct_leakage_db(self, delay_s: float, doppler_hz: float) -> float:
        return _continuous_leakage(self, doppler_hz, -25.0)

    def notes(self) -> list[str]:
        return [
            f"Scattered pilots produce ambiguities starting near {self.ambiguous_range_m / 1e3:.0f} km of path; "
            "reference conditioning should suppress them.",
        ]

    def generate(self, fs_hz: float, duration_s: float, rng: np.random.Generator) -> np.ndarray:
        return _ofdm_samples(DVBT_CARRIERS, self.useful_symbol_s, self.useful_symbol_s / 8.0, fs_hz, duration_s, rng)
