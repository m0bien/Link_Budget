"""Waveform models. Register a new waveform by adding it to WAVEFORMS."""

from __future__ import annotations

from ..config import WaveformConfig
from .base import Waveform
from .broadcast import DAB, DVBT, FMBroadcast
from .noise_like import PhaseCoded, ZadoffChu
from .pulsed_lfm import PulsedLFM

WAVEFORMS: dict[str, type[Waveform]] = {
    "pulsed_lfm": PulsedLFM,
    "phase_coded": PhaseCoded,
    "zadoff_chu": ZadoffChu,
    "fm": FMBroadcast,
    "dab": DAB,
    "dvbt": DVBT,
}


def make_waveform(cfg: WaveformConfig) -> Waveform:
    return WAVEFORMS[cfg.kind](cfg)


__all__ = ["Waveform", "WAVEFORMS", "make_waveform"]
