"""Vary any scenario input and record any output: the generic trade-off tool.

    x, y = sweep(scenario, "array.n_circular", [4, 8, 16], "Target SINR (dB)")
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .config import Scenario
from .pipeline import run_link_budget

# Inputs offered by the app: dotted path -> (label, default start, default stop, is integer).
SWEEPABLE = {
    "frequency_hz": ("Frequency (Hz)", 100e6, 3e9, False),
    "transmitter.power_w": ("Transmit power (W)", 100.0, 10_000.0, False),
    "transmitter.height_m": ("Transmitter height (m)", 10.0, 100.0, False),
    "array.n_circular": ("Ring elements", 4, 32, True),
    "array.radius_wavelengths": ("Ring radius (wavelengths)", 0.3, 3.0, False),
    "array.n_vertical": ("Vertical column elements", 0, 12, True),
    "array.height_m": ("Array height (m)", 5.0, 100.0, False),
    "array.calibration_error_deg": ("Channel calibration error (deg)", 0.2, 10.0, False),
    "geometry.baseline_km": ("Baseline (km)", 2.0, 100.0, False),
    "geometry.target_range_rx_km": ("Target range from receiver (km)", 5.0, 150.0, False),
    "geometry.target_azimuth_deg": ("Target azimuth (deg)", 5.0, 175.0, False),
    "geometry.target_altitude_m": ("Target altitude (m)", 50.0, 10_000.0, False),
    "target.rcs_dbsm": ("Target RCS (dBsm)", -20.0, 30.0, False),
    "frontend.adc_bits": ("ADC bits", 8, 20, True),
    "frontend.lna_nf_db": ("LNA noise figure (dB)", 0.5, 6.0, False),
    "frontend.if_gain_db": ("IF gain (dB)", 0.0, 50.0, False),
    "frontend.analog_canceller_db": ("Analog cancellation (dB)", 0.0, 60.0, False),
    "frontend.digital_canceller_limit_db": ("Digital cancellation limit (dB)", 0.0, 80.0, False),
    "waveform.cpi_s": ("Coherent interval (s)", 0.01, 2.0, False),
    "waveform.bandwidth_hz": ("Bandwidth (Hz)", 100e3, 20e6, False),
    "waveform.prf_hz": ("PRF (Hz)", 200.0, 10_000.0, False),
    "detection.false_alarms_per_s": ("False alarms per second", 1e-3, 1.0, False),
    "detection.noncoherent_looks": ("Non-coherent looks", 1, 32, True),
    "tracker.confirm_n": ("Confirmation window N", 3, 10, True),
}


def sweep_values(path: str, start: float, stop: float, points: int) -> np.ndarray:
    """Evenly spaced, or logarithmic when the range spans more than a decade."""
    _, _, _, is_integer = SWEEPABLE.get(path, ("", 0, 0, False))
    if is_integer:
        return np.unique(np.round(np.linspace(start, stop, points)).astype(int))
    if start > 0 and stop / start >= 10.0:
        return np.geomspace(start, stop, points)
    return np.linspace(start, stop, points)


def sweep(
    s: Scenario, path: str, values, metric: str, on_progress: Callable[[float], None] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (values actually run, metric at each). Infeasible settings give NaN."""
    values = np.asarray(values)
    out = np.full(values.shape, np.nan, dtype=float)
    for i, value in enumerate(values):
        try:
            out[i] = run_link_budget(s.with_value(path, value.item())).metrics[metric]
        except ValueError:
            pass  # an input the validators reject, or an M-of-N that cannot be met
        if on_progress:
            on_progress((i + 1) / len(values))
    return values, out
