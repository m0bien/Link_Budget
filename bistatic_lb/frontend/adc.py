"""ADC arithmetic.

SQNR of a full-scale sine = 6.02 ENOB + 1.76 dB, spread over the Nyquist band fs/2.
Oversampling by fs/(2B) moves part of that noise outside the signal band.
"""

from __future__ import annotations

import numpy as np


def ideal_sqnr_db(effective_bits: float) -> float:
    return 6.02 * effective_bits + 1.76


def quantization_noise_in_band_dbm(full_scale_dbm: float, effective_bits: float, sample_rate_hz: float, bandwidth_hz: float) -> float:
    processing_gain_db = 10.0 * np.log10(sample_rate_hz / (2.0 * bandwidth_hz))
    return full_scale_dbm - ideal_sqnr_db(effective_bits) - processing_gain_db


def sensitivity_loss_db(thermal_at_adc_dbm: float, quantization_dbm: float) -> float:
    """SNR lost when quantization noise adds to the amplified thermal noise."""
    return float(10.0 * np.log10(1.0 + 10.0 ** ((quantization_dbm - thermal_at_adc_dbm) / 10.0)))
