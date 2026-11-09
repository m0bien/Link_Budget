"""Monte Carlo checks on the analytic detection and estimation results."""

from __future__ import annotations

import numpy as np

from ..antenna.arrays import ReceiveArray
from ..units import undb
from .detection import Fluctuation, threshold


def _draw_target_power(rng, snr: float, n: int, trials: int, model: Fluctuation) -> np.ndarray:
    """Per-look target power for each trial: shape (trials, n)."""
    if model.kind == "steady":
        return np.full((trials, n), snr)
    if model.kind == "scan":
        return np.tile(rng.gamma(model.m, snr / model.m, (trials, 1)), (1, n))
    if model.kind == "pulse":
        return rng.gamma(model.m, snr / model.m, (trials, n))
    # Rician: dominant plus diffuse complex parts, constant over the looks.
    k = model.rician_k
    a = np.sqrt(k / (1.0 + k)) + (rng.standard_normal(trials) + 1j * rng.standard_normal(trials)) / np.sqrt(2.0 * (1.0 + k))
    return np.tile((snr * np.abs(a) ** 2)[:, None], (1, n))


def detection_monte_carlo(snr_db: float, pfa: float, n: int, model: Fluctuation, trials: int, seed: int = 1) -> tuple[float, float]:
    """Simulate the square-law detector. Returns (Pd, Pfa) estimates."""
    rng = np.random.default_rng(seed)
    t = threshold(pfa, n)
    power = _draw_target_power(rng, float(undb(snr_db)), n, trials, model)
    noise = (rng.standard_normal((trials, n)) + 1j * rng.standard_normal((trials, n))) / np.sqrt(2.0)
    phase = np.exp(2j * np.pi * rng.random((trials, n)))
    with_target = np.sum(np.abs(np.sqrt(power) * phase + noise) ** 2, axis=1)
    noise_only = rng.gamma(n, 1.0, trials)
    return float(np.mean(with_target > t)), float(np.mean(noise_only > t))


def delay_estimator_rmse(snr_db_values, n_samples: int = 1024, trials: int = 400, seed: int = 1) -> np.ndarray:
    """RMSE of the delay estimate, in samples at the Nyquist rate of a flat spectrum.

    A matched filter with interpolated peak search over all delays. E/N0 is the SNR, and
    the RMS bandwidth is 1/sqrt(12) of the sample rate. Below threshold, outliers appear.
    """
    rng = np.random.default_rng(seed)
    oversample = 8
    n_fine = n_samples * oversample
    freqs = np.fft.fftfreq(n_samples)
    results = []
    for snr_db in np.atleast_1d(snr_db_values):
        snr = float(undb(snr_db))
        errors = np.zeros(trials)
        for i in range(trials):
            true_delay = rng.uniform(0, n_samples)
            # Matched-filter output spectrum: flat signal, with noise scaled so that E/N0 = snr.
            noise = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2.0)
            spectrum = np.sqrt(snr / n_samples) * np.exp(-2j * np.pi * freqs * true_delay) + noise
            padded = np.zeros(n_fine, complex)
            half = n_samples // 2
            padded[:half] = spectrum[:half]
            padded[-half:] = spectrum[-half:]
            output = np.abs(np.fft.ifft(padded)) ** 2
            peak = int(np.argmax(output))
            left, centre, right = output[peak - 1], output[peak], output[(peak + 1) % n_fine]
            offset = 0.5 * (left - right) / (left - 2.0 * centre + right)
            estimate = (peak + offset) / oversample
            errors[i] = (estimate - true_delay + n_samples / 2.0) % n_samples - n_samples / 2.0
        results.append(np.sqrt(np.mean(errors**2)))
    return np.array(results)


def doppler_estimator_rmse(snr_db_values, n_samples: int = 512, trials: int = 400, seed: int = 2) -> np.ndarray:
    """RMSE of a tone-frequency estimate, in cycles per sample, by interpolated FFT peak."""
    rng = np.random.default_rng(seed)
    oversample = 16
    t = np.arange(n_samples)
    results = []
    for snr_db in np.atleast_1d(snr_db_values):
        snr = float(undb(snr_db))
        errors = np.zeros(trials)
        for i in range(trials):
            f_true = rng.uniform(-0.25, 0.25)
            noise = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2.0)
            x = np.sqrt(snr / n_samples) * np.exp(2j * np.pi * f_true * t) + noise
            output = np.abs(np.fft.fft(x, n_samples * oversample)) ** 2
            peak = int(np.argmax(output))
            n_fft = len(output)
            left, centre, right = output[peak - 1], output[peak], output[(peak + 1) % n_fft]
            offset = 0.5 * (left - right) / (left - 2.0 * centre + right)
            estimate = ((peak + offset) / n_fft + 0.5) % 1.0 - 0.5
            errors[i] = (estimate - f_true + 0.5) % 1.0 - 0.5
        results.append(np.sqrt(np.mean(errors**2)))
    return np.array(results)


def angle_estimator_rmse(
    array: ReceiveArray, az_rad: float, el_rad: float, snr_isotropic_db_values, trials: int = 300, seed: int = 3
) -> np.ndarray:
    """RMSE of azimuth, in radians, from a maximum-likelihood grid search with local refinement."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(-np.pi, np.pi, 1441)[:-1]
    manifold = array.steering(grid, np.full_like(grid, el_rad))
    manifold_power = np.sum(np.abs(manifold) ** 2, axis=1)
    a_true = array.steering(az_rad, el_rad)
    results = []
    for snr_db in np.atleast_1d(snr_isotropic_db_values):
        snr = float(undb(snr_db))
        errors = np.zeros(trials)
        for i in range(trials):
            n = array.n_elements
            noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
            x = np.sqrt(snr) * a_true * np.exp(2j * np.pi * rng.random()) + noise
            spectrum = np.abs(np.conj(manifold) @ x) ** 2 / manifold_power
            peak = int(np.argmax(spectrum))
            left, centre, right = spectrum[peak - 1], spectrum[peak], spectrum[(peak + 1) % len(grid)]
            offset = 0.5 * (left - right) / (left - 2.0 * centre + right)
            estimate = grid[peak] + offset * (grid[1] - grid[0])
            errors[i] = (estimate - az_rad + np.pi) % (2.0 * np.pi) - np.pi
        results.append(np.sqrt(np.mean(errors**2)))
    return np.array(results)
