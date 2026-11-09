"""Theoretical tracking accuracy.

* Steady-state Kalman gain and covariance, from the discrete algebraic Riccati equation.
* Alpha-beta gains from the tracking index (Kalata, 1984).
* Posterior Cramer-Rao bound with an information-reduction factor for missed detections
  and clutter (Hernandez, Farina and Ristic, 2006).
"""

from __future__ import annotations

import numpy as np


def constant_velocity_model(scan_s: float, accel_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """State (position, rate) with white-noise acceleration."""
    t = scan_s
    f = np.array([[1.0, t], [0.0, 1.0]])
    q = accel_sigma**2 * np.array([[t**4 / 4.0, t**3 / 2.0], [t**3 / 2.0, t**2]])
    return f, q


def steady_state_covariance(f: np.ndarray, q: np.ndarray, h: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the steady-state (predicted, updated) covariances."""
    # Plain Riccati iteration: slower than a direct solver, but stable for the very
    # different scales of range (m) and range rate (m/s) met here.
    identity = np.eye(f.shape[0])
    predicted = q + r_like_start(h, r, f.shape[0])
    updated = predicted
    for _ in range(20000):
        s = h @ predicted @ h.T + r
        gain = predicted @ h.T @ np.linalg.inv(s)
        updated = (identity - gain @ h) @ predicted
        next_predicted = f @ updated @ f.T + q
        if np.max(np.abs(next_predicted - predicted)) <= 1e-10 * np.max(np.abs(predicted)):
            predicted = next_predicted
            break
        predicted = next_predicted
    return predicted, updated


def r_like_start(h: np.ndarray, r: np.ndarray, size: int) -> np.ndarray:
    """A starting covariance on the scale of the measurement noise."""
    return h.T @ r @ h if h.shape[0] == size else np.eye(size) * float(np.max(r))


def alpha_beta_gains(sigma_meas: float, accel_sigma: float, scan_s: float) -> tuple[float, float, float]:
    """Steady alpha-beta gains for position-only measurements. Returns (index, alpha, beta)."""
    index = accel_sigma * scan_s**2 / sigma_meas
    r = (4.0 + index - np.sqrt(8.0 * index + index**2)) / 4.0
    return float(index), float(1.0 - r**2), float(2.0 * (1.0 - r) ** 2)


def information_reduction_factor(
    pd: float, clutter_per_unit_volume: float, gate_sigma: float = 4.0, trials: int = 4000, seed: int = 7
) -> float:
    """Fraction of the measurement information left after misses and clutter.

    Normalized 1-D measurement with unit variance. clutter_per_unit_volume is the number
    of false measurements per standard deviation of measurement space. Estimated as the
    expected squared score of the association-marginalized likelihood.
    """
    if clutter_per_unit_volume <= 0.0:
        return pd
    rng = np.random.default_rng(seed)
    half = gate_sigma
    mean_false = clutter_per_unit_volume * 2.0 * half
    total = 0.0
    for _ in range(trials):
        points = list(rng.uniform(-half, half, rng.poisson(mean_false)))
        if rng.random() < pd:
            points.append(rng.standard_normal())
        if not points:
            continue
        z = np.array(points)
        kernel = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
        denominator = (1.0 - pd) * clutter_per_unit_volume + pd * np.sum(kernel)
        score = pd * np.sum(z * kernel) / denominator
        total += score**2
    return float(np.clip(total / trials, 0.0, pd))


def pcrlb_series(
    f: np.ndarray, q: np.ndarray, h: np.ndarray, r: np.ndarray, reduction: np.ndarray | float, p0: np.ndarray, scans: int
) -> np.ndarray:
    """Bound on the state covariance at each scan. Returns an array of shape (scans, n, n)."""
    reduction = np.broadcast_to(reduction, (scans,))
    info = np.linalg.inv(p0)
    bound = np.zeros((scans, f.shape[0], f.shape[0]))
    r_info = h.T @ np.linalg.inv(r) @ h
    for k in range(scans):
        predicted = np.linalg.inv(q + f @ np.linalg.inv(info) @ f.T)
        info = predicted + reduction[k] * r_info
        bound[k] = np.linalg.inv(info)
    return bound
