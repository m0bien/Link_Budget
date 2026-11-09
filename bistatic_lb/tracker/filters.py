"""Track filters. One class handles both the linear and the extended Kalman case.

A filter is defined by its state transition and a measurement function. The linear filter
(bi-static range and range rate) has a constant H. The EKF (Cartesian state, with bi-static
range, range rate and azimuth as measurements) differentiates its measurement function.

Association: 'nearest' takes the closest gated measurement; 'pda' weights all of them
(Bar-Shalom, Li and Kirubarajan, ch. 11).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class GaussianTrack:
    def __init__(
        self,
        x0: np.ndarray,
        p0: np.ndarray,
        f: np.ndarray,
        q: np.ndarray,
        measure: Callable[[np.ndarray], np.ndarray],
        wrap_dims: tuple[int, ...] = (),
    ):
        self.x, self.p, self.f, self.q, self.measure = x0.astype(float), p0.astype(float), f, q, measure
        self.wrap_dims = wrap_dims

    def predict(self) -> None:
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q

    def jacobian(self) -> np.ndarray:
        z0 = self.measure(self.x)
        h = np.zeros((z0.size, self.x.size))
        for i in range(self.x.size):
            step = 1e-3 * max(abs(self.x[i]), 1.0)
            delta = np.zeros(self.x.size)
            delta[i] = step
            h[:, i] = self._wrap(self.measure(self.x + delta) - self.measure(self.x - delta)) / (2.0 * step)
        return h

    def _wrap(self, innovation: np.ndarray) -> np.ndarray:
        out = np.array(innovation, float)
        for d in self.wrap_dims:
            out[d] = (out[d] + np.pi) % (2.0 * np.pi) - np.pi
        return out

    def gate(self, r: np.ndarray, measurements: np.ndarray, gate_sigma: float):
        """Return the measurements' innovations inside the validation gate, H and S."""
        h = self.jacobian()
        s = h @ self.p @ h.T + r
        predicted = self.measure(self.x)
        s_inverse = np.linalg.inv(s)
        kept = []
        for z in measurements:
            nu = self._wrap(z - predicted)
            if nu @ s_inverse @ nu <= gate_sigma**2:
                kept.append(nu)
        return np.array(kept).reshape(-1, predicted.size), h, s

    def update_nearest(self, innovations: np.ndarray, h: np.ndarray, s: np.ndarray) -> bool:
        if innovations.size == 0:
            return False
        s_inverse = np.linalg.inv(s)
        d2 = np.einsum("ni,ij,nj->n", innovations, s_inverse, innovations)
        nu = innovations[int(np.argmin(d2))]
        k = self.p @ h.T @ s_inverse
        self.x = self.x + k @ nu
        self.p = (np.eye(self.x.size) - k @ h) @ self.p
        return True

    def update_pda(self, innovations: np.ndarray, h: np.ndarray, s: np.ndarray, pd: float, clutter_density: float) -> bool:
        """Probabilistic data association, assuming gate probability near one."""
        if innovations.size == 0:
            return False
        s_inverse = np.linalg.inv(s)
        d2 = np.einsum("ni,ij,nj->n", innovations, s_inverse, innovations)
        norm = np.sqrt(np.linalg.det(2.0 * np.pi * s))
        likelihood = pd * np.exp(-0.5 * d2) / norm
        weights = np.r_[likelihood, (1.0 - pd) * max(clutter_density, 1e-30)]
        weights = weights / weights.sum()
        beta, beta0 = weights[:-1], weights[-1]

        combined = beta @ innovations
        k = self.p @ h.T @ s_inverse
        self.x = self.x + k @ combined
        updated = (np.eye(self.x.size) - k @ h) @ self.p
        spread = k @ (np.einsum("n,ni,nj->ij", beta, innovations, innovations) - np.outer(combined, combined)) @ k.T
        self.p = beta0 * self.p + (1.0 - beta0) * updated + spread
        return True
