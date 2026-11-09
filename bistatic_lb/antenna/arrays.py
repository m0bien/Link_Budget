"""Receiver array: dipoles on a ring, with an optional central vertical column.

The array output for unit-norm weights w and a plane wave from direction u is
    power gain = |w^H a(u)|^2,   a_n(u) = sqrt(g_n(u)) exp(j k p_n . u)
so internal channel noise keeps unit gain whatever the weights, and the gain value
is directly comparable with an isotropic antenna. Mutual coupling is neglected.

To use measured or EM-simulated patterns, override ``element_gain``. To add a new
layout, build a ReceiveArray with other ``positions_m`` and ``axes``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Scenario
from ..units import wavelength_m
from .elements import dipole_gain


def unit_vector(az_rad, el_rad):
    az, el = np.asarray(az_rad, float), np.asarray(el_rad, float)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


@dataclass(frozen=True)
class ReceiveArray:
    positions_m: np.ndarray  # (N, 3)
    axes: np.ndarray  # (N, 3) unit vectors along each dipole
    wavelength_m: float
    taper_name: str = "uniform"
    has_elevation: bool = False

    @property
    def n_elements(self) -> int:
        return len(self.positions_m)

    def element_gain(self, u: np.ndarray) -> np.ndarray:
        """Per-element linear gain toward unit vectors u: shape (..., N)."""
        return dipole_gain(np.einsum("...i,ni->...n", u, self.axes))

    def steering(self, az_rad, el_rad) -> np.ndarray:
        """Element responses a(u): shape (..., N)."""
        u = unit_vector(az_rad, el_rad)
        phase = 2.0 * np.pi / self.wavelength_m * np.einsum("...i,ni->...n", u, self.positions_m)
        return np.sqrt(self.element_gain(u)) * np.exp(1j * phase)

    def mean_element_gain_linear(self, az_rad: float, el_rad: float) -> float:
        return float(np.mean(np.abs(self.steering(az_rad, el_rad)) ** 2))

    def beamformer(self, az_rad: float, el_rad: float, null_dir: tuple[float, float] | None = None) -> np.ndarray:
        """Unit-norm weights steered to (az, el), optionally with a null in null_dir."""
        a = self.steering(az_rad, el_rad)
        w = self._taper(az_rad) * a
        if null_dir is not None:
            b = self.steering(*null_dir)
            w = w - b * (np.vdot(b, w) / np.vdot(b, b))
        norm = np.linalg.norm(w)
        # A target exactly in the null direction cannot be separated; fall back to no null.
        return w / norm if norm > 1e-9 else a / np.linalg.norm(a)

    def _taper(self, look_az_rad: float) -> np.ndarray:
        if self.taper_name == "uniform":
            return np.ones(self.n_elements)
        # Cardioid weighting favours elements on the look side and lowers ring sidelobes.
        phi = np.arctan2(self.positions_m[:, 1], self.positions_m[:, 0])
        on_ring = np.linalg.norm(self.positions_m[:, :2], axis=1) > 0
        return np.where(on_ring, 0.15 + 0.85 * (1.0 + np.cos(phi - look_az_rad)) / 2.0, 1.0)

    def gain_linear(self, w: np.ndarray, az_rad, el_rad) -> np.ndarray:
        return np.abs(np.einsum("n,...n->...", np.conj(w), self.steering(az_rad, el_rad))) ** 2

    def gain_dbi(self, w: np.ndarray, az_rad, el_rad):
        return 10.0 * np.log10(np.maximum(self.gain_linear(w, az_rad, el_rad), 1e-12))

    def gain_with_errors_linear(self, w: np.ndarray, az_rad: float, el_rad: float, error_rms_deg: float) -> float:
        """Expected gain with random channel errors; sets the floor of a digital null."""
        a = self.steering(az_rad, el_rad)
        eps2 = np.deg2rad(error_rms_deg) ** 2
        return float(np.abs(np.vdot(w, a)) ** 2 + eps2 * np.sum(np.abs(w * a) ** 2))

    def _azimuth_cut(self, w: np.ndarray, az_rad: float, el_rad: float):
        offsets = np.deg2rad(np.linspace(-180.0, 180.0, 1441))
        return offsets, self.gain_linear(w, az_rad + offsets, np.full_like(offsets, el_rad))

    def beamwidth_az_deg(self, w: np.ndarray, az_rad: float, el_rad: float) -> float:
        """Half-power azimuth width, read from the pattern around the look direction."""
        offsets, cut = self._azimuth_cut(w, az_rad, el_rad)
        centre = int(np.argmin(np.abs(offsets)))
        half = cut[centre] / 2.0
        lo = centre
        while lo > 0 and cut[lo] > half:
            lo -= 1
        hi = centre
        while hi < len(cut) - 1 and cut[hi] > half:
            hi += 1
        return float(np.rad2deg(offsets[hi] - offsets[lo]))

    def peak_sidelobe_db(self, w: np.ndarray, az_rad: float, el_rad: float) -> float:
        """Highest azimuth sidelobe relative to the look-direction gain."""
        offsets, cut = self._azimuth_cut(w, az_rad, el_rad)
        centre = int(np.argmin(np.abs(offsets)))
        lo = centre
        while lo > 0 and cut[lo - 1] <= cut[lo]:
            lo -= 1
        hi = centre
        while hi < len(cut) - 1 and cut[hi + 1] <= cut[hi]:
            hi += 1
        outside = np.r_[cut[:lo], cut[hi + 1 :]]
        return float(10.0 * np.log10(max(outside.max(), 1e-12) / cut[centre])) if outside.size else -np.inf

    def max_spacing_wavelengths(self) -> float:
        """Largest adjacent-ring spacing; above about 0.5 lambda, sidelobes and ambiguity grow."""
        ring = self.positions_m[np.linalg.norm(self.positions_m[:, :2], axis=1) > 0]
        if len(ring) < 2:
            return 0.0
        phi = np.sort(np.arctan2(ring[:, 1], ring[:, 0]))
        radius = np.linalg.norm(ring[0, :2])
        gap = np.max(np.diff(np.r_[phi, phi[0] + 2.0 * np.pi]))
        return float(2.0 * radius * np.sin(gap / 2.0) / self.wavelength_m)

    def integrate_over_sphere(self, w: np.ndarray, temperature_of_elevation, n_el: int = 181, n_az: int = 180) -> float:
        """(1/4pi) integral of gain(u) T(u) dOmega: antenna temperature at the array output."""
        el = (np.arange(n_el) + 0.5) / n_el * np.pi - np.pi / 2.0
        az = (np.arange(n_az) + 0.5) / n_az * 2.0 * np.pi - np.pi
        az_grid, el_grid = np.meshgrid(az, el)
        gain = self.gain_linear(w, az_grid, el_grid)
        solid_angle = np.cos(el_grid) * (np.pi / n_el) * (2.0 * np.pi / n_az)
        return float(np.sum(gain * temperature_of_elevation(el_grid) * solid_angle) / (4.0 * np.pi))

    def angle_crlb_rad(self, snr_linear_isotropic: float, az_rad: float, el_rad: float) -> tuple[float, float | None]:
        """CRLB for (az, el) from the Fisher information of the actual geometry.

        snr_linear_isotropic is E/N0 for one isotropic-gain channel. The complex
        amplitude is unknown, hence the projection (Van Trees, Optimum Array
        Processing, ch. 8). Without a vertical column, only azimuth is returned.
        """
        step = 1e-5
        a = self.steering(az_rad, el_rad)
        derivative = [(self.steering(az_rad + step, el_rad) - self.steering(az_rad - step, el_rad)) / (2 * step)]
        if self.has_elevation:
            derivative.append((self.steering(az_rad, el_rad + step) - self.steering(az_rad, el_rad - step)) / (2 * step))
        d = np.stack(derivative, axis=1)
        projected = d - np.outer(a, np.conj(a) @ d) / np.vdot(a, a)
        fim = 2.0 * snr_linear_isotropic * np.real(np.conj(projected.T) @ projected)
        cov = np.linalg.pinv(fim)
        sigma_az = float(np.sqrt(max(cov[0, 0], 0.0)))
        sigma_el = float(np.sqrt(max(cov[1, 1], 0.0))) if self.has_elevation else None
        return sigma_az, sigma_el


def array_from_scenario(s: Scenario) -> ReceiveArray:
    a = s.array
    design_lambda = wavelength_m(a.design_frequency_hz or s.frequency_hz)
    radius = a.radius_wavelengths * design_lambda if a.n_circular > 1 else 0.0

    phi = 2.0 * np.pi * np.arange(a.n_circular) / a.n_circular
    ring = np.c_[radius * np.cos(phi), radius * np.sin(phi), np.zeros(a.n_circular)]
    if a.orientation == "vertical":
        ring_axes = np.tile([0.0, 0.0, 1.0], (a.n_circular, 1))
    else:
        ring_axes = np.c_[-np.sin(phi), np.cos(phi), np.zeros(a.n_circular)]  # tangent to the ring

    z = (np.arange(a.n_vertical) - (a.n_vertical - 1) / 2.0) * a.vertical_spacing_wavelengths * design_lambda
    column = np.c_[np.zeros(a.n_vertical), np.zeros(a.n_vertical), z]
    column_axes = np.tile([0.0, 0.0, 1.0] if a.orientation == "vertical" else [0.0, 1.0, 0.0], (a.n_vertical, 1))

    return ReceiveArray(
        positions_m=np.vstack([ring, column]),
        axes=np.vstack([ring_axes, column_axes]),
        wavelength_m=wavelength_m(s.frequency_hz),
        taper_name=a.taper,
        has_elevation=a.n_vertical >= 2,
    )
