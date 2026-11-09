"""Bi-static geometry for one transmitter, one receiver and one target.

The receiver sits at the origin (at its mast height), the transmitter on the +x
axis. A flat local frame is used for angles and ranges, which is accurate at the
ranges of interest. Earth curvature enters through the propagation models.

Reference: Willis, Bistatic Radar, ch. 3-5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Scenario
from .units import C_M_S


@dataclass(frozen=True)
class BistaticGeometry:
    tx: np.ndarray
    rx: np.ndarray
    target: np.ndarray
    velocity: np.ndarray

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.tx - self.rx))

    @property
    def range_tx_m(self) -> float:
        return float(np.linalg.norm(self.target - self.tx))

    @property
    def range_rx_m(self) -> float:
        return float(np.linalg.norm(self.target - self.rx))

    @property
    def bistatic_range_m(self) -> float:
        """Extra path relative to the direct signal: Rt + Rr - L."""
        return self.range_tx_m + self.range_rx_m - self.baseline_m

    @property
    def bistatic_angle_rad(self) -> float:
        """Angle at the target between the Tx and Rx directions. 180 deg is forward scatter."""
        to_tx = (self.tx - self.target) / self.range_tx_m
        to_rx = (self.rx - self.target) / self.range_rx_m
        return float(np.arccos(np.clip(np.dot(to_tx, to_rx), -1.0, 1.0)))

    @property
    def bistatic_range_rate_m_s(self) -> float:
        los_tx = (self.target - self.tx) / self.range_tx_m
        los_rx = (self.target - self.rx) / self.range_rx_m
        return float(np.dot(self.velocity, los_tx + los_rx))

    def doppler_hz(self, wavelength_m: float) -> float:
        return -self.bistatic_range_rate_m_s / wavelength_m

    @property
    def azimuth_at_rx_rad(self) -> float:
        d = self.target - self.rx
        return float(np.arctan2(d[1], d[0]))

    @property
    def elevation_at_rx_rad(self) -> float:
        d = self.target - self.rx
        return float(np.arcsin(d[2] / np.linalg.norm(d)))

    @property
    def elevation_at_tx_rad(self) -> float:
        d = self.target - self.tx
        return float(np.arcsin(d[2] / np.linalg.norm(d)))

    @property
    def direct_path_elevation_at_rx_rad(self) -> float:
        d = self.tx - self.rx
        return float(np.arcsin(d[2] / np.linalg.norm(d)))

    @property
    def direct_path_elevation_at_tx_rad(self) -> float:
        d = self.rx - self.tx
        return float(np.arcsin(d[2] / np.linalg.norm(d)))

    def range_resolution_m(self, bandwidth_hz: float) -> float:
        """Resolution of a target along the bisector; degrades as cos(beta/2) -> 0."""
        return C_M_S / (2.0 * bandwidth_hz * max(np.cos(self.bistatic_angle_rad / 2.0), 1e-3))


def build_geometry(s: Scenario, target_xyz=None) -> BistaticGeometry:
    """Geometry from the scenario, optionally with the target moved to target_xyz."""
    g = s.geometry
    az = np.deg2rad(g.target_azimuth_deg)
    heading = np.deg2rad(g.target_heading_deg)
    rx = np.array([0.0, 0.0, s.array.height_m])
    tx = np.array([g.baseline_km * 1e3, 0.0, s.transmitter.height_m])
    if target_xyz is None:
        target = np.array(
            [g.target_range_rx_km * 1e3 * np.cos(az), g.target_range_rx_km * 1e3 * np.sin(az), g.target_altitude_m]
        )
    else:
        target = np.asarray(target_xyz, float)
    velocity = g.target_speed_m_s * np.array([np.cos(heading), np.sin(heading), 0.0])
    return BistaticGeometry(tx=tx, rx=rx, target=target, velocity=velocity)
