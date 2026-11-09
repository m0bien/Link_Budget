"""Terrain sources.

A TerrainSource supplies a path profile. To add automatic DEM lookup, implement
``DemTerrain.profile`` (sample a height along the great circle between the
terminals) and return it from ``terrain_from_scenario``. Nothing else needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import EnvironmentConfig


class TerrainSource(ABC):
    @abstractmethod
    def is_smooth(self) -> bool:
        """True when spherical-Earth diffraction alone is the right model."""

    @abstractmethod
    def profile(self, path_km: float) -> tuple[np.ndarray, np.ndarray]:
        """Return distances (km, from the Tx) and heights above sea level (m)."""


class SmoothEarth(TerrainSource):
    def is_smooth(self) -> bool:
        return True

    def profile(self, path_km: float):
        return np.array([0.0, path_km]), np.zeros(2)


class ProfileTerrain(TerrainSource):
    """User-supplied (distance, height) points, such as from a CSV file or as obstacles."""

    def __init__(self, points: list[tuple[float, float]]):
        data = np.array(sorted(points), float)
        self.distances_km, self.heights_m = data[:, 0], data[:, 1]

    def is_smooth(self) -> bool:
        return False

    def profile(self, path_km: float):
        d = np.r_[0.0, self.distances_km[(self.distances_km > 0) & (self.distances_km < path_km)], path_km]
        h = np.interp(d, self.distances_km, self.heights_m)
        return d, h


class DemTerrain(TerrainSource):
    """Extension point for automatic terrain lookup from a digital elevation model."""

    def __init__(self, tx_lat_lon: tuple[float, float], rx_lat_lon: tuple[float, float]):
        self.tx_lat_lon, self.rx_lat_lon = tx_lat_lon, rx_lat_lon

    def is_smooth(self) -> bool:
        return False

    def profile(self, path_km: float):
        raise NotImplementedError("DEM lookup is an extension point and is not part of this version.")


def terrain_from_scenario(env: EnvironmentConfig) -> TerrainSource:
    return ProfileTerrain(env.terrain_profile) if len(env.terrain_profile) >= 2 else SmoothEarth()


def profile_from_csv(path: str) -> list[tuple[float, float]]:
    """Read 'distance_km,height_m' rows, with or without a header line."""
    points = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            try:
                points.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                continue
    return points
