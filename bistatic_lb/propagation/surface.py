"""Ground electrical constants.

Representative values per ground class, in the spirit of Rec. ITU-R P.527. P.527-6
has a full frequency-dependent soil and water model; to use it, add a function that
returns a SurfaceConstants for the frequency and pass it in place of this table.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SurfaceConstants:
    permittivity: float
    conductivity_s_m: float
    is_sea: bool = False


GROUND_TYPES = {
    "sea": SurfaceConstants(70.0, 5.0, is_sea=True),
    "wet": SurfaceConstants(30.0, 0.02),
    "medium": SurfaceConstants(15.0, 0.001),
    "dry": SurfaceConstants(3.0, 0.0001),
}
