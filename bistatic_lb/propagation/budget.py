"""Choose the propagation mechanisms for a path, and say why each one applies.

``path_terms`` is the one entry point, used for the direct path, the two target paths
and clutter. ``compute_direct_path`` then turns the terms into the three signal levels
that drive the front-end: maximum, nominal and minimum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..antenna.arrays import ReceiveArray
from ..antenna.elements import MonopoleTx
from ..config import Scenario
from ..geometry import BistaticGeometry
from ..results import Contribution
from ..units import C_M_S, db
from . import refraction
from .clutter_loss import P2108_MAX_GHZ, P2108_MIN_GHZ, terrestrial_clutter_loss_db
from .diffraction import SphericalEarthResult, general_path_loss_db, spherical_earth_loss_db
from .gases import gas_specific_attenuation_db_km, zenith_gas_attenuation_db
from .multipath import clear_air_fade_applicability, clear_air_fade_depth_db, two_ray
from .rain import rain_attenuation_db
from .surface import GROUND_TYPES
from .terrain import SmoothEarth, TerrainSource

NEGLIGIBLE_DB = 0.1
SIGNIFICANT_REFLECTION = 0.1  # below this magnitude, the ground ray changes the signal by under 1 dB
DUCT_ENHANCEMENT_DB = 3.0  # engineering allowance above free space during ducting


@dataclass(frozen=True)
class PathTerms:
    contributions: list[Contribution]
    free_space_db: float
    gas_db: float
    diffraction_db: float
    clutter_db: float
    multipath_nominal_db: float  # gain: positive means more signal
    multipath_mean_db: float
    multipath_max_db: float
    multipath_min_db: float
    rain_db: float
    fade_db: float
    line_of_sight: bool
    effective_radius_km: float

    @property
    def nominal_loss_db(self) -> float:
        return self.free_space_db + self.gas_db + self.diffraction_db + self.clutter_db - self.multipath_nominal_db

    @property
    def excess_loss_mean_db(self) -> float:
        """Loss beyond free space, with rain and phase-averaged multipath, for moving targets."""
        return self.gas_db + self.diffraction_db + self.clutter_db + self.rain_db - self.multipath_mean_db


def free_space_loss_db(d_km: float, frequency_hz: float) -> float:
    """P.525: 20 log10(4 pi d / lambda)."""
    return float(20.0 * np.log10(4.0 * np.pi * d_km * 1e3 * frequency_hz / C_M_S))


def path_terms(
    s: Scenario,
    d_km: float,
    h1_m: float,
    h2_m: float,
    terrain: TerrainSource | None = None,
    clutter_ends: int = 0,
    elevation_rad: float = 0.0,
) -> PathTerms:
    """Evaluate every mechanism on one path. h1 is at the transmit end, h2 at the far end."""
    env = s.environment
    f = s.frequency_hz
    f_ghz = f / 1e9
    vertical = s.array.orientation == "vertical"
    surface = GROUND_TYPES[env.ground_type]
    terrain = terrain or SmoothEarth()
    k = refraction.k_factor(env.refractivity_gradient_n_per_km)
    a_km = refraction.effective_earth_radius_km(env.refractivity_gradient_n_per_km)
    ducting = refraction.is_ducting(env.refractivity_gradient_n_per_km)
    items: list[Contribution] = []

    fsl = free_space_loss_db(d_km, f)
    items.append(Contribution("Free-space loss", fsl, "dB", True, "Always applies: the basic spreading of the wave.", "P.525"))

    items.append(
        Contribution(
            "Refraction (k-factor)", k, "", True,
            f"Refractivity gradient {env.refractivity_gradient_n_per_km:.0f} N/km gives an effective Earth radius of {a_km:.0f} km."
            + (" This gradient forms a duct, so signals can travel far past the horizon." if ducting else ""),
            "P.453",
        )
    )

    # Diffraction: the Fresnel clearance decides whether it applies.
    if terrain.is_smooth():
        sph: SphericalEarthResult = spherical_earth_loss_db(d_km, h1_m, h2_m, f, a_km, surface, vertical)
        diffraction = sph.loss_db
        method = "Spherical Earth"
        ref = "P.526 sec. 3.2"
    else:
        distances, heights = terrain.profile(d_km)
        diffraction, sph = general_path_loss_db(distances, heights, h1_m, h2_m, f, a_km, surface, vertical)
        method = "Terrain profile (Bullington and spherical Earth)"
        ref = "P.526 sec. 4.5"
    clear_los = diffraction < NEGLIGIBLE_DB
    if np.isnan(sph.required_clearance_m):
        clearance_text = f"The path ({d_km:.1f} km) is beyond the radio horizon ({sph.los_distance_km:.1f} km)."
    else:
        clearance_text = (
            f"Clearance {sph.clearance_m:.1f} m against the {sph.required_clearance_m:.1f} m needed for zero loss: "
            f"{sph.case}."
        )
    items.append(
        Contribution(
            f"Diffraction: {method}", diffraction, "dB", not clear_los,
            clearance_text + (" No diffraction loss applies." if clear_los else " The ray grazes or crosses the surface, so it applies."),
            ref,
            "" if terrain.is_smooth() else "Effective heights use a simplified smooth-surface fit.",
        )
    )

    # Gases: horizontal path, limited to the whole atmosphere above for a steep path.
    gamma = gas_specific_attenuation_db_km(f, env)
    gas = gamma * d_km
    if elevation_rad > 0.0:
        gas = min(gas, zenith_gas_attenuation_db(f, env) / max(np.sin(elevation_rad), 1e-3))
    use_gas = gas >= NEGLIGIBLE_DB
    items.append(
        Contribution(
            "Atmospheric gases", gas, "dB", use_gas,
            f"Specific attenuation {gamma:.4f} dB/km over this path"
            + (" is significant." if use_gas else f" gives under {NEGLIGIBLE_DB} dB, so it is negligible."),
            "P.676",
            "P.676 starts at 1 GHz; the 1 GHz value is used as an upper bound." if f_ghz < 1.0 else "",
        )
    )

    # Rain: statistics for a stated time percentage, so it is held out of the nominal level.
    rain_path_km = d_km
    if elevation_rad > 0.0:
        rain_path_km = min(d_km, 4.0 / max(np.sin(elevation_rad), 1e-3))
    rain = rain_attenuation_db(f, env.rain_rate_mm_h, rain_path_km, env.fade_time_percent, vertical)
    use_rain = rain >= NEGLIGIBLE_DB
    if env.rain_rate_mm_h <= 0.0:
        rain_reason = "Disabled: the rain rate is zero."
    elif f_ghz < 1.0:
        rain_reason = "Rejected: below the 1 GHz start of P.838, and raindrops are too small to absorb much here."
    elif use_rain:
        rain_reason = f"Exceeded for {env.fade_time_percent:g}% of the time. It lowers the minimum level."
    else:
        rain_reason = f"Rejected: under {NEGLIGIBLE_DB} dB on this path at this frequency."
    items.append(Contribution("Rain attenuation", rain, "dB", use_rain, rain_reason, "P.838, P.530 sec. 2.4.1"))
    rain = rain if use_rain else 0.0

    # Ground reflection: not added when diffraction applies, which already includes the ground.
    nominal_mp = mean_mp = max_mp = min_mp = 0.0
    if clear_los and terrain.is_smooth():
        tr = two_ray(d_km, h1_m, h2_m, f, a_km, surface, env.roughness_rms_m, vertical)
        if tr.reflection_magnitude >= SIGNIFICANT_REFLECTION:
            nominal_mp, mean_mp, max_mp, min_mp = tr.nominal_db, tr.mean_db, tr.maximum_db, tr.minimum_db
            items.append(
                Contribution(
                    "Ground reflection (two-ray)", tr.nominal_db, "dB", True,
                    f"Clear path with a significant specular ray: magnitude {tr.reflection_magnitude:.2f} at "
                    f"{tr.grazing_deg:.2f} deg grazing. Range {tr.minimum_db:.1f} to {tr.maximum_db:+.1f} dB with phase.",
                    "P.527 (ground values), P.530",
                    "Sensitive to exact heights; treat the nominal value as one sample of the range.",
                )
            )
        else:
            items.append(
                Contribution(
                    "Ground reflection (two-ray)", 0.0, "dB", False,
                    f"Rejected: roughness and divergence reduce the ground ray to magnitude {tr.reflection_magnitude:.3f}, under 1 dB effect.",
                    "P.527 (ground values), P.530",
                )
            )
    else:
        items.append(
            Contribution(
                "Ground reflection (two-ray)", 0.0, "dB", False,
                "Rejected: the diffraction model already accounts for the ground on this path."
                if not clear_los
                else "Rejected: a terrain profile gives no single specular point for a two-ray estimate.",
                "P.526",
            )
        )

    # Clear-air multipath fading: a statistic, so it only lowers the minimum level.
    use_fade, fade_reason = clear_air_fade_applicability(d_km, f_ghz, sph.line_of_sight, h1_m, h2_m)
    fade = 0.0
    if use_fade:
        fade = clear_air_fade_depth_db(
            d_km, f_ghz, h1_m, h2_m, env.log10_k_geoclimatic, env.dn75, env.fade_time_percent, env.mean_terrain_elevation_m
        )
    items.append(
        Contribution(
            "Clear-air multipath fade", fade, "dB", use_fade, fade_reason, "P.530 sec. 2.3.1",
            "K and dN75 are inputs here: the ITU grids are not bundled, so check them for your site." if use_fade else "",
        )
    )

    # Terminal clutter.
    clutter = 0.0
    min_path = 1.0 if clutter_ends == 2 else 0.25
    if clutter_ends == 0:
        clutter_reason, use_clutter = "Disabled: no terminal on this path is set as inside clutter.", False
    elif not P2108_MIN_GHZ <= f_ghz <= P2108_MAX_GHZ:
        clutter_reason, use_clutter = (f"Rejected: P.2108 sec. 3.2 starts at {P2108_MIN_GHZ} GHz.", False)
    elif d_km < min_path:
        clutter_reason, use_clutter = (f"Rejected: the path is under the {min_path} km minimum.", False)
    else:
        clutter = clutter_ends * terrestrial_clutter_loss_db(f_ghz, d_km, env.clutter_location_percent)
        clutter_reason, use_clutter = (f"Applied at {clutter_ends} end(s) of the path, as set by the user.", True)
    items.append(
        Contribution(
            "Terminal clutter loss", clutter, "dB", use_clutter, clutter_reason, "P.2108 sec. 3.2",
            "Fitted to terminals up to 6 m high and well below clutter height." if use_clutter else "",
        )
    )

    return PathTerms(
        contributions=items,
        free_space_db=fsl,
        gas_db=gas if use_gas else 0.0,
        diffraction_db=diffraction if not clear_los else 0.0,
        clutter_db=clutter,
        multipath_nominal_db=nominal_mp,
        multipath_mean_db=mean_mp,
        multipath_max_db=max_mp,
        multipath_min_db=min_mp,
        rain_db=rain,
        fade_db=fade,
        line_of_sight=clear_los,
        effective_radius_km=a_km,
    )


@dataclass(frozen=True)
class DirectPathResult:
    terms: PathTerms
    eirp_dbw: float
    tx_gain_dbi: float
    element_gain_dbi: float
    array_gain_toward_tx_dbi: float
    polarization_loss_db: float
    rx_mismatch_loss_db: float
    element_nominal_dbm: float
    element_maximum_dbm: float
    element_minimum_dbm: float
    output_nominal_dbm: float
    maximum_note: str
    closest_target_element_dbm: float


def transmit_eirp_dbw(s: Scenario, monopole: MonopoleTx, elevation_rad: float, tx_mismatch_db: float) -> tuple[float, float]:
    """EIRP toward an elevation angle, and the antenna gain used."""
    gain = monopole.gain_dbi(elevation_rad)
    if s.transmitter.eirp_override_dbw is not None:
        return s.transmitter.eirp_override_dbw, gain
    return float(db(s.transmitter.power_w)) - s.transmitter.line_loss_db - tx_mismatch_db + gain, gain


def compute_direct_path(
    s: Scenario,
    geometry: BistaticGeometry,
    array: ReceiveArray,
    weights: np.ndarray,
    monopole: MonopoleTx,
    terrain: TerrainSource,
    tx_mismatch_db: float,
    rx_mismatch_db: float,
) -> DirectPathResult:
    """The strongest signal in a bi-static receiver, at each element and after beamforming."""
    env = s.environment
    a = s.array
    d_km = geometry.baseline_m / 1e3
    terms = path_terms(
        s, d_km, s.transmitter.height_m, a.height_m, terrain,
        clutter_ends=int(env.tx_in_clutter) + int(env.rx_in_clutter),
    )
    eirp, tx_gain = transmit_eirp_dbw(s, monopole, geometry.direct_path_elevation_at_tx_rad, tx_mismatch_db)
    el_rx = geometry.direct_path_elevation_at_rx_rad

    element_gain = float(db(array.mean_element_gain_linear(0.0, el_rx)))
    # A cross-polarized dipole rejects the vertically polarized monopole.
    polarization = a.cross_pol_isolation_db if a.orientation == "horizontal" else 0.0
    received = eirp + 30.0 - polarization - rx_mismatch_db

    nominal = received + element_gain - terms.nominal_loss_db
    array_gain = float(db(array.gain_with_errors_linear(weights, 0.0, el_rx, a.calibration_error_deg)))
    output_nominal = received + array_gain - terms.nominal_loss_db

    # Maximum: no rain or fade, in-phase ground ray. Ducting can restore a blocked path.
    best_case_loss = terms.free_space_db + terms.gas_db + terms.clutter_db
    if env.include_ducting_in_maximum:
        enhancement = max(terms.multipath_max_db, DUCT_ENHANCEMENT_DB)
        maximum = received + element_gain - best_case_loss + enhancement
        note = (
            f"Free-space level plus {enhancement:.1f} dB: in-phase ground ray or ducting. "
            "Diffraction is removed, because ducting can lift a blocked path to near free space."
        )
    else:
        maximum = nominal - terms.multipath_nominal_db + terms.multipath_max_db
        note = "Nominal level with an in-phase ground ray; ducting is excluded by user choice."

    minimum = nominal - terms.multipath_nominal_db + terms.multipath_min_db - terms.rain_db - terms.fade_db

    # A very close target can also be a strong signal; compare it with the direct path.
    rcs_dbsm = s.target.rcs_dbsm
    r_rx = s.geometry.minimum_target_range_km * 1e3
    r_tx = max(geometry.baseline_m - r_rx, r_rx)
    lam_db = db(C_M_S / s.frequency_hz)
    close_target = (
        eirp + 30.0 + element_gain + rcs_dbsm + 2.0 * lam_db - 3.0 * db(4.0 * np.pi)
        - 2.0 * db(r_tx) - 2.0 * db(r_rx) - polarization - rx_mismatch_db
    )
    close_target = float(close_target)
    return DirectPathResult(
        terms=terms,
        eirp_dbw=eirp,
        tx_gain_dbi=tx_gain,
        element_gain_dbi=element_gain,
        array_gain_toward_tx_dbi=array_gain,
        polarization_loss_db=polarization,
        rx_mismatch_loss_db=rx_mismatch_db,
        element_nominal_dbm=float(nominal),
        element_maximum_dbm=float(max(maximum, close_target)),
        element_minimum_dbm=float(minimum),
        output_nominal_dbm=float(output_nominal),
        maximum_note=note,
        closest_target_element_dbm=close_target,
    )
