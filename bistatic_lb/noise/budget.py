"""Stage 1: how much noise reaches the detector, and where it comes from.

The sky and ground are integrated over the actual pattern, per element and at the
beamformer output. Receiver noise is uncorrelated between channels, so it keeps unit
gain with unit-norm weights. External noise is spatially correlated, so its output
level depends on the pattern. The two therefore get different array gains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..antenna.arrays import ReceiveArray
from ..config import Scenario
from ..propagation.gases import gas_specific_attenuation_db_km, zenith_gas_attenuation_db
from ..propagation.rain import rain_specific_attenuation_db_km
from ..results import Contribution
from ..units import K_BOLTZMANN, T_CMB_K, db
from . import itu_p372 as itu
from .receiver import Lineup
from .statistics import NoiseStatistics, man_made_decile_deviations_db, noise_statistics

SOLAR_SOLID_ANGLE_SR = 6.8e-5
RAIN_HEIGHT_KM = 4.0
SIGNIFICANCE_FRACTION = 0.01  # a source below 1% of receiver noise is listed as negligible


@dataclass(frozen=True)
class NoiseResult:
    contributions: list[Contribution]
    external_element_k: float
    external_output_k: float
    receiver_k: float
    system_element_k: float
    system_output_k: float
    noise_element_dbm_hz: float
    noise_output_dbm_hz: float
    dominant_source: str
    array_noise_gain_note: str
    statistics: NoiseStatistics
    design_output_k: float
    design_output_dbm_hz: float


def sky_temperature_function(s: Scenario):
    """Return T(elevation): atmosphere, rain, galaxy and CMB above the horizon, ground below."""
    env = s.environment
    a_zenith = zenith_gas_attenuation_db(s.frequency_hz, env)
    gamma_surface = gas_specific_attenuation_db_km(s.frequency_hz, env)
    t_galactic = float(itu.fa_to_temperature_k(itu.galactic_fam_db(s.frequency_hz)))
    rain_gamma = rain_specific_attenuation_db_km(s.frequency_hz, env.rain_rate_mm_h)

    def attenuation(el):
        sin_el = np.maximum(np.sin(el), 1e-3)
        # Very low rays are limited to a long horizontal path, not 1/sin(el) -> infinity.
        gas = np.minimum(a_zenith / sin_el, gamma_surface * 300.0)
        rain = np.minimum(rain_gamma * RAIN_HEIGHT_KM / sin_el, rain_gamma * 50.0)
        return gas + rain

    def temperature(el_rad):
        el = np.asarray(el_rad, float)
        sky = itu.brightness_through_absorption_k(attenuation(el), t_galactic + T_CMB_K)
        return np.where(el >= 0.0, sky, env.ground_temperature_k)

    def emission(el_rad):
        """The atmosphere's own emission, without the background behind it."""
        el = np.asarray(el_rad, float)
        return np.where(el >= 0.0, itu.brightness_through_absorption_k(attenuation(el), 0.0), 0.0)

    return temperature, emission, a_zenith


def _horizon_weight(array: ReceiveArray, w: np.ndarray) -> float:
    """Pattern-averaged gain near the horizon, relative to the P.372 reference monopole."""
    az = np.deg2rad(np.arange(0.0, 360.0, 2.0))
    el = np.deg2rad(np.arange(0.0, 11.0, 2.0))
    az_grid, el_grid = np.meshgrid(az, el)
    mean_gain = float(np.mean(array.gain_linear(w, az_grid, el_grid)))
    return mean_gain / 10.0 ** (itu.REFERENCE_MONOPOLE_GAIN_DBI / 10.0)


def compute_noise(
    s: Scenario,
    array: ReceiveArray,
    weights: np.ndarray,
    look_dir: tuple[float, float],
    lineup: Lineup,
    rx_mismatch_db: float,
) -> NoiseResult:
    env = s.environment
    f = s.frequency_hz
    sky, emission, a_zenith = sky_temperature_function(s)
    receiver_k = lineup.noise_temperature_k
    threshold_k = SIGNIFICANCE_FRACTION * receiver_k

    # Man-made noise arrives near the horizon, so scale it by this antenna's horizon gain.
    if env.man_made_override_db is not None:
        t_mm_ref = float(itu.fa_to_temperature_k(env.man_made_override_db))
        mm_reason = "Measured site value entered by the user replaces the P.372 fit."
        mm_note = ""
        mm_used = True
    elif not itu.man_made_applies(f, env.man_made_category):
        t_mm_ref = 0.0
        mm_reason = (
            "Rejected: the quiet-rural curve stops at 30 MHz. It is ionospherically reflected noise, "
            "and above that the ionosphere is transparent, so galactic noise dominates."
        )
        mm_note = ""
        mm_used = False
    else:
        mm_used = True
        t_mm_ref = float(itu.fa_to_temperature_k(itu.man_made_fam_db(f, env.man_made_category)))
        mm_reason = f"P.372 median for the '{env.man_made_category}' category."
        mm_note = (
            f"Estimate: {f/1e6:.0f} MHz is above the 250 MHz limit of eq. 17. Continues from there at the slope of "
            "Fig. 3 curve A, which is for city noise and stops near 900 MHz. Measure at the site if it matters."
            if f > itu.MAN_MADE_MAX_VALID_HZ
            else ""
        )

    per_element_horizon, per_element_sky = [], []
    for n in range(array.n_elements):
        single = np.zeros(array.n_elements, complex)
        single[n] = 1.0
        per_element_horizon.append(_horizon_weight(array, single))
        per_element_sky.append(array.integrate_over_sphere(single, sky, n_el=91, n_az=8))
    t_mm_element = t_mm_ref * float(np.mean(per_element_horizon))
    t_mm_output = t_mm_ref * _horizon_weight(array, weights)

    # Sky and ground at the output; separate the parts for the applicability table.
    t_sky_ground_element = float(np.mean(per_element_sky))
    t_sky_ground_output = array.integrate_over_sphere(weights, sky)
    t_ground_output = array.integrate_over_sphere(
        weights, lambda el: np.where(np.asarray(el) < 0, env.ground_temperature_k, 0.0)
    )
    t_atmosphere_output = array.integrate_over_sphere(weights, emission)
    t_extraterrestrial_output = max(t_sky_ground_output - t_ground_output - t_atmosphere_output, 0.0)

    t_sun = 0.0
    if env.sun_in_beam:
        peak_gain = float(array.gain_linear(weights, *look_dir))
        t_sun = peak_gain * SOLAR_SOLID_ANGLE_SR / (4.0 * np.pi) * itu.quiet_sun_brightness_k(f)

    external_element = t_mm_element + t_sky_ground_element
    external_output = t_mm_output + t_sky_ground_output + t_sun

    # Reflective mismatch reduces external noise but not receiver noise.
    mismatch = 10.0 ** (-rx_mismatch_db / 10.0)
    system_element = external_element * mismatch + receiver_k
    system_output = external_output * mismatch + receiver_k

    contributions = [
        Contribution(
            "Man-made noise", t_mm_output, "K", mm_used,
            mm_reason + ("" if not mm_used else " Above 1% of receiver noise." if t_mm_output > threshold_k else " Below 1% of receiver noise."),
            "P.372-18 eq. 17, Table 1, Figs 3 and 39", mm_note,
        ),
        Contribution(
            "Galactic and cosmic background", t_extraterrestrial_output, "K", t_extraterrestrial_output > threshold_k,
            "Falls about 23 dB per decade, so it matters most at VHF and low UHF."
            if t_extraterrestrial_output > threshold_k
            else "Below 1% of receiver noise at this frequency, so it does not limit sensitivity.",
            "P.372-18 eq. 15, Fig. 3 curve B",
            "Equation stated up to about 100 MHz; Fig. 3 curve B continues at close to the same slope, to about 5 GHz."
            if f > itu.GALACTIC_STATED_MAX_HZ
            else "Shielded by the ionosphere if foF2 is above about a third of this frequency.",
        ),
        Contribution(
            "Atmospheric gases and rain emission", t_atmosphere_output, "K", t_atmosphere_output > threshold_k,
            f"Zenith gas absorption {a_zenith:.3f} dB; absorbers emit at about 275 K."
            + (" Rain is included." if env.rain_rate_mm_h > 0 else ""),
            "P.372-18 eq. 10 and 11, P.676, P.838",
            "275 K is the P.372-18 rain value. Its clear-air Tmr data file is not bundled, so 275 K is used for clear air too.",
        ),
        Contribution(
            "Ground pick-up", t_ground_output, "K", t_ground_output > threshold_k,
            "The dipole pattern is not shielded from the ground, which radiates at about 290 K.",
            "Pattern integration",
            "No mast or ground-plane image in this estimate.",
        ),
        Contribution(
            "Sun in beam", t_sun, "K", env.sun_in_beam,
            "Enabled by the user." if env.sun_in_beam else "Disabled: the beam is not assumed to point at the Sun.",
            "P.372-18 Fig. 3 curve D",
            "Read from the figure. A disturbed Sun can be far higher.",
        ),
        Contribution(
            "Lightning (atmospheric) noise", 0.0, "K", False,
            "Rejected: an HF and lower-VHF effect, negligible against man-made and galactic noise at 50 MHz and above.",
            "P.372-18 Part 5, Fig. 2",
        ),
    ]

    sources = {
        "man-made": t_mm_output * mismatch,
        "galactic": t_extraterrestrial_output * mismatch,
        "atmosphere": t_atmosphere_output * mismatch,
        "ground": t_ground_output * mismatch,
        "sun": t_sun * mismatch,
        "receiver": receiver_k,
    }
    note = (
        f"Receiver noise is unchanged by beamforming ({receiver_k:.0f} K), while external noise "
        f"goes from {external_element:.0f} K per element to {external_output:.0f} K at the output."
    )
    # How far noise can move: only man-made and galactic noise vary here (P.372-18 Part 7).
    measured = env.man_made_override_db is not None
    upper_dev, lower_dev = man_made_decile_deviations_db(env.man_made_category, include_location=not measured)
    if env.man_made_upper_decile_db is not None:
        upper_dev = env.man_made_upper_decile_db
    if env.man_made_lower_decile_db is not None:
        lower_dev = env.man_made_lower_decile_db
    measured_deviations = env.man_made_upper_decile_db is not None and env.man_made_lower_decile_db is not None
    statistics_note = ""
    if mm_used and f > itu.MAN_MADE_MAX_VALID_HZ and not measured_deviations:
        statistics_note = (
            "Table 2 is an average for 0.3 to 250 MHz, used here beyond its range. Older outdoor measurements at 322-425 MHz "
            "(P.372-13) show smaller deviations, so this margin may be pessimistic. Enter measured deviations if available."
        )
    t_man_made_random = t_mm_output * mismatch
    t_galactic_random = t_extraterrestrial_output * mismatch
    statistics = noise_statistics(
        t_man_made_random, t_galactic_random, system_output - t_man_made_random - t_galactic_random, upper_dev, lower_dev, statistics_note
    )
    design_k = statistics.system_upper_k if env.noise_design_point == "upper_decile" else system_output

    return NoiseResult(
        statistics=statistics,
        design_output_k=design_k,
        design_output_dbm_hz=float(db(K_BOLTZMANN * design_k) + 30.0),
        contributions=contributions,
        external_element_k=external_element,
        external_output_k=external_output,
        receiver_k=receiver_k,
        system_element_k=system_element,
        system_output_k=system_output,
        noise_element_dbm_hz=float(db(K_BOLTZMANN * system_element) + 30.0),
        noise_output_dbm_hz=float(db(K_BOLTZMANN * system_output) + 30.0),
        dominant_source=max(sources, key=sources.get),
        array_noise_gain_note=note,
    )
