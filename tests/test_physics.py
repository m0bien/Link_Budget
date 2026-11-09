"""Reference-value tests: each model against a published number or a known limit."""

from __future__ import annotations

import numpy as np
import pytest

from bistatic_lb.antenna.arrays import array_from_scenario
from bistatic_lb.antenna.elements import DIPOLE_DIRECTIVITY, MonopoleTx, mismatch_loss_db
from bistatic_lb.config import Scenario
from bistatic_lb.frontend.adc import ideal_sqnr_db
from bistatic_lb.noise.itu_p372 import galactic_fam_db, man_made_fam_db
from bistatic_lb.noise.receiver import build_lineup
from bistatic_lb.propagation.budget import free_space_loss_db
from bistatic_lb.propagation.clutter_loss import terrestrial_clutter_loss_db
from bistatic_lb.propagation.diffraction import (
    _distance_term_db,
    bullington_loss_db,
    knife_edge_exact_db,
    knife_edge_j_db,
    spherical_earth_loss_db,
)
from bistatic_lb.propagation.multipath import clear_air_fade_depth_db, two_ray
from bistatic_lb.propagation.rain import rain_attenuation_db
from bistatic_lb.propagation.refraction import k_factor
from bistatic_lb.propagation.surface import GROUND_TYPES
from bistatic_lb.target.detection import (
    FLUCTUATION_MODELS,
    Fluctuation,
    albersheim_snr_db,
    cfar_loss_db,
    detection_probability,
    required_snr_db,
    shnidman_snr_db,
    threshold,
)
from bistatic_lb.target.estimation import delay_crlb_s, doppler_crlb_hz, position_covariance
from bistatic_lb.target.montecarlo import (
    angle_estimator_rmse,
    delay_estimator_rmse,
    detection_monte_carlo,
    doppler_estimator_rmse,
)
from bistatic_lb.tracker.bounds import constant_velocity_model, pcrlb_series, steady_state_covariance
from bistatic_lb.tracker.initiation import confirm_probability, scans_to_confirm
from bistatic_lb.units import undb


# --- Noise -----------------------------------------------------------------------------
def test_p372_man_made_table_values():
    assert man_made_fam_db(100e6, "city") == pytest.approx(76.8 - 27.7 * 2)
    assert man_made_fam_db(100e6, "quiet_rural") == pytest.approx(53.6 - 28.6 * 2)
    assert galactic_fam_db(100e6) == pytest.approx(52.0 - 46.0)


def test_friis_cascade_matches_hand_calculation():
    s = Scenario()
    s.frontend.feed_loss_db = s.frontend.limiter_loss_db = s.frontend.rf_filter_loss_db = 0.0
    s.frontend.lna_gain_db, s.frontend.lna_nf_db = 20.0, 1.0
    s.frontend.mixer_loss_db = 10.0
    s.frontend.if_gain_db, s.frontend.if_nf_db = 20.0, 5.0
    nf = build_lineup(s.frontend).noise_figure_db
    f = 10**0.1 + (10 - 1) / 100 + (10**0.5 - 1) / (100 * 0.1)
    assert nf == pytest.approx(10 * np.log10(f), abs=1e-6)


# --- Antennas ----------------------------------------------------------------------------
def test_dipole_and_monopole_gain():
    assert 10 * np.log10(DIPOLE_DIRECTIVITY) == pytest.approx(2.15, abs=0.01)
    assert MonopoleTx().gain_dbi(0.0) == pytest.approx(5.15, abs=0.02)


def test_folded_dipole_is_wider_than_dipole():
    assert mismatch_loss_db(660e6, 600e6, "folded_dipole") < mismatch_loss_db(660e6, 600e6, "dipole")
    assert mismatch_loss_db(600e6, 600e6, "dipole") == pytest.approx(0.0)


def test_single_element_array_is_the_element():
    s = Scenario()
    s.array.n_circular = 1
    arr = array_from_scenario(s)
    w = arr.beamformer(0.3, 0.0)
    assert arr.gain_dbi(w, 0.3, 0.0) == pytest.approx(2.15, abs=0.01)


def test_ring_array_gain_in_uniform_temperature():
    s = Scenario()
    s.array.n_circular, s.array.radius_wavelengths = 8, 1.3
    arr = array_from_scenario(s)
    w = arr.beamformer(0.0, 0.0)
    # With unit-norm weights and spacing beyond half a wavelength, integrated gain stays near one.
    assert arr.integrate_over_sphere(w, lambda el: np.ones_like(el)) == pytest.approx(1.0, abs=0.2)


def test_ring_factor_matches_bessel_form():
    from scipy.special import j0

    s = Scenario()
    s.array.n_circular, s.array.radius_wavelengths = 64, 3.0
    arr = array_from_scenario(s)
    w = arr.beamformer(0.0, 0.0)
    psi = np.deg2rad(20.0)
    argument = 2 * np.pi * 3.0 * np.abs(2 * np.sin(psi / 2))
    assert arr.gain_linear(w, psi, 0.0) / arr.gain_linear(w, 0.0, 0.0) == pytest.approx(j0(argument) ** 2, abs=1e-3)


def test_angle_crlb_matches_monte_carlo():
    s = Scenario()
    s.array.n_circular, s.array.radius_wavelengths = 8, 0.65
    arr = array_from_scenario(s)
    snr_db = 15.0
    bound, _ = arr.angle_crlb_rad(float(undb(snr_db)), 0.5, 0.0)
    measured = angle_estimator_rmse(arr, 0.5, 0.0, [snr_db], trials=400)[0]
    assert measured == pytest.approx(bound, rel=0.2)


def test_digital_null_is_limited_by_calibration():
    s = Scenario()
    arr = array_from_scenario(s)
    w = arr.beamformer(np.deg2rad(60), 0.0, null_dir=(0.0, 0.0))
    assert arr.gain_linear(w, 0.0, 0.0) < 1e-20
    with_error = arr.gain_with_errors_linear(w, 0.0, 0.0, 2.0)
    assert 1e-5 < with_error < 1e-1


# --- Propagation -------------------------------------------------------------------------
def test_free_space_loss():
    # 32.45 + 20 log f(MHz) + 20 log d(km)
    assert free_space_loss_db(10.0, 1e9) == pytest.approx(32.45 + 60 + 20, abs=0.01)


def test_k_factor_standard_atmosphere():
    assert k_factor(-40.0) == pytest.approx(1.342, abs=0.01)


def test_p526_distance_term_is_continuous():
    assert _distance_term_db(1.6 - 1e-9) == pytest.approx(_distance_term_db(1.6), abs=0.02)


def test_knife_edge_grazing_and_exact_agreement():
    assert knife_edge_exact_db(0.0) == pytest.approx(6.02, abs=0.01)
    nu = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    assert np.all(np.abs(knife_edge_j_db(nu) - knife_edge_exact_db(nu)) < 0.5)


def test_tall_mast_has_no_diffraction():
    r = spherical_earth_loss_db(20.0, 500.0, 500.0, 1e9, 8500.0, GROUND_TYPES["medium"])
    assert r.loss_db == 0.0 and r.line_of_sight


def test_diffraction_grows_beyond_the_horizon():
    medium = GROUND_TYPES["medium"]
    near = spherical_earth_loss_db(60.0, 30.0, 15.0, 600e6, 8500.0, medium)
    far = spherical_earth_loss_db(120.0, 30.0, 15.0, 600e6, 8500.0, medium)
    assert far.loss_db > near.loss_db > 10.0


def test_bullington_single_obstacle_uses_the_knife_edge():
    d = np.array([0.0, 10.0, 20.0])
    h = np.array([0.0, 100.0, 0.0])
    loss = bullington_loss_db(d, h, 30.0, 15.0, 600e6, 1e9)  # nearly flat Earth
    assert loss > 15.0


def test_two_ray_limits():
    # Grazing over smooth ground: reflection near -1, so deep lobing is possible.
    r = two_ray(20.0, 30.0, 15.0, 600e6, 8500.0, GROUND_TYPES["sea"], 0.0)
    assert r.reflection_magnitude > 0.5
    assert r.maximum_db <= 6.03
    # Very rough ground removes the specular ray.
    rough = two_ray(5.0, 100.0, 100.0, 10e9, 8500.0, GROUND_TYPES["medium"], 5.0)
    assert rough.reflection_magnitude < 0.01


def test_p530_fade_rises_with_distance_and_rarity():
    a = clear_air_fade_depth_db(30.0, 2.0, 50.0, 40.0, -6.0, 30.0, 0.01)
    b = clear_air_fade_depth_db(60.0, 2.0, 50.0, 40.0, -6.0, 30.0, 0.01)
    c = clear_air_fade_depth_db(30.0, 2.0, 50.0, 40.0, -6.0, 30.0, 0.001)
    assert b > a and c == pytest.approx(a + 10.0, abs=1e-6) or c > a


def test_p530_rain_scaling_and_p838():
    a = rain_attenuation_db(10e9, 50.0, 10.0, 0.01)
    assert a > 5.0
    assert rain_attenuation_db(10e9, 50.0, 10.0, 1.0) < a
    assert rain_attenuation_db(500e6, 50.0, 10.0, 0.01) == 0.0


def test_p2108_median_is_bounded():
    value = terrestrial_clutter_loss_db(2.0, 10.0)
    assert value == terrestrial_clutter_loss_db(2.0, 2.0)  # limited by the 2 km value
    assert 15.0 < value < 45.0


# --- Detection ---------------------------------------------------------------------------
def test_threshold_is_exact_for_one_look():
    assert threshold(1e-6, 1) == pytest.approx(np.log(1e6))


def test_published_required_snr():
    """Well known: Pd 0.9, Pfa 1e-6, one look. Steady about 13.2 dB, Swerling 1 about 21 dB."""
    assert required_snr_db(0.9, 1e-6, 1, FLUCTUATION_MODELS["swerling0"]) == pytest.approx(13.2, abs=0.2)
    assert required_snr_db(0.9, 1e-6, 1, FLUCTUATION_MODELS["swerling1"]) == pytest.approx(21.0, abs=0.2)


@pytest.mark.parametrize("n", [1, 10])
def test_albersheim_agrees_with_exact(n):
    exact = required_snr_db(0.9, 1e-6, n, FLUCTUATION_MODELS["swerling0"])
    assert exact == pytest.approx(albersheim_snr_db(0.9, 1e-6, n), abs=0.3)


@pytest.mark.parametrize("sw", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("n", [1, 10])
def test_shnidman_agrees_with_exact(sw, n):
    exact = required_snr_db(0.8, 1e-6, n, FLUCTUATION_MODELS[f"swerling{sw}"])
    assert exact == pytest.approx(shnidman_snr_db(0.8, 1e-6, n, sw), abs=1.0)


@pytest.mark.parametrize(
    "model",
    [
        FLUCTUATION_MODELS["swerling0"],
        FLUCTUATION_MODELS["swerling1"],
        FLUCTUATION_MODELS["swerling2"],
        FLUCTUATION_MODELS["swerling3"],
        FLUCTUATION_MODELS["swerling4"],
        Fluctuation("chi2", "scan", 1.5),
        Fluctuation("rician", "rician", rician_k=4.0),
    ],
    ids=lambda m: m.name,
)
@pytest.mark.parametrize("n", [1, 4])
def test_pd_and_pfa_match_monte_carlo(model, n):
    snr_db, pfa = 12.0, 1e-2
    analytic = detection_probability(float(undb(snr_db)), pfa, n, model)
    pd_mc, pfa_mc = detection_monte_carlo(snr_db, pfa, n, model, trials=200_000)
    assert pd_mc == pytest.approx(analytic, abs=0.005)
    assert pfa_mc == pytest.approx(pfa, abs=0.002)


def test_cfar_loss_falls_with_more_reference_cells():
    assert cfar_loss_db(0.9, 1e-6, 16) > cfar_loss_db(0.9, 1e-6, 64) > 0.0
    assert cfar_loss_db(0.9, 1e-6, 0) == 0.0


# --- Estimation --------------------------------------------------------------------------
def test_delay_bound_vs_monte_carlo():
    snr_db = 20.0
    bound = delay_crlb_s(float(undb(snr_db)), 1.0 / np.sqrt(12.0))
    measured = delay_estimator_rmse([snr_db], trials=500)[0]
    assert measured == pytest.approx(bound, rel=0.2)


def test_delay_estimator_breaks_down_at_low_snr():
    low = 2.0
    bound = delay_crlb_s(float(undb(low)), 1.0 / np.sqrt(12.0))
    assert delay_estimator_rmse([low], trials=200)[0] > 3.0 * bound


def test_doppler_bound_vs_monte_carlo():
    snr_db = 20.0
    n = 512
    bound = doppler_crlb_hz(float(undb(snr_db)), n / np.sqrt(12.0))
    measured = doppler_estimator_rmse([snr_db], n_samples=n, trials=500)[0]
    assert measured == pytest.approx(bound, rel=0.2)


def test_position_degrades_toward_the_baseline():
    tx, rx = np.array([20e3, 0, 30.0]), np.array([0, 0, 15.0])
    away = position_covariance(np.array([0.0, 30e3, 1e3]), tx, rx, 50.0, 0.01, None)
    near_baseline = position_covariance(np.array([10e3, 300.0, 200.0]), tx, rx, 50.0, 0.01, None)
    assert np.trace(near_baseline) > np.trace(away)


# --- Tracker -----------------------------------------------------------------------------
def test_mofn_binomial_and_markov():
    assert confirm_probability(0.5, 2, 3) == pytest.approx(0.5)
    mean, by_n = scans_to_confirm(0.5, 2, 3)
    assert by_n == pytest.approx(0.5)
    assert mean > 2.0
    # Certain detection confirms in exactly m scans.
    assert scans_to_confirm(1.0, 3, 4)[0] == pytest.approx(3.0)


def test_mofn_markov_vs_simulation():
    rng = np.random.default_rng(5)
    p, m, n = 0.6, 3, 4
    trials = rng.random((40_000, 300)) < p
    counts = np.apply_along_axis(lambda a: np.convolve(a, np.ones(n), "full")[: len(a)], 1, trials)
    first = np.argmax(counts >= m, axis=1) + 1
    assert first.mean() == pytest.approx(scans_to_confirm(p, m, n)[0], rel=0.02)


def test_kalman_steady_state_reaches_the_pcrlb():
    f, q = constant_velocity_model(1.0, 2.0)
    h, r = np.eye(2), np.diag([100.0**2, 5.0**2])
    _, updated = steady_state_covariance(f, q, h, r)
    bound = pcrlb_series(f, q, h, r, 1.0, r * 4.0, 200)[-1]
    assert bound[0, 0] == pytest.approx(updated[0, 0], rel=1e-4)
    assert updated[0, 0] < r[0, 0]


# --- P.372-18 details that matter across this tool's frequency range ---------------------
def test_man_made_is_continuous_and_flattens_above_250_mhz():
    at_limit = man_made_fam_db(250e6, "city")
    assert man_made_fam_db(250.001e6, "city") == pytest.approx(at_limit, abs=0.01)
    assert man_made_fam_db(250e6, "city") - man_made_fam_db(2500e6, "city") == pytest.approx(11.5)


def test_man_made_above_250_mhz_against_published_values():
    # Fig. 3 curve A reads about 7.5 dB at 900 MHz.
    assert man_made_fam_db(900e6, "city") == pytest.approx(7.5, abs=4.0)
    # Outdoor measurements at 425 MHz, P.372-13 Table 3: city 6 dB, residential 4 dB.
    assert man_made_fam_db(425e6, "city") == pytest.approx(6.0, abs=2.5)
    assert man_made_fam_db(425e6, "residential") == pytest.approx(4.0, abs=2.5)


def test_quiet_rural_is_rejected_above_its_30_mhz_curve():
    s = Scenario()
    s.frequency_hz = 100e6
    s.environment.man_made_category = "quiet_rural"
    from bistatic_lb import run_link_budget

    item = next(c for c in run_link_budget(s).noise.contributions if c.name == "Man-made noise")
    assert not item.applicable and item.value == 0.0 and "30 MHz" in item.reason


def test_galactic_law_follows_fig_3_curve_b():
    # Read from Fig. 3 curve B: about -8 dB at 408 MHz and -15.5 dB at 1 GHz.
    assert galactic_fam_db(408e6) == pytest.approx(-8.0, abs=1.5)
    assert galactic_fam_db(1e9) == pytest.approx(-15.5, abs=2.5)


def test_quiet_sun_follows_fig_3_curve_d():
    from bistatic_lb.noise.itu_p372 import quiet_sun_brightness_k

    # P.372-18 text: about 1e6 K at 50 to 200 MHz, and at least 1e4 K at 10 GHz.
    assert quiet_sun_brightness_k(100e6) == pytest.approx(7.3e5, rel=0.1)
    assert quiet_sun_brightness_k(10e9) == pytest.approx(1e4, rel=0.1)


# --- P.372-18 Part 7: noise spread ---------------------------------------------------------
def _two_piece(rng, median_k, upper_db, lower_db, n):
    z = rng.standard_normal(n)
    return median_k * 10 ** (np.where(z > 0, z * upper_db / 1.282, z * lower_db / 1.282) / 10)


def test_part7_reduces_to_simple_cases():
    from bistatic_lb.noise.statistics import combine_noise_figures

    # One source: its own median and sigma.
    median, sigma = combine_noise_figures([10.0], [5.0])
    assert median == pytest.approx(10.0) and sigma == pytest.approx(5.0)
    # No variation: a plain power sum.
    median, sigma = combine_noise_figures([10.0, 10.0], [0.0, 0.0])
    assert median == pytest.approx(13.01, abs=0.01) and sigma == pytest.approx(0.0, abs=1e-6)


def test_part7_decile_restriction_applies_above_12_db():
    from bistatic_lb.noise.statistics import C_DB, combine_noise_figures

    medians, sigmas = [10.0, 5.0], [20.0 / 1.282, 2.0]
    _, restricted = combine_noise_figures(medians, sigmas)
    alpha = np.exp(np.array(medians) / C_DB + np.array(sigmas) ** 2 / (2 * C_DB**2))
    gamma = np.sum(np.exp(np.array(medians) / C_DB))
    assert restricted <= C_DB * np.sqrt(2 * np.log(alpha.sum() / gamma)) + 1e-9


def test_table_2_time_and_location_add_in_power():
    from bistatic_lb.noise.statistics import man_made_decile_deviations_db

    assert man_made_decile_deviations_db("rural")[0] == pytest.approx(np.hypot(9.2, 6.8))
    assert man_made_decile_deviations_db("rural", include_location=False) == (9.2, 4.6)
    assert man_made_decile_deviations_db("quiet_rural") == (0.0, 0.0)


@pytest.mark.parametrize("man_made_k", [100.0, 1000.0, 10000.0])
@pytest.mark.parametrize("category", ["rural", "city"])
def test_part7_upper_decile_against_simulation(man_made_k, category):
    from bistatic_lb.noise.statistics import man_made_decile_deviations_db, noise_statistics

    rng = np.random.default_rng(3)
    up, low = man_made_decile_deviations_db(category)
    stats = noise_statistics(man_made_k, 50.0, 400.0, up, low)
    n = 400_000
    total = _two_piece(rng, man_made_k, up, low, n) + _two_piece(rng, 50.0, 2.0, 2.0, n) + 400.0
    assert 10 * np.log10(stats.system_upper_k / np.quantile(total, 0.9)) == pytest.approx(0.0, abs=1.5)
    assert 10 * np.log10(stats.system_median_k / np.median(total)) == pytest.approx(0.0, abs=1.0)


def test_noise_cannot_fall_below_the_constant_part():
    from bistatic_lb.noise.statistics import noise_statistics

    stats = noise_statistics(100.0, 15.0, 600.0, 11.4, 8.1)
    assert stats.system_lower_k >= 600.0
    assert stats.system_lower_k <= stats.system_median_k <= stats.system_upper_k


def test_upper_decile_design_point_flows_through_the_link():
    from bistatic_lb import run_link_budget

    median = Scenario()
    median.frequency_hz = 100e6
    median.environment.man_made_category = "city"
    median.array.n_circular, median.array.radius_wavelengths = 8, 0.5
    upper = median.with_value("environment.noise_design_point", "upper_decile")
    a, b = run_link_budget(median), run_link_budget(upper)
    assert a.noise.statistics.upper_margin_db > 5.0
    assert b.noise.design_output_k == pytest.approx(a.noise.statistics.system_upper_k)
    assert b.link.snr_db < a.link.snr_db
    assert b.detection.pd == pytest.approx(a.detection.pd_upper_decile, abs=0.02)
    # The front-end keeps using median noise, so changing the design point leaves the gain window alone.
    assert b.frontend.gain_window_db == pytest.approx(a.frontend.gain_window_db)


def test_measured_deviations_replace_table_2():
    from bistatic_lb import run_link_budget

    s = Scenario()
    s.environment.man_made_upper_decile_db = 2.0
    s.environment.man_made_lower_decile_db = 2.0
    stats = run_link_budget(s).noise.statistics
    assert stats.man_made_upper_deviation_db == 2.0 and stats.note == ""


def test_migration_loss_limits_and_numerical_check():
    from bistatic_lb.target.migration import doppler_walk_loss_db, range_walk_loss_db

    assert range_walk_loss_db(0.0) == 0.0 and doppler_walk_loss_db(0.0) == 0.0
    # Far beyond one cell, only the time spent in the cell integrates: amplitude 1/M.
    assert range_walk_loss_db(40.0) == pytest.approx(20.0 * np.log10(40.0), abs=0.3)
    # Against direct numerical integration of the two responses.
    for cells in (0.5, 1.0, 3.0):
        x = np.linspace(-cells / 2.0, cells / 2.0, 20001)
        assert range_walk_loss_db(cells) == pytest.approx(-20.0 * np.log10(np.mean(np.sinc(x))), abs=1e-3)
        u = np.linspace(-0.5, 0.5, 20001)
        assert doppler_walk_loss_db(cells) == pytest.approx(-20.0 * np.log10(abs(np.mean(np.exp(1j * np.pi * cells * u**2)))), abs=1e-3)


def test_range_acceleration_matches_a_finite_difference():
    from bistatic_lb.geometry import build_geometry

    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml")
    g = build_geometry(s)
    dt = 0.01
    ahead = build_geometry(s, g.target + g.velocity * dt)
    behind = build_geometry(s, g.target - g.velocity * dt)
    numeric = (ahead.bistatic_range_rate_m_s - behind.bistatic_range_rate_m_s) / (2.0 * dt)
    assert g.bistatic_range_accel_m_s2 == pytest.approx(numeric, rel=1e-4)
