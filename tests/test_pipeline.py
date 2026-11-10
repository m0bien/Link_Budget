"""End-to-end behaviour and physical limit checks on the whole budget."""

from __future__ import annotations

import numpy as np
import pytest

from bistatic_lb import Scenario, run_link_budget
from bistatic_lb.waveforms import WAVEFORMS, make_waveform
from bistatic_lb.config import WaveformConfig
from bistatic_lb.tracker.montecarlo import run_tracker_monte_carlo


@pytest.fixture(scope="module")
def base():
    return run_link_budget(Scenario())


@pytest.mark.parametrize("kind", list(WAVEFORMS))
def test_every_waveform_runs_end_to_end(kind):
    s = Scenario()
    s.waveform.kind = kind
    r = run_link_budget(s)
    assert np.isfinite(r.link.sinr_db)
    assert 0.0 <= r.detection.pd <= 1.0


def test_defaults_give_reasonable_values(base):
    assert 200 < base.noise.system_output_k < 2000
    assert base.direct.element_maximum_dbm >= base.direct.element_nominal_dbm >= base.direct.element_minimum_dbm
    assert base.link.sinr_db <= base.link.snr_db + 1e-9


def test_monostatic_limit_of_radar_equation():
    """Beta near zero: Rt = Rr = R, so doubling R loses 12 dB in free space."""
    def snr(range_km):
        s = Scenario()
        s.geometry.baseline_km = 0.01
        s.geometry.target_range_rx_km = range_km
        s.geometry.target_altitude_m = 20000.0
        s.geometry.target_heading_deg = s.geometry.target_azimuth_deg  # radial flight: off zero Doppler, clear of the canceller notch
        s.environment.radar_clutter_enabled = False
        s.array.null_on_transmitter = False
        return run_link_budget(s).link.snr_db

    # Elevation patterns change slightly, so allow a small tolerance.
    assert snr(100.0) - snr(200.0) == pytest.approx(12.04, abs=1.5)


def test_tall_masts_reach_free_space():
    s = Scenario()
    s.transmitter.height_m = 500.0
    s.array.height_m = 500.0
    s.environment.roughness_rms_m = 50.0
    s.frequency_hz = 2e9
    r = run_link_budget(s)
    assert r.direct.terms.diffraction_db == 0.0
    assert r.direct.terms.nominal_loss_db == pytest.approx(r.direct.terms.free_space_db, abs=1.5)


def test_short_adc_gives_shortfall_and_analog_suppression_helps():
    s = Scenario()
    s.frontend.gain_control = "continuous"  # gated gain would let the pulsed default avoid the shortfall
    s.frontend.adc_bits = 6
    s.frontend.adc_enob_loss_bits = 1.0
    r = run_link_budget(s)
    assert r.frontend.shortfall_db > 0.0
    s.frontend.analog_canceller_db = 60.0
    fixed = run_link_budget(s)
    assert fixed.frontend.shortfall_db == pytest.approx(0.0) or fixed.frontend.shortfall_db < r.frontend.shortfall_db


def test_digital_null_does_not_widen_the_gain_window():
    on = Scenario()
    off = Scenario()
    off.array.null_on_transmitter = False
    a, b = run_link_budget(on), run_link_budget(off)
    assert a.frontend.gain_max_db == pytest.approx(b.frontend.gain_max_db, abs=1e-6)
    assert a.direct.output_nominal_dbm < b.direct.output_nominal_dbm


def test_analog_subarray_widens_the_gain_window():
    digital = Scenario()
    analog = Scenario()
    analog.array.digitization = "analog_subarray"
    a, b = run_link_budget(digital), run_link_budget(analog)
    assert b.frontend.gain_window_db == pytest.approx(a.frontend.gain_window_db + analog.array.analog_null_depth_db, abs=1e-6)


def test_more_elements_improve_gain_and_angle():
    small, big = Scenario(), Scenario()
    big.array.n_circular, big.array.radius_wavelengths = 16, 1.3
    a, b = run_link_budget(small), run_link_budget(big)
    assert b.antenna.gain_toward_target_dbi > a.antenna.gain_toward_target_dbi
    assert b.estimation.sigma_azimuth_deg < a.estimation.sigma_azimuth_deg


def test_vertical_column_gives_elevation():
    s = Scenario()
    s.array.n_vertical = 6
    s.geometry.target_altitude_m = 5000.0
    r = run_link_budget(s)
    assert r.estimation.sigma_elevation_deg is not None
    assert r.estimation.sigma_z_m is not None


def test_tracker_improves_accuracy_and_reduces_required_snr(base):
    assert base.tracker.tracked_sigma_rate_m_s > 0
    assert base.tracker.range_improvement_db > 0
    assert base.tracker.trade.gain_db > 0
    assert base.tracker.false_tracks_per_hour <= base.scenario.tracker.allowed_false_tracks_per_hour


def test_tracker_monte_carlo_beats_single_look_and_respects_bound():
    s = Scenario()
    s.monte_carlo.runs, s.monte_carlo.scans = 60, 40
    s.monte_carlo.initiation = "on_truth"  # filter accuracy alone; initiation has its own test
    s.target.fluctuation = "swerling0"
    s.target.rcs_dbsm = 25.0
    r = run_link_budget(s)
    mc = run_tracker_monte_carlo(r.analyzer, r.detection.pfa)
    assert mc.continuity > 0.8
    tail = -15
    assert np.mean(mc.rmse_range_m[tail:]) < np.mean(mc.single_look_range_m[tail:])
    assert np.mean(mc.rmse_position_m[tail:]) < np.mean(mc.single_look_position_m[tail:])
    # An efficient filter should sit near the bound, not far below it.
    assert np.mean(mc.rmse_range_m[tail:]) > 0.7 * np.mean(mc.bound_range_m[tail:])


def test_yaml_round_trip():
    s = Scenario()
    s.array.n_circular = 12
    again = Scenario.from_yaml(s.to_yaml())
    assert again == s


def test_with_value_validates():
    s = Scenario().with_value("array.n_circular", 16)
    assert s.array.n_circular == 16
    with pytest.raises(KeyError):
        Scenario().with_value("array.no_such_field", 1)
    with pytest.raises(ValueError):
        Scenario().with_value("frequency_hz", 1e6)


# --- Waveform model values against generated signals --------------------------------------
@pytest.mark.parametrize("kind,fs", [("dab", 4.096e6), ("dvbt", 16e6)])
def test_ofdm_papr_and_bandwidth(kind, fs):
    wf = make_waveform(WaveformConfig(kind=kind))
    x = wf.generate(fs, 0.2, np.random.default_rng(1))
    power = np.abs(x) ** 2
    measured = 10 * np.log10(np.quantile(power, 1.0 - 1.3e-5) / power.mean())
    assert measured == pytest.approx(wf.papr_db, abs=0.7)
    spectrum = np.abs(np.fft.fft(x)) ** 2
    freqs = np.abs(np.fft.fftfreq(len(x), 1 / fs))
    occupied = 2 * freqs[spectrum > 0.05 * np.median(spectrum[freqs < wf.bandwidth_hz / 4])].max()
    assert occupied == pytest.approx(wf.bandwidth_hz, rel=0.1)


def test_constant_envelope_waveforms_have_no_papr():
    rng = np.random.default_rng(1)
    for kind in ("fm", "zadoff_chu"):
        wf = make_waveform(WaveformConfig(kind=kind, bandwidth_hz=1e6, pulse_width_s=1e-3))
        x = wf.generate(4e6, 0.02, rng)
        assert np.max(np.abs(x) ** 2) / np.mean(np.abs(x) ** 2) == pytest.approx(1.0, abs=1e-6)


def test_lfm_generated_bandwidth_and_duty():
    wf = make_waveform(WaveformConfig(kind="pulsed_lfm"))
    fs = 8e6
    x = wf.generate(fs, 0.01, np.random.default_rng(1))
    assert np.mean(np.abs(x) > 0) == pytest.approx(wf.duty_cycle, abs=0.01)
    pulse = x[: int(wf.cfg.pulse_width_s * fs)]
    freq = np.diff(np.unwrap(np.angle(pulse))) * fs / (2 * np.pi)
    assert freq.max() - freq.min() == pytest.approx(wf.bandwidth_hz, rel=0.05)


def test_zadoff_chu_periodic_autocorrelation_is_zero():
    wf = make_waveform(WaveformConfig(kind="zadoff_chu", bandwidth_hz=1e6, pulse_width_s=1.27e-4))
    code = wf.generate(1e6, 1.27e-4, np.random.default_rng(1))
    corr = np.fft.ifft(np.abs(np.fft.fft(code)) ** 2)
    assert np.max(np.abs(corr[1:])) / np.abs(corr[0]) < 1e-3


def test_random_code_floor_matches_time_bandwidth():
    wf = make_waveform(WaveformConfig(kind="phase_coded", bandwidth_hz=1e6, cpi_s=2e-3))
    code = wf.generate(1e6, 2e-3, np.random.default_rng(1))
    corr = np.abs(np.fft.ifft(np.abs(np.fft.fft(code, 2 * len(code))) ** 2)) ** 2
    floor_db = 10 * np.log10(np.mean(corr[10:-10]) / corr[0])
    # Half the lags are empty in this zero-padded correlation; allow for that.
    assert floor_db == pytest.approx(wf.direct_leakage_db(1e-3, 1e3), abs=4.0)


def test_noise_over_quantization_follows_from_full_scale_direct_path_and_noise():
    r = run_link_budget(Scenario())
    f = r.frontend
    expected = f.instantaneous_dynamic_range_db - f.clipping_backoff_db - f.direct_to_noise_at_adc_db
    assert f.noise_over_quantization_db == pytest.approx(expected, abs=1e-9)
    assert f.gain_window_db == pytest.approx(f.noise_over_quantization_db, abs=1e-9)
    # Raising full scale by 6 dB raises quantization by 6 dB too, so the ratio is unchanged.
    s = Scenario()
    s.frontend.adc_full_scale_dbm += 6.0
    assert run_link_budget(s).frontend.noise_over_quantization_db == pytest.approx(f.noise_over_quantization_db, abs=1e-9)
    # A stronger direct path takes dB for dB from the noise over quantization.
    s = Scenario()
    s.transmitter.power_w *= 10.0
    assert run_link_budget(s).frontend.noise_over_quantization_db == pytest.approx(f.noise_over_quantization_db - 10.0, abs=0.05)


def test_required_bits_spend_exactly_the_detection_margin():
    from bistatic_lb.frontend.adc import sensitivity_loss_db

    r = run_link_budget(Scenario())
    f = r.frontend
    assert f.allowed_quantization_loss_db == pytest.approx(r.detection.margin_db + f.sensitivity_loss_db, abs=1e-9)
    # With the required effective bits, the quantization loss at the maximum direct level equals the allowance.
    noq = f.noise_over_quantization_worst_db + 6.02 * (f.required_effective_bits - f.effective_bits)
    assert sensitivity_loss_db(noq, 0.0) == pytest.approx(f.allowed_quantization_loss_db, abs=1e-6)
    assert any(q.item == "ADC resolution" for q in f.requirements)


def test_required_bits_unreachable_when_the_link_fails_anyway():
    s = Scenario()
    s.target.rcs_dbsm = -40.0
    r = run_link_budget(s)
    assert r.detection.margin_db + r.frontend.sensitivity_loss_db < 0.0
    assert r.frontend.required_adc_bits == float("inf")
    row = next(q for q in r.frontend.requirements if q.item == "ADC resolution")
    assert row.met is False and row.required == "Not reachable"


def test_old_scenarios_with_a_noise_over_quantization_target_still_load():
    text = Scenario().to_yaml().replace("frontend:\n", "frontend:\n  noise_over_quantization_db: 6.0\n", 1)
    assert "noise_over_quantization_db" in text
    assert Scenario.from_yaml(text) == Scenario()


def test_lineup_levels_follow_gain_and_agc():
    from bistatic_lb.frontend.levels import levels_for

    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml")
    base = levels_for(run_link_budget(s))
    r = run_link_budget(s.with_value("frontend.lna_gain_db", s.frontend.lna_gain_db + 20.0))
    hot = levels_for(r)
    f = r.frontend
    # Before the AGC, noise and the direct path rise dB for dB with LNA gain.
    lna = [p for p in hot.points if p.name == "LNA"][0]
    lna0 = [p for p in base.points if p.name == "LNA"][0]
    assert lna.noise_dbm - lna0.noise_dbm == pytest.approx(20.0, abs=0.5)
    assert lna.direct_dbm - lna0.direct_dbm == pytest.approx(20.0)
    # The AGC brings the strongest direct signal back to the clipping limit, never past it.
    end = hot.points[-1]
    assert hot.agc_cut_db == pytest.approx(f.configured_gain_db - f.gain_max_db)
    assert end.direct_dbm == pytest.approx(hot.clipping_limit_dbm)
    assert hot.overshoot_without_agc_db == pytest.approx(hot.agc_cut_db)
    assert end.noise_dbm == pytest.approx(f.noise_at_input_dbm + f.gain_max_db)
    assert base.agc_cut_db == 0.0 and base.overshoot_without_agc_db < 0


def test_stage_compression_is_flagged_and_warned():
    from bistatic_lb.frontend.levels import levels_for

    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml").with_value("frontend.lna_input_p1db_dbm", -45.0)
    r = run_link_budget(s)
    names = [p.name for p in levels_for(r).compressing]
    assert names == ["LNA"]
    assert any("LNA compresses" in w for w in r.warnings)


def test_level_plots_draw():
    from bistatic_lb import plots

    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml")
    for scenario in (s, s.with_value("frontend.lna_gain_db", s.frontend.lna_gain_db + 20.0)):
        r = run_link_budget(scenario)
        assert plots.lineup_levels(r).axes and plots.dynamic_range_ladder(r).axes


def test_gated_gain_restores_noise_between_direct_pulses():
    s = Scenario.from_yaml("scenarios/pulsed_lfm_uhf.yaml").with_value("geometry.baseline_km", 2.0)
    cont = run_link_budget(s.with_value("frontend.gain_control", "continuous"))
    gated = run_link_budget(s)  # auto resolves to gated for a pulsed waveform
    fc, fg = cont.frontend, gated.frontend
    assert fc.gain_control == "continuous" and fg.gain_control == "gated"
    # With the direct pulse present both cut the gain the same way.
    assert fg.pulse_gain_db == pytest.approx(fc.pulse_gain_db)
    assert fc.pulse_gain_db < fc.configured_gain_db
    # Between pulses the gated receiver gets its configured gain back.
    assert fg.listening_gain_db == pytest.approx(fg.configured_gain_db)
    assert fg.noise_over_quantization_nominal_db == pytest.approx(fg.noise_at_input_dbm + fg.configured_gain_db - fg.quantization_dbm)
    assert fg.sensitivity_loss_db < fc.sensitivity_loss_db - 1.0
    assert fg.shortfall_db == 0.0 < fc.shortfall_db
    assert gated.detection.margin_db > cont.detection.margin_db


def test_gated_gain_is_ignored_for_continuous_waveforms():
    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml")
    r = run_link_budget(s.with_value("frontend.gain_control", "gated"))
    assert r.frontend.gain_control == "continuous" and r.frontend.listening_gain_db is None
    assert r.frontend.sensitivity_loss_db == pytest.approx(run_link_budget(s).frontend.sensitivity_loss_db)


def test_gain_switch_settling_eclipses_close_echoes():
    s = Scenario.from_yaml("scenarios/pulsed_lfm_uhf.yaml")
    wf = make_waveform(s.waveform)
    tau = s.waveform.pulse_width_s
    delay = 1.5 * tau  # echo starts half a pulse after the direct pulse ends
    assert wf.signal_loss_db(delay, 0.0) == pytest.approx(0.0)
    # Blind for 1 pulse width + 1.0 tau of settling: only half of the echo is heard.
    assert wf.signal_loss_db(delay, 0.0, blind_extra_s=tau) == pytest.approx(10 * np.log10(2.0), abs=1e-6)


def test_single_element_array_runs_and_reports_no_azimuth():
    r = run_link_budget(Scenario.from_yaml("scenarios/dvbt_uhf.yaml").with_value("array.n_circular", 1))
    assert np.isinf(r.estimation.sigma_azimuth_deg) and np.isinf(r.estimation.position_rms_m)
    assert np.isfinite(r.estimation.sigma_bistatic_range_m)
    assert any("cannot measure azimuth" in w for w in r.warnings)
    mc_scenario = r.scenario.with_value("monte_carlo.runs", 3).with_value("monte_carlo.scans", 6).with_value("monte_carlo.initiation", "on_truth")
    x = run_link_budget(mc_scenario)
    mc = run_tracker_monte_carlo(x.analyzer, x.detection.pfa)
    assert np.all(np.isfinite(mc.rmse_range_m))


def test_very_distant_target_runs():
    r = run_link_budget(Scenario.from_yaml("scenarios/dvbt_uhf.yaml").with_value("geometry.target_range_rx_km", 500.0))
    assert r.detection.pd < 1e-6


def test_direct_path_alone_matches_the_full_budget():
    from bistatic_lb.pipeline import direct_path_for

    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml").with_value("geometry.baseline_km", 7.0)
    full, alone = run_link_budget(s).direct, direct_path_for(s)
    for field in ("element_nominal_dbm", "element_maximum_dbm", "element_minimum_dbm", "output_nominal_dbm"):
        assert getattr(alone, field) == pytest.approx(getattr(full, field))


def test_gain_window_plot_formula_matches_a_full_run():
    from bistatic_lb import plots

    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml")
    fig = plots.gain_window_vs_bits(run_link_budget(s))
    bits, widths = fig.axes[0].lines[0].get_data()
    for b in (8, 12, 20):
        expected = run_link_budget(s.with_value("frontend.adc_bits", b)).frontend.gain_window_db
        assert widths[list(bits).index(b)] == pytest.approx(expected)


def test_update_interval_shorter_than_a_frame_is_warned():
    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml")
    assert not any("update interval" in w for w in run_link_budget(s).warnings)
    assert any("update interval" in w for w in run_link_budget(s.with_value("tracker.update_interval_s", 0.05)).warnings)


def test_a_long_coherent_interval_no_longer_gives_free_gain():
    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml")
    short = run_link_budget(s)
    long = run_link_budget(s.with_value("waveform.cpi_s", 10.0))
    assert long.link.migration.loss_db > 20.0
    assert long.detection.margin_db < short.detection.margin_db
    assert any("coherent interval" in w and "Keep the interval under" in w for w in long.warnings)
    # A processor that follows the target gets the integration gain back.
    followed = run_link_budget(s.with_value("waveform.cpi_s", 10.0).with_value("waveform.migration_compensation", "range_doppler"))
    assert followed.link.migration.loss_db == 0.0
    assert followed.detection.margin_db > short.detection.margin_db + 10.0


def test_fastest_design_target_is_warned_about():
    r = run_link_budget(Scenario.from_yaml("scenarios/dvbt_uhf.yaml"))
    assert any("design maximum speed" in w for w in r.warnings)
    compensated = run_link_budget(r.scenario.with_value("waveform.migration_compensation", "range"))
    assert not any("design maximum speed" in w for w in compensated.warnings)


def test_lo_phase_noise_of_the_direct_signal_reaches_the_target_cell():
    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml").with_value("array.null_on_transmitter", False)
    clean = run_link_budget(s)
    noisy = run_link_budget(s.with_value("frontend.lo_phase_noise_dbc_hz", s.frontend.lo_phase_noise_dbc_hz + 30.0))
    assert noisy.link.phase_noise_cell_dbm == pytest.approx(clean.link.phase_noise_cell_dbm + 30.0)
    assert noisy.link.sinr_db < clean.link.sinr_db - 5.0
    assert noisy.link.snr_db == pytest.approx(clean.link.snr_db)
    assert any("LO phase noise" in w for w in noisy.warnings)
    # A pulsed waveform separates the direct pulse in time.
    pulsed = run_link_budget(Scenario.from_yaml("scenarios/pulsed_lfm_uhf.yaml"))
    assert pulsed.link.phase_noise_cell_dbm <= -299.0


def test_third_order_distortion_applies_only_to_varying_envelopes():
    dvbt = Scenario.from_yaml("scenarios/dvbt_uhf.yaml").with_value("geometry.baseline_km", 2.0)
    strong = run_link_budget(dvbt)
    weak_lna = run_link_budget(dvbt.with_value("frontend.lna_iip3_dbm", dvbt.frontend.lna_iip3_dbm - 20.0))
    assert strong.frontend.direct_intermod_dbc is not None
    # 2 dB of distortion per dB of cascade intercept (later stages share the cascade value with the LNA).
    drop = strong.lineup.iip3_dbm - weak_lna.lineup.iip3_dbm
    assert drop > 1.0
    assert weak_lna.link.intermod_cell_dbm == pytest.approx(strong.link.intermod_cell_dbm + 2.0 * drop)
    assert weak_lna.link.sinr_db < strong.link.sinr_db
    fm = run_link_budget(Scenario.from_yaml("scenarios/fm_vhf.yaml"))
    assert fm.frontend.direct_intermod_dbc is None and fm.link.intermod_cell_dbm <= -299.0


def test_compression_marks_results_as_not_valid_unless_gated():
    dvbt = Scenario.from_yaml("scenarios/dvbt_uhf.yaml")
    assert run_link_budget(dvbt).valid
    compressed = run_link_budget(dvbt.with_value("frontend.lna_input_p1db_dbm", -45.0))
    assert not compressed.valid and "LNA compresses" in compressed.invalid_reasons[0]
    pulsed = Scenario.from_yaml("scenarios/pulsed_lfm_uhf.yaml").with_value("frontend.lna_input_p1db_dbm", -60.0)
    gated = run_link_budget(pulsed)
    assert gated.valid and any("LNA compresses" in w for w in gated.warnings)
    assert not run_link_budget(pulsed.with_value("frontend.gain_control", "continuous")).valid


def test_a_stationary_target_is_removed_by_the_canceller():
    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml")
    moving = run_link_budget(s)
    still = run_link_budget(s.with_value("geometry.target_speed_m_s", 0.0))
    assert moving.link.notch_loss_db < 0.5 and moving.detection.pd > 0.9
    assert still.link.notch_loss_db == pytest.approx(still.link.cancellation_applied_db, abs=0.5)
    assert still.detection.pd < 0.01
    assert any("canceller's notch" in w for w in still.warnings)


def test_minimum_detectable_range_rate():
    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml")
    r = run_link_budget(s)
    slowest = r.detection.min_detectable_range_rate_m_s
    assert 0.0 < slowest < abs(r.geometry.bistatic_range_rate_m_s)
    # At that range rate the link closes; at half of it, it does not.
    required = r.detection.required_snr_db
    lam = r.analyzer.lam
    assert r.analyzer.link(doppler_hz=slowest / lam).sinr_db >= required
    assert r.analyzer.link(doppler_hz=0.5 * slowest / lam).sinr_db < required
    # A target too weak at any Doppler has none.
    assert np.isinf(run_link_budget(s.with_value("target.rcs_dbsm", -40.0)).detection.min_detectable_range_rate_m_s)


def test_clutter_ring_feeds_the_link_and_can_be_switched_off():
    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml")
    on = run_link_budget(s).link
    assert -260.0 < on.clutter_zero_doppler_dbm < -100.0
    assert on.clutter_cell_dbm < on.clutter_zero_doppler_dbm
    off = run_link_budget(s.with_value("environment.radar_clutter_enabled", False)).link
    assert off.clutter_zero_doppler_dbm <= -299.0 and off.clutter_cell_dbm <= -299.0
    # A target with a short extra path shares its range cell with ground close to the sites.
    near = run_link_budget(s.with_value("geometry.target_range_rx_km", 6.0).with_value("geometry.target_azimuth_deg", 20.0)).link
    assert near.clutter_zero_doppler_dbm > on.clutter_zero_doppler_dbm + 10.0


def test_actual_ground_reflection_phase_shows_the_lobing():
    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml")
    average = run_link_budget(s)
    actual = run_link_budget(s.with_value("environment.target_multipath", "actual"))
    extra = average.link.lobing_extra_loss_db
    assert abs(extra) > 0.5
    assert actual.link.received_dbm == pytest.approx(average.link.received_dbm - extra, abs=1e-6)
    assert any("near a null" in w for w in average.warnings) == (extra >= 3.0)
    assert not any("near a null" in w for w in actual.warnings)


def test_tracks_started_from_detections_confirm_as_predicted():
    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml").with_value("monte_carlo.runs", 150).with_value("monte_carlo.scans", 40)
    r = run_link_budget(s)
    mc = run_tracker_monte_carlo(r.analyzer, r.tracker.trade.best_pfa)
    assert mc.initiation == "from_detections" and mc.confirmed_fraction > 0.98
    # Simulated time to confirm agrees with the Markov-chain prediction.
    assert mc.mean_scans_to_confirm == pytest.approx(r.tracker.mean_scans_to_confirm, abs=0.25)
    # No confirmed track exists before M scans have passed.
    assert np.all(np.isnan(mc.rmse_range_m[: s.tracker.confirm_m - 1]))
    assert mc.tracked_runs[-1] >= 0.9 * mc.runs
    tail = -12
    assert np.nanmean(mc.rmse_range_m[tail:]) < np.mean(mc.single_look_range_m[tail:])
    # Starting from one detection, with the velocity unknown, the position still converges below single look.
    assert np.nanmean(mc.rmse_position_m[tail:]) < np.mean(mc.single_look_position_m[tail:])
    best = run_tracker_monte_carlo(run_link_budget(s.with_value("monte_carlo.initiation", "on_truth")).analyzer, r.tracker.trade.best_pfa)
    assert best.mean_scans_to_confirm == 1.0
    # Compared once both have confirmed tracks: the unknown initial velocity costs accuracy early on.
    assert np.nanmean(best.rmse_position_m[3:10]) < np.nanmean(mc.rmse_position_m[3:10])


def test_a_hard_turn_loses_a_filter_tuned_for_gentle_targets():
    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml").with_value("monte_carlo.runs", 40).with_value("monte_carlo.scans", 40)
    r = run_link_budget(s)
    gentle = run_tracker_monte_carlo(run_link_budget(s.with_value("monte_carlo.manoeuvre_accel_m_s2", s.tracker.process_accel_m_s2)).analyzer, r.tracker.trade.best_pfa)
    hard = run_tracker_monte_carlo(run_link_budget(s.with_value("monte_carlo.manoeuvre_accel_m_s2", 10.0)).analyzer, r.tracker.trade.best_pfa)
    assert gentle.continuity > 0.9
    assert hard.continuity < 0.5


def test_altitude_uncertainty_adds_to_a_2d_position_fix_only():
    from bistatic_lb.target.estimation import altitude_sensitivity

    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml").with_value("geometry.target_altitude_m", 8000.0)
    exact = run_link_budget(s).estimation
    unsure = run_link_budget(s.with_value("geometry.target_altitude_sigma_m", 1000.0)).estimation
    shift = exact.position_shift_per_m_altitude
    assert unsure.position_rms_m**2 == pytest.approx(exact.position_rms_m**2 + (1000.0 * shift) ** 2, rel=1e-6)
    # The sensitivity is the real displacement of the range-azimuth fix when the altitude is wrong.
    from bistatic_lb.geometry import build_geometry
    from bistatic_lb.tracker.montecarlo import position_from_measurement

    g = build_geometry(s)
    wrong = position_from_measurement(g.tx, g.rx, g.target[2] + 50.0, g.baseline_m, g.bistatic_range_m, g.azimuth_at_rx_rad)
    assert np.linalg.norm(wrong - g.target[:2]) == pytest.approx(50.0 * shift, rel=0.05)
    assert np.linalg.norm(altitude_sensitivity(g.target, g.tx, g.rx)) == pytest.approx(shift)
    # An array that measures elevation needs no assumed altitude.
    three_d = run_link_budget(s.with_value("array.n_vertical", 4).with_value("geometry.target_altitude_sigma_m", 1000.0)).estimation
    assert three_d.altitude_sigma_m is None and three_d.position_shift_per_m_altitude is None


def test_canceller_reference_fidelity_sets_a_floor_the_depth_cannot_remove():
    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml").with_value("array.null_on_transmitter", False)
    good = run_link_budget(s)
    poor = run_link_budget(s.with_value("transmitter.signal_fidelity_db", s.transmitter.signal_fidelity_db - 10.0))
    assert good.analyzer.canceller.reference == "known_waveform"
    assert poor.link.fidelity_cell_dbm == pytest.approx(good.link.fidelity_cell_dbm + 10.0)
    assert poor.link.sinr_db < good.link.sinr_db - 3.0
    # More cancellation depth does not help: the reference does not contain this part.
    deeper = run_link_budget(poor.scenario.with_value("frontend.digital_canceller_limit_db", 90.0))
    assert deeper.link.fidelity_cell_dbm == pytest.approx(poor.link.fidelity_cell_dbm)
    assert any("reference lacks" in w for w in poor.warnings)
    # The floor is P_direct - fidelity - 10 log10(B T), at the beam output.
    wf, c = good.waveform, good.analyzer.canceller
    direct_out = good.link.phase_noise_cell_dbm - s.frontend.lo_phase_noise_dbc_hz + 10 * np.log10(wf.integration_time_s)
    assert good.link.fidelity_cell_dbm == pytest.approx(direct_out - c.fidelity_db - 10 * np.log10(wf.bandwidth_hz * wf.integration_time_s))


def test_canceller_reference_choice_and_pulsed_exemption():
    s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml")
    assert run_link_budget(s).analyzer.canceller.reference == "known_waveform"
    third_party = run_link_budget(s.with_value("transmitter.eirp_override_dbw", 40.0)).analyzer.canceller
    assert third_party.reference == "reference_beam"
    beam = run_link_budget(s.with_value("frontend.canceller_reference", "reference_beam")).analyzer.canceller
    # A beam on the transmitter sees the direct signal far above noise.
    assert 40.0 < beam.fidelity_db < 120.0
    stronger = run_link_budget(s.with_value("frontend.canceller_reference", "reference_beam").with_value("transmitter.power_w", 10 * s.transmitter.power_w))
    assert stronger.analyzer.canceller.fidelity_db == pytest.approx(beam.fidelity_db + 10.0, abs=0.01)
    pulsed = run_link_budget(Scenario.from_yaml("scenarios/pulsed_lfm_uhf.yaml"))
    assert pulsed.link.fidelity_cell_dbm <= -299.0


def test_clutter_beyond_the_canceller_taps_is_not_cancelled():
    near = Scenario.from_yaml("scenarios/dvbt_uhf.yaml").with_value("geometry.target_range_rx_km", 6.0).with_value("geometry.target_azimuth_deg", 20.0)
    covered = run_link_budget(near)
    short = run_link_budget(near.with_value("frontend.canceller_range_extent_km", 0.3))
    assert covered.link.clutter_cancelled and not short.link.clutter_cancelled
    assert short.link.clutter_cell_dbm > covered.link.clutter_cell_dbm + 10.0
    assert short.detection.margin_db < covered.detection.margin_db
    assert any("taps cover" in w for w in short.warnings)
    c = covered.analyzer.canceller
    assert c.range_extent_m == pytest.approx(near.detection.max_bistatic_range_km * 1e3)
    assert c.taps == int(np.ceil(c.range_extent_m / (299_792_458.0 / covered.waveform.bandwidth_hz)))
    assert 0.0 < c.estimation_loss_db < 0.1
