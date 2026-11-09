"""Run the whole budget: antenna, noise, direct path, front-end, target, tracker.

``run_link_budget`` is the single entry point used by the notebooks, the app and the
report. It takes a Scenario and returns a LinkBudgetResult holding each stage's output.
``result.metrics`` flattens the key numbers, for sweeps and pass/fail summaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .antenna import ReceiveArray, array_from_scenario, mismatch_loss_db, monopole_from_scenario, usable_band_hz
from .antenna.elements import MonopoleTx
from .config import Scenario
from .frontend import FrontEndResult, compute_frontend
from .frontend.levels import lineup_levels
from .frontend.requirements import with_detection_margin
from .geometry import BistaticGeometry, build_geometry
from .noise import Lineup, NoiseResult, build_lineup, compute_noise
from .propagation import DirectPathResult, compute_direct_path, terrain_from_scenario
from .target import DetectionSummary, EstimationResult, TargetAnalyzer, TargetLink, compute_estimation
from .tracker import TrackerResult, compute_tracker
from .units import C_M_S, wavelength_m
from .waveforms import Waveform, make_waveform

MIN_TX_POWER_W, MAX_TX_POWER_W = 100.0, 10_000.0


@dataclass(frozen=True)
class AntennaSummary:
    n_elements: int
    design_frequency_hz: float
    usable_band_hz: tuple[float, float]
    rx_mismatch_db: float
    tx_mismatch_db: float
    gain_toward_target_dbi: float
    gain_toward_tx_dbi: float
    null_depth_db: float
    beamwidth_az_deg: float
    peak_sidelobe_db: float
    max_spacing_wavelengths: float


@dataclass(frozen=True)
class LinkBudgetResult:
    scenario: Scenario
    geometry: BistaticGeometry
    array: ReceiveArray
    monopole: MonopoleTx
    waveform: Waveform
    lineup: Lineup
    weights: np.ndarray
    antenna: AntennaSummary
    noise: NoiseResult
    direct: DirectPathResult
    frontend: FrontEndResult
    analyzer: TargetAnalyzer
    link: TargetLink
    detection: DetectionSummary
    estimation: EstimationResult
    tracker: TrackerResult
    warnings: list[str]
    metrics: dict[str, float]


COUPLING_WARNING_DB = 1.5  # integrated gain above this means correlated elements matter


def _collect_warnings(s: Scenario, a: AntennaSummary, array: ReceiveArray, weights: np.ndarray, frontend: FrontEndResult, wf: Waveform, est: EstimationResult, link: TargetLink, detection: DetectionSummary) -> list[str]:
    out = []
    if not MIN_TX_POWER_W <= s.transmitter.power_w <= MAX_TX_POWER_W and s.transmitter.eirp_override_dbw is None:
        out.append(f"Transmit power {s.transmitter.power_w:.0f} W is outside the expected 100 W to 10 kW range.")
    if frontend.pa_saturated_w is not None and frontend.pa_saturated_w > MAX_TX_POWER_W:
        out.append(
            f"The PA would need {frontend.pa_saturated_w / 1e3:.1f} kW saturated for {s.transmitter.power_w:.0f} W mean power with {wf.name}. "
            "If your transmitter rating is a peak rating, enter a lower mean power."
        )
    if a.max_spacing_wavelengths > 0.6:
        out.append(
            f"Ring spacing is {a.max_spacing_wavelengths:.2f} wavelengths, so expect high sidelobes and possible angle ambiguity. "
            "Use more elements or a smaller radius."
        )
    integrated = array.integrate_over_sphere(weights, lambda el: np.ones_like(el), n_el=61, n_az=60)
    if 10.0 * np.log10(integrated) > COUPLING_WARNING_DB:
        out.append(
            f"Elements are closely spaced: the no-coupling model's gain integrates to {10.0 * np.log10(integrated):.1f} dB over the sphere, "
            "where a real lossless array gives 0 dB. Reported gains may be high by up to this amount."
        )
    if not a.usable_band_hz[0] <= s.frequency_hz <= a.usable_band_hz[1]:
        out.append(f"Operating frequency is outside the antenna's usable band; mismatch loss is {a.rx_mismatch_db:.1f} dB.")
    if frontend.shortfall_db > 0:
        out.append(
            f"Noise stays {frontend.shortfall_db:.1f} dB below ADC quantization even at the highest gain that avoids clipping. "
            "Only suppression before the ADC, or more ADC bits, can fix this."
        )
    elif frontend.gain_control == "gated" and frontend.required_adc_bits is not None and np.isfinite(frontend.required_adc_bits) and frontend.required_adc_bits > s.frontend.adc_bits:
        out.append(
            f"With gated gain, noise sits only {frontend.noise_over_quantization_nominal_db:.1f} dB above quantization while listening, "
            f"costing {frontend.sensitivity_loss_db:.1f} dB. Raise the line-up gain: the direct pulse does not limit it between pulses."
        )
    elif frontend.required_adc_bits is not None and np.isfinite(frontend.required_adc_bits) and frontend.required_adc_bits > s.frontend.adc_bits and detection.margin_db >= 0.0:
        out.append(
            f"Detection holds at the nominal direct-path level, but when the direct path peaks the AGC backs off and quantization costs "
            f"{frontend.worst_sensitivity_loss_db:.1f} dB, more than the margin. That needs {frontend.required_adc_bits:.1f} ADC bits."
        )
    if frontend.digital_cancellation_needed_db > s.frontend.digital_canceller_limit_db:
        out.append("Digital cancellation needed exceeds the stated limit, so direct-path residual will raise the detection floor.")
    if detection.margin_db >= 0.0 > detection.margin_upper_decile_db:
        out.append(
            f"Detection meets the requirement at median noise, but misses it by {-detection.margin_upper_decile_db:.1f} dB at upper-decile noise. "
            "Noise can exceed this level about 10% of the time and locations."
        )
    if not est.bound_valid:
        out.append(f"SINR {est.snr_db:.1f} dB is below the {est.threshold_snr_db:.1f} dB estimation threshold, so accuracy bounds are optimistic.")
    if wf.ambiguous_range_m is not None and link.geometry.bistatic_range_m > wf.ambiguous_range_m:
        out.append(f"The target's bi-static path exceeds the {wf.ambiguous_range_m / 1e3:.0f} km range ambiguity of {wf.name}.")
    if link.geometry.bistatic_angle_rad > np.deg2rad(160.0):
        out.append("The target is near the baseline (forward scatter): range and Doppler resolution are very poor here.")
    return out


def run_link_budget(s: Scenario) -> LinkBudgetResult:
    lam = wavelength_m(s.frequency_hz)
    g = build_geometry(s)
    array = array_from_scenario(s)
    monopole = monopole_from_scenario(s)
    waveform = make_waveform(s.waveform)
    lineup = build_lineup(s.frontend)
    terrain = terrain_from_scenario(s.environment)

    # Antennas and beamformer -----------------------------------------------------------
    design_rx = s.array.design_frequency_hz or s.frequency_hz
    design_tx = s.transmitter.design_frequency_hz or s.frequency_hz
    rx_mismatch = mismatch_loss_db(s.frequency_hz, design_rx, s.array.element)
    tx_mismatch = mismatch_loss_db(s.frequency_hz, design_tx, "monopole")
    look = (g.azimuth_at_rx_rad, g.elevation_at_rx_rad)
    tx_direction = (0.0, g.direct_path_elevation_at_rx_rad)
    null_dir = tx_direction if s.array.null_on_transmitter and array.n_elements > 1 else None
    weights = array.beamformer(*look, null_dir=null_dir)
    no_null = array.beamformer(*look)
    gain_tx_no_null = float(array.gain_dbi(no_null, *tx_direction))
    gain_tx = float(10.0 * np.log10(array.gain_with_errors_linear(weights, *tx_direction, s.array.calibration_error_deg)))
    antenna = AntennaSummary(
        n_elements=array.n_elements,
        design_frequency_hz=design_rx,
        usable_band_hz=usable_band_hz(design_rx, s.array.element),
        rx_mismatch_db=rx_mismatch,
        tx_mismatch_db=tx_mismatch,
        gain_toward_target_dbi=float(array.gain_dbi(weights, *look)),
        gain_toward_tx_dbi=gain_tx,
        null_depth_db=gain_tx_no_null - gain_tx,
        beamwidth_az_deg=array.beamwidth_az_deg(no_null, *look),
        peak_sidelobe_db=array.peak_sidelobe_db(no_null, *look),
        max_spacing_wavelengths=array.max_spacing_wavelengths(),
    )

    # Stage 1 and 2 ----------------------------------------------------------------------
    noise = compute_noise(s, array, weights, look, lineup, rx_mismatch)
    direct = compute_direct_path(s, g, array, weights, monopole, terrain, tx_mismatch, rx_mismatch)
    delay = g.bistatic_range_m / C_M_S
    doppler = g.doppler_hz(lam)
    frontend = compute_frontend(s, waveform, lineup, noise, direct, delay, doppler)

    # Stage 3 ------------------------------------------------------------------------------
    analyzer = TargetAnalyzer(s, array, monopole, waveform, noise, direct, frontend, tx_mismatch, rx_mismatch)
    link = analyzer.link()
    detection = analyzer.detection_summary(link)
    frontend = with_detection_margin(frontend, s, detection.margin_db)
    fluctuation_m = None if detection.fluctuation.kind in ("steady", "rician") else detection.fluctuation.m
    estimation = compute_estimation(s, array, waveform, link, detection.n_cells, fluctuation_m)

    # Stage 4 ------------------------------------------------------------------------------
    tracker = compute_tracker(s, waveform, detection, estimation, detection.fluctuation, analyzer.detection_at, lam)

    warnings = _collect_warnings(s, antenna, array, weights, frontend, waveform, estimation, link, detection)
    levels = lineup_levels(
        lineup, s.frontend, direct.element_maximum_dbm, frontend.noise_at_input_dbm, waveform.papr_db,
        frontend.pre_adc_suppression_db, min(frontend.configured_gain_db, frontend.gain_max_db), frontend.quantization_dbm,
        frontend.listening_gain_db,
    )
    for p in levels.compressing:
        basis = "estimated from its IIP3" if p.p1db_estimated else "as entered"
        warnings.append(
            f"The {p.name} compresses on the maximum direct-path peaks: {p.peak_dbm:.1f} dBm out against a 1 dB compression point "
            f"of {p.p1db_out_dbm:.1f} dBm ({basis}). The AGC acts only before the ADC, so it cannot prevent this; lower the gain ahead of it."
        )
    metrics = {
        "Noise temperature at array output (K)": noise.system_output_k,
        "Noise density at array output (dBm/Hz)": noise.noise_output_dbm_hz,
        "Upper-decile noise margin (dB)": noise.statistics.upper_margin_db,
        "Direct path, maximum per element (dBm)": direct.element_maximum_dbm,
        "Direct path, nominal per element (dBm)": direct.element_nominal_dbm,
        "Direct path, minimum per element (dBm)": direct.element_minimum_dbm,
        "Gain window width (dB)": frontend.gain_window_db,
        "Noise above quantization, best (dB)": frontend.noise_over_quantization_db,
        "Quantization loss at maximum direct signal (dB)": frontend.worst_sensitivity_loss_db,
        "Required ADC bits": frontend.required_adc_bits,
        "AGC range (dB)": frontend.agc_range_db,
        "Digital cancellation needed (dB)": frontend.digital_cancellation_needed_db,
        "Array gain toward target (dBi)": antenna.gain_toward_target_dbi,
        "Processing gain (dB)": waveform.processing_gain_db,
        "Target SNR (dB)": link.snr_db,
        "Target SINR (dB)": link.sinr_db,
        "Detection margin (dB)": detection.margin_db,
        "Single-look Pd": detection.pd,
        "Pd at upper-decile noise": detection.pd_upper_decile,
        "Detection margin at upper-decile noise (dB)": detection.margin_upper_decile_db,
        "Pfa per cell": detection.pfa,
        "Bi-static range sigma (m)": estimation.sigma_bistatic_range_m,
        "Range-rate sigma (m/s)": estimation.sigma_range_rate_m_s,
        "Azimuth sigma (deg)": estimation.sigma_azimuth_deg,
        "Position RMS, single look (m)": estimation.position_rms_m,
        "Per-look Pd with relaxed threshold": tracker.relaxed_pd,
        "Track confirmation probability": tracker.confirm_probability_at_relaxed_threshold,
        "Tracker threshold gain (dB)": tracker.trade.gain_db,
        "Tracked range sigma (m)": tracker.tracked_sigma_range_m,
        "False tracks per hour": tracker.false_tracks_per_hour,
    }
    return LinkBudgetResult(
        scenario=s,
        geometry=g,
        array=array,
        monopole=monopole,
        waveform=waveform,
        lineup=lineup,
        weights=weights,
        antenna=antenna,
        noise=noise,
        direct=direct,
        frontend=frontend,
        analyzer=analyzer,
        link=link,
        detection=detection,
        estimation=estimation,
        tracker=tracker,
        warnings=warnings,
        metrics=metrics,
    )
