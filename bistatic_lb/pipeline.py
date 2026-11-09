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
from .target.migration import longest_cpi_s, range_walk_loss_db
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
    invalid_reasons: tuple[str, ...] = ()  # why the detection results cannot be trusted; empty when they can

    @property
    def valid(self) -> bool:
        return not self.invalid_reasons


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
    out.extend(_migration_warnings(s, wf, link))
    floor = 10.0 * np.log10(10.0 ** (link.phase_noise_cell_dbm / 10.0) + 10.0 ** (link.intermod_cell_dbm / 10.0))
    if floor > link.noise_cell_dbm - 6.0:
        parts = []
        if link.phase_noise_cell_dbm > link.noise_cell_dbm - 10.0:
            parts.append(f"LO phase noise ({link.phase_noise_cell_dbm - link.noise_cell_dbm:+.1f} dB relative to noise)")
        if link.intermod_cell_dbm > link.noise_cell_dbm - 10.0:
            parts.append(f"third-order distortion ({link.intermod_cell_dbm - link.noise_cell_dbm:+.1f} dB relative to noise)")
        out.append(
            "Front-end products of the direct signal raise the floor in the target's cell: " + " and ".join(parts) + ". "
            "Digital cancellation cannot remove them; a deeper null, analog suppression, a cleaner LO or a higher intercept point can."
        )
    frame_s = wf.cpi_s * s.detection.noncoherent_looks
    if s.tracker.update_interval_s < frame_s * (1.0 - 1e-9):
        out.append(
            f"The tracker update interval ({s.tracker.update_interval_s:g} s) is shorter than one detection frame "
            f"({frame_s:g} s: {wf.cpi_s:g} s coherent x {s.detection.noncoherent_looks} looks). Detections cannot arrive that often, "
            "so tracker results are optimistic. Raise the update interval or shorten the frame."
        )
    if not np.isfinite(est.sigma_azimuth_deg):
        out.append("This array cannot measure azimuth, so the target's position is not resolved: bi-static range alone only places it on an ellipse.")
    if not est.bound_valid:
        out.append(f"SINR {est.snr_db:.1f} dB is below the {est.threshold_snr_db:.1f} dB estimation threshold, so accuracy bounds are optimistic.")
    if wf.ambiguous_range_m is not None and link.geometry.bistatic_range_m > wf.ambiguous_range_m:
        out.append(f"The target's bi-static path exceeds the {wf.ambiguous_range_m / 1e3:.0f} km range ambiguity of {wf.name}.")
    if link.geometry.bistatic_angle_rad > np.deg2rad(160.0):
        out.append("The target is near the baseline (forward scatter): range and Doppler resolution are very poor here.")
    return out


@dataclass(frozen=True)
class _Antennas:
    """Geometry, antennas and beamformer weights: what every later stage starts from."""

    geometry: BistaticGeometry
    array: ReceiveArray
    monopole: MonopoleTx
    weights: np.ndarray
    look: tuple[float, float]
    tx_direction: tuple[float, float]
    design_rx_hz: float
    rx_mismatch_db: float
    tx_mismatch_db: float


def _antennas(s: Scenario) -> _Antennas:
    g = build_geometry(s)
    array = array_from_scenario(s)
    design_rx = s.array.design_frequency_hz or s.frequency_hz
    design_tx = s.transmitter.design_frequency_hz or s.frequency_hz
    look = (g.azimuth_at_rx_rad, g.elevation_at_rx_rad)
    tx_direction = (0.0, g.direct_path_elevation_at_rx_rad)
    null_dir = tx_direction if s.array.null_on_transmitter and array.n_elements > 1 else None
    return _Antennas(
        geometry=g,
        array=array,
        monopole=monopole_from_scenario(s),
        weights=array.beamformer(*look, null_dir=null_dir),
        look=look,
        tx_direction=tx_direction,
        design_rx_hz=design_rx,
        rx_mismatch_db=mismatch_loss_db(s.frequency_hz, design_rx, s.array.element),
        tx_mismatch_db=mismatch_loss_db(s.frequency_hz, design_tx, "monopole"),
    )


def direct_path_for(s: Scenario) -> DirectPathResult:
    """The direct-path stage alone: the same result as run_link_budget(s).direct, far quicker for sweeps."""
    a = _antennas(s)
    return compute_direct_path(
        s, a.geometry, a.array, a.weights, a.monopole, terrain_from_scenario(s.environment), a.tx_mismatch_db, a.rx_mismatch_db
    )


def _migration_warnings(s: Scenario, wf: Waveform, link: TargetLink) -> list[str]:
    """Say when the coherent interval is too long for the target, or for the fastest target designed for."""
    mode = s.waveform.migration_compensation
    g, walk = link.geometry, link.migration
    lam = wavelength_m(s.frequency_hz)
    if walk.loss_db >= 1.0:
        longest = longest_cpi_s(g.bistatic_range_rate_m_s, g.bistatic_range_accel_m_s2, wf.bandwidth_hz, lam, mode)
        return [
            f"The target crosses {walk.range_cells:.1f} range cells and {walk.doppler_cells:.1f} Doppler cells in one {wf.cpi_s:g} s coherent "
            f"interval, which costs {walk.loss_db:.1f} dB. Keep the interval under {longest:.3g} s, or compensate the migration in processing."
        ]
    if mode == "none":
        fastest_cells = 2.0 * s.detection.max_speed_m_s * wf.cpi_s * wf.bandwidth_hz / C_M_S
        fastest_loss = range_walk_loss_db(fastest_cells)
        if fastest_loss >= 1.0:
            return [
                f"This target is fine, but at the design maximum speed ({s.detection.max_speed_m_s:g} m/s) a target can cross "
                f"{fastest_cells:.1f} range cells in one coherent interval and lose up to {fastest_loss:.1f} dB."
            ]
    return []


def run_link_budget(s: Scenario) -> LinkBudgetResult:
    lam = wavelength_m(s.frequency_hz)
    waveform = make_waveform(s.waveform)
    lineup = build_lineup(s.frontend)
    terrain = terrain_from_scenario(s.environment)

    # Antennas and beamformer -----------------------------------------------------------
    front = _antennas(s)
    g, array, monopole, weights = front.geometry, front.array, front.monopole, front.weights
    look, tx_direction, design_rx = front.look, front.tx_direction, front.design_rx_hz
    rx_mismatch, tx_mismatch = front.rx_mismatch_db, front.tx_mismatch_db
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
    invalid = []
    for p in levels.compressing:
        basis = "estimated from its IIP3" if p.p1db_estimated else "as entered"
        warnings.append(
            f"The {p.name} compresses on the maximum direct-path peaks: {p.peak_dbm:.1f} dBm out against a 1 dB compression point "
            f"of {p.p1db_out_dbm:.1f} dBm ({basis}). The AGC acts only before the ADC, so it cannot prevent this; lower the gain ahead of it."
        )
        # With gated gain the pulse is blanked, so compression during it does not reach the target.
        if frontend.gain_control != "gated":
            invalid.append(
                f"The {p.name} compresses while the target is being received. Compression desensitizes the receiver and spreads "
                "distortion that this budget does not model, so Pd and accuracy are not valid until it is fixed."
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
        "Migration loss (dB)": link.migration.loss_db,
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
        invalid_reasons=tuple(invalid),
    )
