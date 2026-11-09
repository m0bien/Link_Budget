"""Build a report as neutral blocks, so every output format shows the same content.

To add a section, append blocks in ``build_report``. To add a format, write one function
that walks ``report.blocks`` (see markdown.py and docx.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .. import plots
from ..pipeline import LinkBudgetResult
from ..references import REFERENCES
from ..results import contributions_table
from ..tracker.montecarlo import TrackerMonteCarloResult


@dataclass
class Heading:
    text: str
    level: int = 1


@dataclass
class Paragraph:
    text: str


@dataclass
class Bullets:
    items: list[str]


@dataclass
class Table:
    rows: list[dict]
    caption: str = ""


@dataclass
class Figure:
    figure: object  # matplotlib Figure
    caption: str
    name: str


@dataclass
class Report:
    title: str
    subtitle: str
    blocks: list = field(default_factory=list)

    def add(self, *blocks) -> None:
        self.blocks.extend(blocks)


def _level(level_dbm: float) -> str:
    return "Does not apply" if level_dbm <= -299.0 else f"{level_dbm:.1f} dBm"


def _yes_no(flag) -> str:
    return "Not rated" if flag is None else ("Met" if flag else "NOT MET")


def control_parameters(r: LinkBudgetResult) -> list[dict]:
    """The settings a designer would freeze."""
    s = r.scenario
    a, f, w, d, t = s.array, s.frontend, s.waveform, s.detection, s.tracker
    return [
        {"Parameter": "Operating frequency", "Value": f"{s.frequency_hz / 1e6:.1f} MHz"},
        {"Parameter": "Waveform", "Value": r.waveform.name},
        {"Parameter": "Bandwidth, CPI", "Value": f"{r.waveform.bandwidth_hz / 1e6:.3g} MHz, {w.cpi_s:g} s"},
        {"Parameter": "Transmit power and PA", "Value": f"{s.transmitter.power_w:.0f} W mean envelope; PA saturated power {r.frontend.pa_saturated_w or 0:.0f} W"},
        {"Parameter": "Transmit antenna", "Value": f"Monopole, {s.transmitter.height_m:.0f} m, {s.transmitter.ground_plane} ground plane"},
        {"Parameter": "Receive array", "Value": f"{a.n_circular} {a.element.replace('_', ' ')} ring, radius {a.radius_wavelengths:.2f} wavelengths; {a.n_vertical} in column"},
        {"Parameter": "Digitization, taper, null", "Value": f"{a.digitization.replace('_', ' ')}, {a.taper}, null on Tx: {'on' if a.null_on_transmitter else 'off'}"},
        {"Parameter": "ADC", "Value": f"{f.adc_bits} bits, {r.frontend.effective_bits:.1f} effective, {r.frontend.sample_rate_hz / 1e6:.3g} MS/s"},
        {"Parameter": "Receiver gain", "Value": f"{r.frontend.configured_gain_db:.1f} dB, AGC range {r.frontend.agc_range_db:.1f} dB"},
        {"Parameter": "Pre-ADC suppression", "Value": f"{r.frontend.pre_adc_suppression_db:.1f} dB"},
        {"Parameter": "Digital cancellation", "Value": f"{r.frontend.digital_cancellation_applied_db:.1f} dB applied"},
        {"Parameter": "Detector", "Value": f"Pfa {r.detection.pfa:.2e}, {d.noncoherent_looks} look(s), {d.cfar_reference_cells} CFAR cells"},
        {"Parameter": "Relaxed per-look Pfa (with tracker)", "Value": f"{r.tracker.trade.best_pfa:.2e}"},
        {"Parameter": "Track confirmation", "Value": f"{t.confirm_m} of {t.confirm_n}, update {t.update_interval_s:g} s"},
    ]


def build_report(
    r: LinkBudgetResult, mc: TrackerMonteCarloResult | None = None, status: str = "Draft", notes: str = ""
) -> Report:
    s = r.scenario
    report = Report(title=f"Bi-static link budget: {s.name}", subtitle=f"{status} design, generated {date.today().isoformat()}")

    # --- Summary -----------------------------------------------------------------------------
    pd_ok = r.detection.margin_db >= 0
    fe_ok = r.frontend.shortfall_db == 0 and r.frontend.required_adc_bits <= s.frontend.adc_bits
    tracked_ok = r.tracker.confirm_probability_at_relaxed_threshold >= s.tracker.target_confirm_probability
    report.add(Heading("Summary"))
    if r.invalid_reasons:
        report.add(Paragraph("RESULTS NOT VALID. " + " ".join(r.invalid_reasons)))
    report.add(
        Table(
            [
                {"Check": f"Single-look Pd of {s.detection.required_pd:g}", "Result": "Met" if pd_ok else "NOT MET", "Detail": f"Pd {r.detection.pd:.3f}, margin {r.detection.margin_db:+.1f} dB"},
                {"Check": "Single-look Pd at upper-decile noise", "Result": "Met" if r.detection.margin_upper_decile_db >= 0 else "NOT MET", "Detail": f"Pd {r.detection.pd_upper_decile:.3f}, margin {r.detection.margin_upper_decile_db:+.1f} dB"},
                {"Check": "Front-end gain window and ADC", "Result": "Met" if fe_ok else "NOT MET", "Detail": f"Noise {r.frontend.noise_over_quantization_db:.1f} dB above quantization at best, {r.frontend.required_adc_bits:.1f} of {s.frontend.adc_bits} bits needed"},
                {"Check": f"Track confirmation of {s.tracker.target_confirm_probability:g}", "Result": "Met" if tracked_ok else "NOT MET", "Detail": f"{r.tracker.confirm_probability_at_relaxed_threshold:.3f} at {r.tracker.false_tracks_per_hour:.2g} false tracks per hour"},
            ]
        )
    )
    if notes:
        report.add(Paragraph(notes))
    if r.warnings:
        report.add(Heading("Warnings", 2), Bullets(list(r.warnings)))

    # --- Control parameters ---------------------------------------------------------------------
    report.add(Heading("Control parameters"), Table(control_parameters(r)))

    report.add(Heading("Scenario", 2))
    g = r.geometry
    report.add(
        Table(
            [
                {"Item": "Baseline", "Value": f"{g.baseline_m / 1e3:.1f} km"},
                {"Item": "Target ranges", "Value": f"Rt {g.range_tx_m / 1e3:.1f} km, Rr {g.range_rx_m / 1e3:.1f} km"},
                {"Item": "Bi-static angle", "Value": f"{np.rad2deg(g.bistatic_angle_rad):.1f} deg"},
                {"Item": "Bi-static range and Doppler", "Value": f"{g.bistatic_range_m / 1e3:.1f} km, {r.link.doppler_hz:.0f} Hz"},
                {"Item": "Target", "Value": f"{r.link.rcs_dbsm:.1f} dBsm, {r.detection.fluctuation.name}, {s.geometry.target_altitude_m:.0f} m altitude"},
            ]
        )
    )

    # --- Antennas --------------------------------------------------------------------------------
    a = r.antenna
    report.add(Heading("Antennas"))
    report.add(
        Table(
            [
                {"Item": "Usable band (VSWR 2)", "Value": f"{a.usable_band_hz[0] / 1e6:.0f} to {a.usable_band_hz[1] / 1e6:.0f} MHz"},
                {"Item": "Element spacing", "Value": f"{a.max_spacing_wavelengths:.2f} wavelengths"},
                {"Item": "Gain toward target", "Value": f"{a.gain_toward_target_dbi:.1f} dBi"},
                {"Item": "Gain toward transmitter", "Value": f"{a.gain_toward_tx_dbi:.1f} dBi (null depth {a.null_depth_db:.1f} dB)"},
                {"Item": "Azimuth beamwidth, peak sidelobe", "Value": f"{a.beamwidth_az_deg:.1f} deg, {a.peak_sidelobe_db:.1f} dB"},
            ]
        ),
        Figure(plots.azimuth_pattern(r), "Azimuth pattern with and without the null on the transmitter.", "azimuth_pattern"),
    )

    # --- Stage 1 ---------------------------------------------------------------------------------
    n = r.noise
    report.add(Heading("Stage 1: Noise"))
    report.add(
        Paragraph(
            f"System noise temperature at the array output is {n.system_output_k:.0f} K ({n.noise_output_dbm_hz:.1f} dBm/Hz). "
            f"The dominant source is {n.dominant_source}. {n.array_noise_gain_note}"
        ),
        Table(contributions_table(n.contributions), "Noise sources considered, including those rejected."),
        Figure(plots.noise_vs_frequency(s), "Noise sources across the frequency range.", "noise_vs_frequency"),
        Heading("Noise spread", 2),
        Paragraph(
            f"Noise is at or under {n.statistics.system_upper_k:.0f} K for about 90% of times and locations, "
            f"{n.statistics.upper_margin_db:.1f} dB above the sum of medians. The design point is "
            f"{s.environment.noise_design_point.replace('_', ' ')} noise. {n.statistics.note}"
        ),
        Table(
            [
                {"Quantity": "System noise, lower decile", "Value": f"{n.statistics.system_lower_k:.0f} K"},
                {"Quantity": "System noise, Part 7 median", "Value": f"{n.statistics.system_median_k:.0f} K"},
                {"Quantity": "System noise, sum of medians", "Value": f"{n.statistics.system_sum_of_medians_k:.0f} K"},
                {"Quantity": "System noise, upper decile", "Value": f"{n.statistics.system_upper_k:.0f} K"},
                {"Quantity": "Constant part (receiver, ground, atmosphere)", "Value": f"{n.statistics.constant_k:.0f} K"},
                {"Quantity": "Man-made decile deviations, upper and lower", "Value": f"{n.statistics.man_made_upper_deviation_db:.1f} dB, {n.statistics.man_made_lower_deviation_db:.1f} dB"},
                {"Quantity": "Pd at median noise", "Value": f"{r.detection.pd:.3f}" if s.environment.noise_design_point == "median" else "See Stage 3"},
                {"Quantity": "Pd at upper-decile noise", "Value": f"{r.detection.pd_upper_decile:.3f}"},
            ],
            "P.372-18 Part 7, applied to man-made and galactic noise; other sources are treated as constant.",
        ),
        Figure(plots.noise_spread_vs_frequency(s), "System noise and its decile band.", "noise_spread"),
    )

    # --- Stage 2 ---------------------------------------------------------------------------------
    d, f = r.direct, r.frontend
    report.add(Heading("Stage 2: Direct path and RF front-end"))
    report.add(
        Paragraph(
            f"EIRP toward the receiver is {d.eirp_dbw:.1f} dBW. Direct signal per element: maximum {d.element_maximum_dbm:.1f} dBm, "
            f"nominal {d.element_nominal_dbm:.1f} dBm, minimum {d.element_minimum_dbm:.1f} dBm. {d.maximum_note}"
        ),
        Table(contributions_table(d.terms.contributions), "Propagation mechanisms on the direct path, and why each was used or rejected."),
        Heading("Dynamic range", 2),
        Paragraph(
            f"With {f.effective_bits:.1f} effective bits at {f.sample_rate_hz / 1e6:.3g} MS/s the ADC offers {f.instantaneous_dynamic_range_db:.1f} dB of "
            f"instantaneous range. Less {f.clipping_backoff_db:.1f} dB of back-off and the {f.direct_to_noise_at_adc_db:.1f} dB direct-to-noise "
            f"ratio at the ADC input, noise can sit at most {f.noise_over_quantization_db:.1f} dB above quantization, "
            f"and {f.noise_over_quantization_nominal_db:.1f} dB in operation. "
            f"The gain must lie between {f.gain_min_db:.1f} and {f.gain_max_db:.1f} dB. Quantization costs "
            f"{f.sensitivity_loss_db:.2f} dB at the nominal level and {f.worst_sensitivity_loss_db:.2f} dB at the maximum level. "
            "Digitizing each element means the direct signal reaches each ADC before beamforming, so a digital null cannot help the ADC."
            + (
                f" Gain is gated around the direct pulse: {f.pulse_gain_db:.1f} dB while it is present and {f.listening_gain_db:.1f} dB "
                f"while listening, so the target sees noise {f.noise_over_quantization_nominal_db:.1f} dB above quantization. "
                f"The receiver is blind for the pulse plus {f.blind_extra_s * 1e6:.1f} us of settling."
                if f.listening_gain_db is not None
                else ""
            )
        ),
        Figure(plots.lineup_levels(r), "Maximum direct signal and noise through the line-up, with compression points and the AGC cut.", "lineup_levels"),
        Figure(plots.dynamic_range_ladder(r), "Levels at the ADC input, at the gain the AGC sets for the nominal direct level.", "dynamic_range"),
        Heading("Derived requirements", 2),
        Table(
            [{"Requirement": q.item, "Needed": q.required, "This design": q.configured, "Status": _yes_no(q.met), "Why": q.rationale} for q in f.requirements]
        ),
        Table(r.lineup.table(), "Receiver line-up for one channel."),
    )

    # --- Stage 3 ---------------------------------------------------------------------------------
    l, det, e = r.link, r.detection, r.estimation
    report.add(Heading("Stage 3: Target, detection and estimation"))
    report.add(
        Table(
            [
                {"Term": "EIRP toward target", "Value": f"{l.eirp_toward_target_dbw:.1f} dBW (Tx gain {l.tx_gain_dbi:.1f} dBi)"},
                {"Term": "Receive gain", "Value": f"{l.rx_gain_dbi:.1f} dBi"},
                {"Term": "Target RCS", "Value": f"{l.rcs_dbsm:.1f} dBsm"},
                {"Term": "Excess path loss, Tx and Rx sides", "Value": f"{l.excess_tx_db:.1f} dB, {l.excess_rx_db:.1f} dB"},
                {"Term": "System losses", "Value": f"{l.total_losses_db:.1f} dB"},
                {"Term": "of which migration: range, Doppler", "Value": f"{l.migration.range_loss_db:.2f} dB, {l.migration.doppler_loss_db:.2f} dB ({l.migration.range_cells:.2f} and {l.migration.doppler_cells:.2f} cells crossed)"},
                {"Term": "Received target power", "Value": f"{l.received_dbm:.1f} dBm"},
                {"Term": "Processing gain", "Value": f"{r.waveform.processing_gain_db:.1f} dB"},
                {"Term": "In the target's cell: noise", "Value": f"{l.noise_cell_dbm:.1f} dBm"},
                {"Term": "In the target's cell: direct-path residual", "Value": f"{l.direct_cell_dbm:.1f} dBm"},
                {"Term": "Clutter at zero Doppler in the range cell", "Value": _level(l.clutter_zero_doppler_dbm)},
                {"Term": "In the target's cell: clutter, after the Doppler filter and canceller", "Value": _level(l.clutter_cell_dbm)},
                {"Term": "Cancellation applied; loss to the target from its notch", "Value": f"{l.cancellation_applied_db:.1f} dB; {l.notch_loss_db:.2f} dB"},
                {"Term": "Minimum detectable bi-static range rate", "Value": f"{det.min_detectable_range_rate_m_s:.2f} m/s"},
                {"Term": "Ground reflection on the target paths", "Value": f"{s.environment.target_multipath}; the actual phase here is {l.lobing_extra_loss_db:+.1f} dB against the average"},
                {"Term": "In the target's cell: LO phase noise of the direct signal", "Value": _level(l.phase_noise_cell_dbm)},
                {"Term": "In the target's cell: third-order distortion of the direct signal", "Value": _level(l.intermod_cell_dbm)},
                {"Term": "SNR, SINR", "Value": f"{l.snr_db:.1f} dB, {l.sinr_db:.1f} dB"},
                {"Term": "Pfa, search cells", "Value": f"{det.pfa:.2e}, {det.n_cells:.3g}"},
                {"Term": "Required SINR, CFAR loss", "Value": f"{det.required_snr_db:.1f} dB, {det.cfar_loss_db:.2f} dB"},
                {"Term": "Pd, margin", "Value": f"{det.pd:.3f}, {det.margin_db:+.1f} dB"},
                {"Term": "Maximum Rt x Rr at required Pd", "Value": f"{det.max_range_product_km2:.0f} km^2"},
            ],
            "Target link budget.",
        ),
        Figure(plots.pd_vs_snr(r), "Detection probability for each fluctuation model.", "pd_vs_snr"),
        Figure(plots.pd_vs_range(r), "Detection along a radial from the receiver.", "pd_vs_range"),
        Figure(plots.detection_map(r), "Detection region over the Tx-Rx plane, for one half-plane.", "detection_map"),
        Heading("Estimation", 2),
        Paragraph(
            ("Accuracy bounds are valid here: " if e.bound_valid else "Caution, accuracy bounds are optimistic here: ")
            + f"SINR is {e.snr_db:.1f} dB against an estimated threshold of {e.threshold_snr_db:.1f} dB."
        ),
        Table(
            [
                {"Parameter": "Bi-static range", "Single-look sigma": f"{e.sigma_bistatic_range_m:.1f} m", "Resolution": f"{e.range_resolution_m:.0f} m along the bisector"},
                {"Parameter": "Doppler", "Single-look sigma": f"{e.sigma_doppler_hz:.3f} Hz", "Resolution": f"{e.doppler_resolution_hz:.2f} Hz"},
                {"Parameter": "Range rate", "Single-look sigma": f"{e.sigma_range_rate_m_s:.3f} m/s", "Resolution": "-"},
                {"Parameter": "Azimuth", "Single-look sigma": f"{e.sigma_azimuth_deg:.2f} deg", "Resolution": f"{a.beamwidth_az_deg:.1f} deg beam"},
                {"Parameter": "Elevation", "Single-look sigma": "Not available" if e.sigma_elevation_deg is None else f"{e.sigma_elevation_deg:.2f} deg", "Resolution": "-"},
                {"Parameter": "Position (rms)", "Single-look sigma": f"{e.position_rms_m:.0f} m (x {e.sigma_x_m:.0f}, y {e.sigma_y_m:.0f})", "Resolution": "-"},
                {"Parameter": "RCS", "Single-look sigma": f"{e.rcs_measurement_std_db:.2f} dB from noise, {e.rcs_total_std_db:.2f} dB with fluctuation", "Resolution": "-"},
            ]
        ),
        Figure(plots.range_accuracy_vs_snr(r), "Range accuracy bound and threshold.", "range_accuracy"),
    )

    # --- Stage 4 ---------------------------------------------------------------------------------
    t = r.tracker
    report.add(Heading("Stage 4: Tracker"))
    report.add(
        Paragraph(
            f"M-of-N confirmation allows a lower per-look threshold. Per-look Pfa can rise from {t.single_look_pfa:.2e} to {t.trade.best_pfa:.2e}, "
            f"reducing the required SNR by {t.trade.gain_db:.1f} dB at {t.false_tracks_per_hour:.2g} false tracks per hour. {t.latency_note}"
        ),
        Table(
            [
                {"Measure": "Pd (per look / confirmed track)", "Single look": f"{t.single_look_pd:.3f}", "With tracker": f"{t.relaxed_pd:.3f} / {t.confirm_probability_at_relaxed_threshold:.3f}"},
                {"Measure": "False reports per hour", "Single look": f"{t.single_look_false_alarms_per_hour:.3g} false alarms", "With tracker": f"{t.false_tracks_per_hour:.3g} false tracks"},
                {"Measure": "Bi-static range sigma", "Single look": f"{t.single_look_sigma_range_m:.2f} m", "With tracker": f"{t.tracked_sigma_range_m:.2f} m ({t.range_improvement_db:+.1f} dB)"},
                {"Measure": "Range-rate sigma", "Single look": f"{t.single_look_sigma_rate_m_s:.3f} m/s", "With tracker": f"{t.tracked_sigma_rate_m_s:.3f} m/s ({t.rate_improvement_db:+.1f} dB)"},
            ],
            "Analytic tracker results.",
        ),
        Figure(plots.threshold_trade_snr(r), "Threshold trade: tracker gain.", "threshold_trade"),
        Figure(plots.confirmation_curves(r), "M-of-N confirmation probability.", "confirmation"),
    )
    if mc is not None:
        report.add(
            Heading("Monte Carlo", 2),
            Paragraph(
                f"{mc.runs} runs with seed {mc.seed}; {s.tracker.association} association. Track continuity {mc.continuity:.1%}. "
                f"Mean position error over the last third of the run is {np.mean(mc.rmse_position_m[-len(mc.time_s) // 3:]):.0f} m, "
                f"against a single-look bound of {np.mean(mc.single_look_position_m[-len(mc.time_s) // 3:]):.0f} m."
            ),
            Figure(plots.tracker_rmse(mc, "range"), "Simulated range error against the single-look bound and the PCRLB.", "tracker_range"),
            Figure(plots.tracker_rmse(mc, "position"), "Simulated Cartesian position error.", "tracker_position"),
        )
    else:
        report.add(Paragraph("Tracker Monte Carlo was not run for this report."))

    # --- Waveform notes, limits and references -----------------------------------------------------
    report.add(Heading("Limitations and notes"))
    report.add(
        Bullets(
            list(r.waveform.notes())
            + [
                "Man-made noise above 250 MHz is an estimate from the slope of P.372-18 Fig. 3, so measure at the site if it matters.",
                "Part 7 noise statistics treat receiver, ground and atmosphere noise as constant. Table 2 deviations are averages for 0.3 to 250 MHz, and may be pessimistic above that.",
                "Array patterns ignore mutual coupling, which affects close-spaced arrays and deep nulls most.",
                "The direct-path maximum is an engineering upper bound, not a statistic.",
                "Accuracy bounds and tracked errors reflect noise only. Timing and synchronization errors, hardware drift and target extent add error floors in a real system.",
                "Ground constants are representative classes, not the full P.527-6 model.",
                "Geoclimatic K and dN75 are inputs, because the P.530 data grids are not bundled.",
                "Results cover one Tx-Rx pair. Multi-static operation and DEM terrain lookup are extension paths.",
            ]
        )
    )
    report.add(Heading("References and verification status"))
    status_text = {
        "checked": "Equations read from this edition",
        "older": "Read from an older edition; re-check",
        "library": "Through the ITU-Rpy package",
        "values": "Representative values only",
        "textbook": "Textbook result",
    }
    report.add(
        Table([{"Source": ref.citation, "Status": status_text[ref.status], "Note": ref.note} for ref in REFERENCES.values()])
    )
    return report
