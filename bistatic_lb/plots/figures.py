"""Figures for each stage. Each function takes results and returns a matplotlib Figure.

Nothing here computes physics beyond calling the library. Add a figure by adding a function,
then call it from an app page or from report/builder.py.
"""

from __future__ import annotations

import numpy as np

from ..config import Scenario
from ..frontend.levels import levels_for
from ..noise.budget import compute_noise
from ..noise.receiver import build_lineup
from ..pipeline import LinkBudgetResult, direct_path_for, run_link_budget
from ..antenna import array_from_scenario
from ..target.detection import FLUCTUATION_MODELS, pd_curve
from ..target.estimation import delay_crlb_s
from ..tracker.initiation import confirm_probability
from ..tracker.montecarlo import TrackerMonteCarloResult
from ..units import C_M_S, undb
from .style import CRITICAL, INK, INK_SECONDARY, LINE_STYLES, SEQUENTIAL, SERIES, legend, marker_note, new_figure, title


# --- Antenna ---------------------------------------------------------------------------------
def azimuth_pattern(r: LinkBudgetResult):
    g = r.geometry
    az = np.deg2rad(np.linspace(-180.0, 180.0, 1441))
    el = g.elevation_at_rx_rad
    fig, ax = new_figure()
    no_null = r.array.beamformer(g.azimuth_at_rx_rad, el)
    ax.plot(np.rad2deg(az), r.array.gain_dbi(no_null, az, np.full_like(az, el)), color=SERIES[0], lw=2, label="Beam on target")
    if r.scenario.array.null_on_transmitter and r.array.n_elements > 1:
        ax.plot(np.rad2deg(az), r.array.gain_dbi(r.weights, az, np.full_like(az, el)), color=SERIES[1], lw=1.5, ls="--", label="With null on transmitter")
    ax.axvline(np.rad2deg(g.azimuth_at_rx_rad), color=INK_SECONDARY, lw=1, ls=":")
    ax.axvline(0.0, color=INK_SECONDARY, lw=1, ls=":")
    ax.set_ylim(-40, max(r.antenna.gain_toward_target_dbi + 5.0, 10.0))
    marker_note(ax, np.rad2deg(g.azimuth_at_rx_rad), ax.get_ylim()[1] - 4, "Target")
    marker_note(ax, 0.0, ax.get_ylim()[1] - 4, "Transmitter")
    ax.set_xlabel("Azimuth from the receiver-to-transmitter direction (deg)")
    ax.set_ylabel("Gain (dBi)")
    title(ax, f"Azimuth pattern: {r.array.n_elements} elements, {r.antenna.beamwidth_az_deg:.0f} deg beamwidth")
    legend(ax, loc="lower right")
    return fig


def elevation_pattern(r: LinkBudgetResult):
    g = r.geometry
    el = np.deg2rad(np.linspace(-90.0, 90.0, 721))
    az = np.full_like(el, g.azimuth_at_rx_rad)
    fig, ax = new_figure()
    ax.plot(np.rad2deg(el), r.array.gain_dbi(r.weights, az, el), color=SERIES[0], lw=2)
    ax.axvline(np.rad2deg(g.elevation_at_rx_rad), color=INK_SECONDARY, lw=1, ls=":")
    marker_note(ax, np.rad2deg(g.elevation_at_rx_rad), 0.0, "Target")
    ax.set_ylim(-40, max(r.antenna.gain_toward_target_dbi + 5.0, 10.0))
    ax.set_xlabel("Elevation (deg)")
    ax.set_ylabel("Gain (dBi)")
    title(ax, "Elevation pattern at the target azimuth")
    return fig


def array_layout(r: LinkBudgetResult):
    p = r.array.positions_m / r.array.wavelength_m
    fig, ax = new_figure(4.2, 3.9)
    on_ring = np.linalg.norm(p[:, :2], axis=1) > 0
    ax.scatter(p[on_ring, 0], p[on_ring, 1], s=60, color=SERIES[0], label="Ring", zorder=3)
    if (~on_ring).any():
        ax.scatter(p[~on_ring, 0], p[~on_ring, 1], s=60, color=SERIES[1], marker="s", label=f"Column, {int((~on_ring).sum())} elements", zorder=3)
        legend(ax, loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (wavelengths)")
    ax.set_ylabel("y (wavelengths)")
    title(ax, "Array layout, seen from above")
    return fig


# --- Noise -----------------------------------------------------------------------------------
def noise_vs_frequency(s: Scenario, points: int = 32):
    """Noise sources at one tuned dipole, across the tool's frequency range."""
    freqs = np.geomspace(50e6, 18e9, points)
    lineup = build_lineup(s.frontend)
    labels = ["Man-made", "Galactic and cosmic", "Atmosphere", "Ground pick-up"]
    rows = {label: [] for label in labels}
    for f in freqs:
        one = s.with_value("frequency_hz", float(f))
        one = one.with_value("array.n_circular", 1).with_value("array.n_vertical", 0)
        one = one.with_value("array.design_frequency_hz", float(f))
        array = array_from_scenario(one)
        weights = np.ones(1, complex)
        result = compute_noise(one, array, weights, (0.0, 0.0), lineup, 0.0)
        for label, c in zip(labels, result.contributions[:4]):
            rows[label].append(max(c.value, 1e-3))
    fig, ax = new_figure()
    for i, label in enumerate(labels):
        ax.loglog(freqs / 1e6, rows[label], color=SERIES[i], ls=LINE_STYLES[i], lw=2, label=label)
    ax.axhline(lineup.noise_temperature_k, color=INK, lw=1.5, ls=(0, (1, 1)), label=f"Receiver, {lineup.noise_figure_db:.1f} dB NF")
    ax.axvline(250.0, color=INK_SECONDARY, lw=1, ls=":")
    ax.annotate("Man-made eq. 17 ends at 250 MHz", (250.0, 1e-2), xytext=(5, 0), textcoords="offset points", fontsize=8, color=INK_SECONDARY)
    ax.axvline(s.frequency_hz / 1e6, color=SERIES[6], lw=1.2)
    ax.annotate("Operating frequency", (s.frequency_hz / 1e6, 2e-3), xytext=(5, 0), textcoords="offset points", fontsize=8, color=INK_SECONDARY)
    ax.set_ylim(1e-3, 3e5)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Noise temperature (K)")
    title(ax, f"Where noise comes from: {s.environment.man_made_category.replace('_', ' ')} site, one dipole")
    legend(ax, loc="upper right", ncols=2)
    return fig


def noise_spread_vs_frequency(s: Scenario, points: int = 32):
    """System noise with its decile band at one tuned dipole, from P.372-18 Part 7."""
    freqs = np.geomspace(50e6, 18e9, points)
    lineup = build_lineup(s.frontend)
    median, upper, lower = [], [], []
    for f in freqs:
        one = s.with_value("frequency_hz", float(f)).with_value("array.n_circular", 1).with_value("array.n_vertical", 0)
        one = one.with_value("array.design_frequency_hz", float(f))
        stats = compute_noise(one, array_from_scenario(one), np.ones(1, complex), (0.0, 0.0), lineup, 0.0).statistics
        median.append(stats.system_median_k)
        upper.append(stats.system_upper_k)
        lower.append(stats.system_lower_k)
    fig, ax = new_figure()
    ax.fill_between(freqs / 1e6, lower, upper, color=SERIES[0], alpha=0.18, lw=0, label="Between the deciles: 80% of times and locations")
    ax.loglog(freqs / 1e6, upper, color=SERIES[1], lw=2, ls="--", label="Upper decile")
    ax.loglog(freqs / 1e6, median, color=SERIES[0], lw=2, label="Median")
    ax.loglog(freqs / 1e6, lower, color=SERIES[2], lw=2, ls="-.", label="Lower decile")
    ax.axvline(250.0, color=INK_SECONDARY, lw=1, ls=":")
    ax.annotate("Table 2 deviations are for 250 MHz and below", (250.0, 0.04 * max(upper)), xytext=(5, 0), textcoords="offset points", fontsize=8, color=INK_SECONDARY)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("System noise temperature (K)")
    title(ax, f"How far noise can move: {s.environment.man_made_category.replace('_', ' ')} site, one dipole")
    legend(ax, loc="upper right")
    return fig


# --- Direct path and front-end -----------------------------------------------------------------
def direct_levels_vs_baseline(s: Scenario, points: int = 25):
    baselines = np.geomspace(1.0, 150.0, points)
    nominal, maximum, minimum = [], [], []
    for b in baselines:
        direct = direct_path_for(s.with_value("geometry.baseline_km", float(b)))
        nominal.append(direct.element_nominal_dbm)
        maximum.append(direct.element_maximum_dbm)
        minimum.append(direct.element_minimum_dbm)
    fig, ax = new_figure()
    ax.semilogx(baselines, maximum, color=SERIES[1], lw=2, ls="--", label="Maximum (sizes ADC and limiter)")
    ax.semilogx(baselines, nominal, color=SERIES[0], lw=2, label="Nominal")
    ax.semilogx(baselines, minimum, color=SERIES[2], lw=2, ls="-.", label="Minimum (sizes AGC)")
    ax.axvline(s.geometry.baseline_km, color=INK_SECONDARY, lw=1, ls=":")
    marker_note(ax, s.geometry.baseline_km, np.min(minimum) + 5, "This design")
    ax.set_xlabel("Baseline (km)")
    ax.set_ylabel("Direct signal per element (dBm)")
    title(ax, "Direct-path level against baseline")
    legend(ax, loc="lower left")
    return fig


def dynamic_range_ladder(r: LinkBudgetResult):
    """Levels at the ADC input, for the operating gain the AGC chooses at the nominal direct level."""
    f = r.frontend
    s = r.scenario.frontend
    gain = f.operating_gain_db
    backoff = r.waveform.papr_db + s.adc_headroom_db
    clip = s.adc_full_scale_dbm - backoff
    levels = [
        ("ADC full scale", s.adc_full_scale_dbm, INK),
        (f"Clipping limit ({backoff:.1f} dB back-off)", clip, CRITICAL),
        ("Direct path, maximum", r.direct.element_maximum_dbm + gain - f.pre_adc_suppression_db, SERIES[1]),
        ("Direct path, nominal", r.direct.element_nominal_dbm + gain - f.pre_adc_suppression_db, SERIES[0]),
        ("Direct path, minimum", r.direct.element_minimum_dbm + gain - f.pre_adc_suppression_db, SERIES[2]),
        ("Receiver noise in band" + (", during the pulse" if f.listening_gain_db is not None else ""), f.noise_at_input_dbm + gain, SERIES[6]),
        ("ADC quantization in band", f.quantization_dbm, INK_SECONDARY),
    ]
    if f.listening_gain_db is not None:
        levels.append(("Receiver noise in band, listening", f.noise_at_input_dbm + f.listening_gain_db, SERIES[5]))
    fig, ax = new_figure(7.2, 4.2)
    order = sorted(levels, key=lambda item: -item[1])
    top = order[0][1] + 3.0
    # Anything above the clipping limit has its peaks cut off at this gain.
    ax.axhspan(clip, top, color=CRITICAL, alpha=0.07, lw=0)
    # Spread label positions so close levels stay readable.
    gap = (order[0][1] - order[-1][1]) / 9.0
    position = order[0][1]
    for name, level, color in order:
        position = min(position, level)
        clips = name.startswith("Direct path") and level > clip
        ax.plot([0.0, 0.42], [level, level], color=color, lw=2.5)
        ax.plot([0.42, 0.5], [level, position], color=color, lw=0.8)
        label = f"{name}: {level:.1f} dBm" + (f" (AGC must cut {level - clip:.1f} dB)" if clips else "")
        ax.text(0.51, position, label, va="center", fontsize=9, color=CRITICAL if clips else INK, fontweight="bold" if clips else "normal")
        position -= gap
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(top=top)
    ax.set_xticks([])
    ax.grid(False, axis="x")
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Level at the ADC input (dBm)")
    if f.listening_gain_db is not None:
        heading = f"Gated gain: {gain:.1f} dB during the direct pulse, {f.listening_gain_db:.1f} dB listening"
    elif f.configured_gain_db - gain > 0.05:
        heading = f"ADC input at {gain:.1f} dB gain: AGC takes {f.configured_gain_db - gain:.1f} dB off the {f.configured_gain_db:.1f} dB configured"
    else:
        heading = f"ADC input at the configured {gain:.1f} dB gain"
    title(ax, f"{heading}\n{f.effective_bits:.1f} effective bits. Shaded: peaks clip")
    return fig


def lineup_levels(r: LinkBudgetResult):
    """The maximum direct path and the noise through each stage, with compression points and the ADC limits."""
    lv = levels_for(r)
    pts = lv.points
    x = np.arange(len(pts))
    fig, ax = new_figure(7.2, 4.6)
    ax.plot(x, [p.peak_dbm for p in pts], color=SERIES[1], lw=1.5, ls="--", marker="o", ms=3.5, label="Direct path, maximum, peak")
    ax.plot(x, [p.direct_dbm for p in pts], color=SERIES[1], lw=2, marker="o", ms=4, label="Direct path, maximum, mean")
    ax.plot(x, [p.noise_dbm for p in pts], color=SERIES[6], lw=2, marker="o", ms=4, label="Noise in band")
    for i, p in enumerate(pts):
        if p.p1db_out_dbm is None:
            continue
        face = "none" if p.p1db_estimated else CRITICAL
        ax.scatter([i], [p.p1db_out_dbm], marker="v", s=55, facecolors=face, edgecolors=CRITICAL, zorder=4)
        if p.compresses:
            marker_note(ax, i, p.peak_dbm, "compresses", offset=(5, 5))
    ax.scatter([], [], marker="v", s=55, facecolors=CRITICAL, edgecolors=CRITICAL, label="Output P1dB (hollow: estimated from IIP3)")
    # ADC limits at the last point.
    end = x[-1]
    label_y = None
    for level, color, style, name in (
        (lv.adc_full_scale_dbm, INK, "-", "Full scale"),
        (lv.clipping_limit_dbm, CRITICAL, "--", "Clipping limit, mean"),
        (lv.quantization_dbm, INK_SECONDARY, ":", "Quantization"),
    ):
        ax.plot([end - 0.45, end + 0.45], [level, level], color=color, ls=style, lw=1.6)
        # Keep labels of close levels apart.
        label_y = level if label_y is None else min(level, label_y - 6.0)
        ax.text(end + 0.5, label_y, name, va="center", fontsize=8.5, color=color)
    if lv.agc_cut_db > 0.05:
        ax.annotate(
            "", (end, pts[-1].direct_dbm), (end, lv.direct_without_agc_dbm),
            arrowprops={"arrowstyle": "->", "color": INK_SECONDARY, "lw": 1.2},
        )
        ax.scatter([end], [lv.direct_without_agc_dbm], marker="o", s=30, facecolors="none", edgecolors=SERIES[1], zorder=4)
    if lv.listening_noise_dbm is not None:
        ax.scatter([end], [lv.listening_noise_dbm], marker="D", s=40, color=SERIES[5], zorder=5, label="Noise at the ADC while listening (gated gain)")
        if lv.listening_noise_dbm - pts[-1].noise_dbm > 0.5:
            ax.annotate(
                "", (end, lv.listening_noise_dbm), (end, pts[-1].noise_dbm),
                arrowprops={"arrowstyle": "->", "color": SERIES[5], "lw": 1.2},
            )
    ax.set_xticks(x, [p.name.replace("IF amplifier", "IF amp").replace("Analog suppression", "Analog supp.") for p in pts], rotation=30, ha="right")
    ax.set_xlim(-0.3, end + 1.6)
    ax.set_ylabel("Level at the stage output (dBm)")
    over = lv.overshoot_without_agc_db
    if lv.listening_noise_dbm is not None and lv.agc_cut_db > 0.05:
        note = f"Gated: {lv.agc_cut_db:.1f} dB cut for the direct pulse only, restored while listening"
    elif lv.agc_cut_db > 0.05:
        note = f"AGC cuts {lv.agc_cut_db:.1f} dB; without it the ADC clips by {over:.1f} dB"
    else:
        note = f"No AGC cut needed: {-over:.1f} dB below the clipping limit"
    title(ax, f"Levels through the line-up: {lv.configured_gain_db:.1f} dB configured gain\n{note}")
    legend(ax, loc="upper left")
    return fig


def gain_window_vs_bits(r: LinkBudgetResult):
    bits = np.arange(8, 21)
    s = r.scenario
    # Only the quantization floor moves with resolution: 6.02 dB per effective bit, at a fixed full scale.
    effective = np.maximum(bits - s.frontend.adc_enob_loss_bits, 1.0)
    widths = r.frontend.gain_window_db + 6.02 * (effective - r.frontend.effective_bits)
    fig, ax = new_figure()
    ax.plot(bits, widths, color=SERIES[0], lw=2, marker="o", ms=5)
    ax.axhline(0.0, color=CRITICAL, lw=1.2)
    ax.annotate("Below this line, noise stays under quantization at any gain", (bits[0], 0.0), xytext=(0, 5), textcoords="offset points", fontsize=9, color=INK)
    ax.scatter([s.frontend.adc_bits], [r.frontend.gain_window_db], s=70, color=SERIES[1], zorder=3)
    marker_note(ax, s.frontend.adc_bits, r.frontend.gain_window_db, "This design", offset=(6, -14))
    ax.set_xlabel("ADC bits (nominal)")
    ax.set_ylabel("Noise above quantization, best gain (dB)")
    title(ax, "ADC resolution against noise above quantization")
    return fig


# --- Target ------------------------------------------------------------------------------------
def pd_vs_snr(r: LinkBudgetResult):
    n = r.scenario.detection.noncoherent_looks
    pfa = r.detection.pfa
    snr = np.linspace(-5.0, 40.0, 91)
    fig, ax = new_figure()
    names = {"swerling0": "Swerling 0", "swerling1": "Swerling 1", "swerling2": "Swerling 2", "swerling3": "Swerling 3", "swerling4": "Swerling 4"}
    for i, (key, name) in enumerate(names.items()):
        ax.plot(snr, pd_curve(snr, pfa, n, FLUCTUATION_MODELS[key]), color=SERIES[i], ls=LINE_STYLES[i], lw=1.8, label=name)
    ax.scatter([r.link.sinr_db - r.detection.cfar_loss_db], [r.detection.pd], color=INK, s=60, zorder=4)
    marker_note(ax, r.link.sinr_db - r.detection.cfar_loss_db, r.detection.pd, f"This design: {r.detection.pd:.2f}", offset=(8, -12))
    ax.axhline(r.scenario.detection.required_pd, color=INK_SECONDARY, lw=1, ls=":")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("SINR after processing and CFAR loss (dB)")
    ax.set_ylabel("Probability of detection")
    title(ax, f"Detection at Pfa = {pfa:.1e}, {n} look(s)")
    legend(ax, loc="lower right")
    return fig


def pd_lookup(r: LinkBudgetResult):
    """Fast Pd from SINR, by interpolation, for sweeps along paths and across maps."""
    grid = np.linspace(-20.0, 60.0, 161)
    values = np.array([r.analyzer.detection_at(float(v)) for v in grid])
    return lambda sinr: np.interp(sinr, grid, values)


def pd_vs_range(r: LinkBudgetResult, points: int = 60):
    s = r.scenario
    ranges = np.linspace(2.0, max(3.0 * s.geometry.target_range_rx_km, 60.0), points)
    az = np.deg2rad(s.geometry.target_azimuth_deg)
    lookup = pd_lookup(r)
    sinr = np.array([r.analyzer.link([d * 1e3 * np.cos(az), d * 1e3 * np.sin(az), s.geometry.target_altitude_m]).sinr_db for d in ranges])
    fig, ax = new_figure()
    ax.plot(ranges, lookup(sinr), color=SERIES[0], lw=2)
    ax.axhline(s.detection.required_pd, color=INK_SECONDARY, lw=1, ls=":")
    ax.axvline(s.geometry.target_range_rx_km, color=INK_SECONDARY, lw=1, ls=":")
    marker_note(ax, s.geometry.target_range_rx_km, 0.05, "This target")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(f"Range from receiver along {s.geometry.target_azimuth_deg:.0f} deg azimuth (km)")
    ax.set_ylabel("Single-look Pd")
    title(ax, f"Detection along a radial, at {s.geometry.target_altitude_m:.0f} m altitude")
    return fig


def detection_map(r: LinkBudgetResult, points: int = 41, half_width_km: float | None = None):
    """Single-look Pd over the Tx-Rx plane: the bi-static detection contour."""
    from matplotlib.colors import LinearSegmentedColormap

    s = r.scenario
    baseline = s.geometry.baseline_km
    span = half_width_km or max(2.5 * baseline, 1.5 * s.geometry.target_range_rx_km)
    xs = np.linspace(baseline / 2.0 - span, baseline / 2.0 + span, points)
    ys = np.linspace(0.5, span, points // 2 + 1)
    lookup = pd_lookup(r)
    sinr = np.zeros((len(ys), len(xs)))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            sinr[i, j] = r.analyzer.link([x * 1e3, y * 1e3, s.geometry.target_altitude_m]).sinr_db
    pd = lookup(sinr)
    fig, ax = new_figure(7.2, 4.4)
    cmap = LinearSegmentedColormap.from_list("pd", SEQUENTIAL)
    filled = ax.contourf(xs, ys, pd, levels=np.linspace(0.0, 1.0, 11), cmap=cmap)
    lines = ax.contour(xs, ys, pd, levels=[s.detection.required_pd], colors=[INK], linewidths=1.5)
    ax.clabel(lines, fmt=f"Pd = {s.detection.required_pd:g}", fontsize=8)
    colorbar = fig.colorbar(filled, ax=ax, label="Single-look Pd")
    colorbar.ax.tick_params(labelsize=8, colors=INK_SECONDARY)
    ax.scatter([0.0], [0.5], marker="^", s=70, color=SERIES[1], zorder=4, label="Receiver")
    ax.scatter([baseline], [0.5], marker="s", s=60, color=SERIES[2], zorder=4, label="Transmitter")
    tgt = r.geometry.target
    ax.scatter([tgt[0] / 1e3], [tgt[1] / 1e3], marker="o", s=50, color=INK, zorder=4, label="This target")
    ax.grid(False)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    title(ax, f"Detection region, {s.geometry.target_altitude_m:.0f} m altitude")
    legend(ax, loc="upper right", frameon=True, facecolor="#fcfcfb", edgecolor="#e4e3df")
    return fig


def range_accuracy_vs_snr(r: LinkBudgetResult, simulated_snr_db=None, simulated_rmse_samples=None):
    snr = np.linspace(0.0, 40.0, 81)
    sigma = C_M_S * delay_crlb_s(undb(snr), r.waveform.rms_bandwidth_hz)
    fig, ax = new_figure()
    ax.semilogy(snr, sigma, color=SERIES[0], lw=2, label="Cramer-Rao bound")
    if simulated_snr_db is not None:
        # Simulation time units are samples at the Nyquist rate of the waveform's bandwidth.
        ax.semilogy(simulated_snr_db, np.asarray(simulated_rmse_samples) * C_M_S / r.waveform.bandwidth_hz, color=SERIES[1], lw=0, marker="o", ms=6, label="Simulated estimator")
    ax.axvline(r.estimation.threshold_snr_db, color=CRITICAL, lw=1.2, ls="--")
    marker_note(ax, r.estimation.threshold_snr_db, sigma[0] * 0.8, "Estimated threshold")
    ax.scatter([r.link.sinr_db], [r.estimation.sigma_bistatic_range_m], color=INK, s=60, zorder=4)
    marker_note(ax, r.link.sinr_db, r.estimation.sigma_bistatic_range_m, "This design")
    ax.set_xlabel("SINR, E/N0 (dB)")
    ax.set_ylabel("Bi-static range error (m)")
    title(ax, f"Range accuracy for {r.waveform.name}")
    legend(ax, loc="upper right" if simulated_snr_db is None else "lower left")
    return fig


# --- Tracker -----------------------------------------------------------------------------------
def threshold_trade_snr(r: LinkBudgetResult):
    t = r.tracker.trade
    limit = r.scenario.tracker.allowed_false_tracks_per_hour
    fig, ax = new_figure()
    ax.semilogx(t.pfa_grid, t.required_snr_db, color=SERIES[0], lw=2, label=f"Per-look SNR for {r.scenario.tracker.confirm_m}-of-{r.scenario.tracker.confirm_n} confirmation")
    feasible = t.false_tracks_per_hour <= limit
    if (~feasible).any():
        ax.axvspan(t.pfa_grid[~feasible].min(), t.pfa_grid.max(), color=CRITICAL, alpha=0.08, label=f"More than {limit:g} false track(s) per hour")
    ax.axhline(t.single_look_snr_db, color=SERIES[1], lw=1.5, ls="--", label="Single-look detection, same probability")
    ax.scatter([t.best_pfa], [t.best_snr_db], color=INK, s=60, zorder=4)
    marker_note(ax, t.best_pfa, t.best_snr_db, f"Gain {t.gain_db:.1f} dB", offset=(-6, 10))
    ax.set_xlabel("Per-look false-alarm probability")
    ax.set_ylabel("Required SNR per look (dB)")
    title(ax, "Lowering the threshold, and letting the tracker clean up")
    legend(ax, loc="lower left")
    return fig


def confirmation_curves(r: LinkBudgetResult):
    p = np.linspace(0.0, 1.0, 101)
    m, n = r.scenario.tracker.confirm_m, r.scenario.tracker.confirm_n
    options = [(m, n), (2, 3), (3, 5), (4, 6)]
    options = list(dict.fromkeys(options))
    fig, ax = new_figure()
    ax.plot(p, p, color=INK_SECONDARY, lw=1, ls=":", label="Single look")
    for i, (a, b) in enumerate(options):
        ax.plot(p, [confirm_probability(v, a, b) for v in p], color=SERIES[i], ls=LINE_STYLES[i], lw=2, label=f"{a} of {b}")
    ax.scatter([r.tracker.relaxed_pd], [r.tracker.confirm_probability_at_relaxed_threshold], color=INK, s=60, zorder=4)
    marker_note(ax, r.tracker.relaxed_pd, r.tracker.confirm_probability_at_relaxed_threshold, "This design", offset=(8, -12))
    ax.set_xlabel("Per-look Pd")
    ax.set_ylabel("Track confirmation probability")
    title(ax, "M-of-N confirmation")
    legend(ax, loc="upper left")
    return fig


def tracker_rmse(mc: TrackerMonteCarloResult, quantity: str = "range"):
    fig, ax = new_figure()
    if quantity == "range":
        series = [(mc.single_look_range_m, "Single look (bound)", SERIES[1], "--"), (mc.rmse_range_m, "Tracker, simulated", SERIES[0], "-"), (mc.bound_range_m, "PCRLB", SERIES[2], "-.")]
        label, name = "Bi-static range error (m)", "Range"
    elif quantity == "rate":
        series = [(mc.single_look_rate_m_s, "Single look (bound)", SERIES[1], "--"), (mc.rmse_rate_m_s, "Tracker, simulated", SERIES[0], "-"), (mc.bound_rate_m_s, "PCRLB", SERIES[2], "-.")]
        label, name = "Range-rate error (m/s)", "Range-rate"
    else:
        series = [(mc.single_look_position_m, "Single look (bound)", SERIES[1], "--"), (mc.rmse_position_m, "Cartesian tracker, simulated", SERIES[0], "-")]
        label, name = "Position error (m)", "Position"
    for values, text, color, style in series:
        ax.semilogy(mc.time_s, values, color=color, ls=style, lw=2, label=text)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(label)
    title(ax, f"{name} error: {mc.runs} runs, continuity {mc.continuity:.0%}")
    legend(ax, loc="best")
    return fig


def sweep_line(x, y, x_label: str, y_label: str, current_x: float | None = None):
    fig, ax = new_figure()
    x = np.asarray(x, float)
    if x.min() > 0 and x.max() / x.min() >= 10.0:
        ax.set_xscale("log")
    ax.plot(x, y, color=SERIES[0], lw=2, marker="o", ms=4)
    if current_x is not None:
        ax.axvline(current_x, color=INK_SECONDARY, lw=1, ls=":")
        marker_note(ax, current_x, np.nanmin(y), "Current design")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    title(ax, f"{y_label} against {x_label.lower()}")
    return fig

