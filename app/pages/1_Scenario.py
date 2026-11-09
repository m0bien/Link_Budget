"""Geometry, transmitter, waveform and target inputs."""

import numpy as np
import streamlit as st

from common import STATE_KEY, choice, finish_page, number, scenario, start_page

start_page("Scenario", "Set the geometry, transmitter, waveform and target. Later pages use these values.")

left, middle, right = st.columns(3)
with left:
    st.subheader("General")
    current_name = scenario().name
    new_name = st.text_input("Scenario name", value=current_name)
    if new_name != current_name:
        st.session_state[STATE_KEY] = scenario().with_value("name", new_name).to_yaml()
    number("frequency_hz", "Frequency (MHz)", scale=1e6, min_value=50.0, max_value=18000.0, step=10.0)

    st.subheader("Geometry")
    number("geometry.baseline_km", "Baseline (km)", min_value=0.1, step=1.0)
    number("geometry.target_range_rx_km", "Target ground range from receiver (km)", min_value=0.1, step=1.0)
    number("geometry.target_azimuth_deg", "Target azimuth from the Tx direction (deg)", step=5.0)
    number("geometry.target_altitude_m", "Target altitude (m)", min_value=0.0, step=100.0)
    number("geometry.target_speed_m_s", "Target speed (m/s)", min_value=0.0, step=10.0)
    number("geometry.target_heading_deg", "Target heading (deg)", step=10.0)
    number("geometry.minimum_target_range_km", "Closest target to receiver (km)", min_value=0.01, step=0.5, help="Checked in case a close target exceeds the direct signal.")

with middle:
    st.subheader("Transmitter: monopole")
    number("transmitter.power_w", "Mean envelope power (W)", min_value=1.0, step=100.0, help="Power while transmitting. The expected range is 100 W to 10 kW.")
    number("transmitter.height_m", "Antenna height (m)", min_value=1.0, step=5.0)
    number("transmitter.line_loss_db", "Line loss (dB)", min_value=0.0, step=0.1)
    ground = choice("transmitter.ground_plane", "Ground plane", {"ideal": "Ideal", "finite": "Finite (approximate correction)"})
    if ground == "finite":
        number("transmitter.finite_ground_tilt_deg", "Main-lobe elevation (deg)", min_value=1.0, max_value=80.0, step=5.0)
        number("transmitter.finite_ground_horizon_loss_db", "Horizon gain reduction (dB)", min_value=0.0, step=1.0)

    st.subheader("Target")
    number("target.rcs_dbsm", "RCS (dBsm)", step=1.0)
    rcs_model = choice(
        "target.rcs_model", "RCS model",
        {"constant": "Entered value", "equivalence": "Monostatic-bistatic equivalence", "forward_scatter": "Add forward-scatter lobe"},
    )
    if rcs_model == "forward_scatter":
        number("target.forward_scatter_area_m2", "Silhouette area (m^2)", min_value=0.01, step=1.0)
    fluctuation = choice(
        "target.fluctuation", "Fluctuation model",
        {
            "swerling0": "Swerling 0 (steady)",
            "swerling1": "Swerling 1 (slow, many scatterers)",
            "swerling2": "Swerling 2 (fast, many scatterers)",
            "swerling3": "Swerling 3 (slow, one dominant)",
            "swerling4": "Swerling 4 (fast, one dominant)",
            "chi2_scan": "Chi-square, any order (slow)",
            "rician": "Rician (dominant plus diffuse)",
        },
    )
    if fluctuation == "chi2_scan":
        number("target.chi2_dof_m", "Order m (degrees of freedom / 2)", min_value=0.1, step=0.5)
    if fluctuation == "rician":
        number("target.rician_k_db", "K factor (dB)", step=1.0)

with right:
    st.subheader("Waveform")
    kind = choice(
        "waveform.kind", "Type",
        {"pulsed_lfm": "Pulsed LFM", "phase_coded": "Noise-like: random phase code", "zadoff_chu": "Noise-like: Zadoff-Chu", "fm": "FM broadcast", "dab": "DAB (Mode I)", "dvbt": "DVB-T (8k)"},
    )
    number("waveform.cpi_s", "Coherent interval (s)", min_value=0.001, step=0.05, fmt="%.3f")
    if kind in ("pulsed_lfm", "phase_coded", "zadoff_chu"):
        number("waveform.bandwidth_hz", "Bandwidth (MHz)", scale=1e6, min_value=0.001, step=0.5, fmt="%.3f")
    if kind == "pulsed_lfm":
        number("waveform.pulse_width_s", "Pulse width (us)", scale=1e-6, min_value=0.1, step=10.0)
        number("waveform.prf_hz", "PRF (Hz)", min_value=1.0, step=100.0)
    if kind == "zadoff_chu":
        number("waveform.pulse_width_s", "Code period (us)", scale=1e-6, min_value=0.1, step=10.0)
    if kind == "fm":
        choice("waveform.fm_programme", "Programme content", {"low": "Low: speech, quiet", "typical": "Typical: music", "high": "High: fast music"})
    if kind == "dvbt":
        choice("waveform.dvbt_channel_mhz", "Channel width", {6: "6 MHz", 7: "7 MHz", 8: "8 MHz"})
    number("waveform.window_sidelobe_db", "Range-Doppler window sidelobe (dB)", step=5.0)
    number("waveform.window_loss_db", "Window loss (dB)", min_value=0.0, step=0.1)
    choice(
        "waveform.migration_compensation", "Migration compensation",
        {"none": "None", "range": "Range (keystone or stretch processing)", "range_doppler": "Range and Doppler rate"},
        help="A target that crosses range or Doppler cells within one coherent interval loses integration gain, "
        "unless the processor follows it.",
    )

r = finish_page()
st.divider()
st.subheader("What this gives")
g = r.geometry
a, b, c, d = st.columns(4)
a.metric("Ranges Rt, Rr", f"{g.range_tx_m / 1e3:.1f}, {g.range_rx_m / 1e3:.1f} km")
b.metric("Bi-static angle", f"{np.rad2deg(g.bistatic_angle_rad):.1f} deg")
c.metric("Bi-static range", f"{g.bistatic_range_m / 1e3:.1f} km")
d.metric("Doppler", f"{r.link.doppler_hz:.0f} Hz")
a, b, c, d = st.columns(4)
a.metric("Processing gain", f"{r.waveform.processing_gain_db:.1f} dB")
b.metric("Waveform bandwidth", f"{r.waveform.bandwidth_hz / 1e6:.3g} MHz")
c.metric("PAPR", f"{r.waveform.papr_db:.1f} dB")
d.metric("Duty cycle", f"{r.waveform.duty_cycle:.2f}")
for note in r.waveform.notes():
    st.caption(note)
