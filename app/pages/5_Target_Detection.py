"""Stage 3: target signal, detection and parameter estimation."""

import numpy as np
import streamlit as st

from common import applicability_table, finish_page, number, show, show_plot, start_page, toggle
from bistatic_lb import plots
from bistatic_lb.target.montecarlo import delay_estimator_rmse, detection_monte_carlo
from bistatic_lb.units import undb


def absent(level_dbm: float) -> str:
    return "Does not apply" if level_dbm <= -299.0 else f"{level_dbm:.1f} dBm"

start_page("Stage 3: Target and detection", "Fluctuating-target detection at the achieved SINR, and how accurately parameters can be estimated.")

inputs, outputs = st.columns([1, 2])
with inputs:
    number("detection.required_pd", "Required Pd", min_value=0.01, max_value=0.999, step=0.05)
    number("detection.false_alarms_per_s", "False alarms per second", min_value=1e-6, step=0.01, fmt="%.4f")
    number("detection.max_bistatic_range_km", "Maximum bi-static range searched (km)", min_value=1.0, step=10.0)
    number("detection.max_speed_m_s", "Maximum target speed searched (m/s)", min_value=1.0, step=50.0)
    number("detection.noncoherent_looks", "Non-coherent looks", min_value=1, max_value=256, step=1, integer=True)
    number("detection.cfar_reference_cells", "CFAR reference cells", min_value=0, step=4, integer=True, help="0: fixed threshold with no CFAR loss.")
    number("detection.processing_loss_db", "Processing loss (dB)", min_value=0.0, step=0.5)
    toggle("environment.radar_clutter_enabled", "Include ground clutter")
    number("environment.clutter_gamma_db", "Clutter gamma (dB)", step=1.0)

r = finish_page()
l, det, e = r.link, r.detection, r.estimation
with outputs:
    a, b, c, d = st.columns(4)
    a.metric("SINR", f"{l.sinr_db:.1f} dB", f"SNR {l.snr_db:.1f} dB", delta_color="off")
    b.metric("Pd", f"{det.pd:.3f}", f"{det.margin_db:+.1f} dB margin")
    c.metric("Pfa per cell", f"{det.pfa:.2e}")
    d.metric("Rt x Rr limit", f"{det.max_range_product_km2:.0f} km^2", "at required Pd", delta_color="off")
    st.dataframe(
        [
            {"Term": "EIRP toward target", "Value": f"{l.eirp_toward_target_dbw:.1f} dBW"},
            {"Term": "Receive gain", "Value": f"{l.rx_gain_dbi:.1f} dBi"},
            {"Term": "Target RCS", "Value": f"{l.rcs_dbsm:.1f} dBsm"},
            {"Term": "Excess path loss: Tx side, Rx side", "Value": f"{l.excess_tx_db:.1f} dB, {l.excess_rx_db:.1f} dB"},
            {"Term": "System losses", "Value": f"{l.total_losses_db:.1f} dB"},
            {"Term": "of which migration: range, Doppler", "Value": f"{l.migration.range_loss_db:.2f} dB, {l.migration.doppler_loss_db:.2f} dB ({l.migration.range_cells:.2f} and {l.migration.doppler_cells:.2f} cells crossed)"},
            {"Term": "Received power", "Value": f"{l.received_dbm:.1f} dBm"},
            {"Term": "Noise in the cell", "Value": f"{l.noise_cell_dbm:.1f} dBm"},
            {"Term": "Direct-path residual in the cell", "Value": f"{l.direct_cell_dbm:.1f} dBm"},
            {"Term": "Clutter in the cell", "Value": f"{l.clutter_cell_dbm:.1f} dBm"},
            {"Term": "LO phase noise of the direct signal in the cell", "Value": absent(l.phase_noise_cell_dbm)},
            {"Term": "Third-order distortion of the direct signal in the cell", "Value": absent(l.intermod_cell_dbm)},
            {"Term": "Required SINR (including CFAR loss)", "Value": f"{det.required_snr_db:.1f} dB ({det.cfar_loss_db:.2f} dB)"},
            {"Term": "At upper-decile noise: SINR, Pd, margin", "Value": f"{det.sinr_upper_decile_db:.1f} dB, {det.pd_upper_decile:.3f}, {det.margin_upper_decile_db:+.1f} dB"},
        ],
        width="stretch",
        hide_index=True,
    )

tab_pd, tab_range, tab_map = st.tabs(["Pd against SINR", "Pd against range", "Detection map"])
with tab_pd:
    show_plot("pd_vs_snr")
    if st.button("Check this design's Pd and Pfa by simulation"):
        n = r.scenario.detection.noncoherent_looks
        check_pfa = max(det.pfa, 1e-4)
        sinr = l.sinr_db - det.cfar_loss_db
        pd_mc, pfa_mc = detection_monte_carlo(sinr, check_pfa, n, det.fluctuation, trials=500_000, seed=r.scenario.monte_carlo.seed)
        from bistatic_lb.target.detection import detection_probability

        st.write(
            f"At Pfa {check_pfa:.1e}: simulated Pd {pd_mc:.4f}, analytic {detection_probability(float(undb(sinr)), check_pfa, n, det.fluctuation):.4f}; "
            f"simulated Pfa {pfa_mc:.2e}. Pfa is raised to 1e-4 if needed, so the simulation can resolve it."
        )
with tab_range:
    show_plot("pd_vs_range")
with tab_map:
    if st.button("Compute detection map (a few seconds)"):
        with st.spinner("Evaluating the link across the Tx-Rx plane"):
            show_plot("detection_map")

st.subheader("Estimation accuracy")
if not e.bound_valid:
    st.warning(f"SINR {e.snr_db:.1f} dB is below the estimated {e.threshold_snr_db:.1f} dB threshold, so these bounds are optimistic.")
left, right = st.columns([1, 1])
with left:
    st.dataframe(
        [
            {"Parameter": "Bi-static range", "Sigma": f"{e.sigma_bistatic_range_m:.2f} m", "Resolution": f"{e.range_resolution_m:.0f} m along bisector"},
            {"Parameter": "Doppler", "Sigma": f"{e.sigma_doppler_hz:.3f} Hz", "Resolution": f"{e.doppler_resolution_hz:.2f} Hz"},
            {"Parameter": "Range rate", "Sigma": f"{e.sigma_range_rate_m_s:.3f} m/s", "Resolution": "-"},
            {"Parameter": "Azimuth", "Sigma": f"{e.sigma_azimuth_deg:.2f} deg", "Resolution": f"{r.antenna.beamwidth_az_deg:.1f} deg beam"},
            {"Parameter": "Elevation", "Sigma": "Needs a column" if e.sigma_elevation_deg is None else f"{e.sigma_elevation_deg:.2f} deg", "Resolution": "-"},
            {"Parameter": "Position, rms", "Sigma": f"{e.position_rms_m:.0f} m", "Resolution": "-"},
            {"Parameter": "RCS", "Sigma": f"{e.rcs_measurement_std_db:.2f} dB noise; {e.rcs_total_std_db:.2f} dB with fluctuation", "Resolution": "-"},
        ],
        width="stretch",
        hide_index=True,
    )
with right:
    simulated = st.session_state.get("delay_mc")
    if st.button("Overlay a simulated estimator"):
        with st.spinner("Simulating"):
            snr_axis = np.arange(0.0, 31.0, 3.0)
            simulated = (snr_axis, delay_estimator_rmse(snr_axis, seed=r.scenario.monte_carlo.seed))
            st.session_state["delay_mc"] = simulated
    show(plots.range_accuracy_vs_snr(r, *(simulated or (None, None))))

applicability_table([l.rcs_contribution] + l.contributions_tx, "Tx-to-target path.")
applicability_table(l.contributions_rx, "Target-to-Rx path.")
