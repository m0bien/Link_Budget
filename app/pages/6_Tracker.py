"""Stage 4: what tracking adds to Pd, Pfa and accuracy."""

import streamlit as st

from common import choice, finish_page, number, show, start_page
from bistatic_lb import plots
from bistatic_lb.tracker import run_tracker_monte_carlo

start_page("Stage 4: Tracker", "M-of-N confirmation allows a lower threshold, and filtering improves accuracy. The price is delay.")

inputs, outputs = st.columns([1, 2])
with inputs:
    number("tracker.confirm_m", "Confirm: M detections", min_value=1, max_value=12, step=1, integer=True)
    number("tracker.confirm_n", "Confirm: in N scans", min_value=1, max_value=12, step=1, integer=True)
    number("tracker.update_interval_s", "Update interval (s)", min_value=0.01, step=0.5)
    number("tracker.target_confirm_probability", "Required confirmation probability", min_value=0.01, max_value=0.999, step=0.01)
    number("tracker.allowed_false_tracks_per_hour", "Allowed false tracks per hour", min_value=1e-4, step=0.5, fmt="%.3f")
    number("tracker.process_accel_m_s2", "Target acceleration, rms (m/s^2)", min_value=0.01, step=0.5)
    number("tracker.gate_sigma", "Gate size (sigma)", min_value=1.0, step=0.5)
    choice("tracker.association", "Association", {"pda": "Probabilistic data association", "nearest": "Nearest neighbour"})
    st.divider()
    number("monte_carlo.runs", "Monte Carlo runs", min_value=1, step=50, integer=True)
    number("monte_carlo.scans", "Scans per run", min_value=5, step=10, integer=True)
    number("monte_carlo.seed", "Random seed", step=1, integer=True)

r = finish_page()
t = r.tracker
with outputs:
    a, b, c, d = st.columns(4)
    a.metric("Threshold gain", f"{t.trade.gain_db:.1f} dB")
    b.metric("Relaxed per-look Pfa", f"{t.trade.best_pfa:.1e}")
    c.metric("False tracks", f"{t.false_tracks_per_hour:.2g} per hour")
    d.metric("Time to confirm", f"{t.mean_time_to_confirm_s:.1f} s")
    st.dataframe(
        [
            {"Measure": "Pd (per look / confirmed)", "Single look": f"{t.single_look_pd:.3f}", "With tracker": f"{t.relaxed_pd:.3f} / {t.confirm_probability_at_relaxed_threshold:.3f}"},
            {"Measure": "False reports per hour", "Single look": f"{t.single_look_false_alarms_per_hour:.3g} false alarms", "With tracker": f"{t.false_tracks_per_hour:.3g} false tracks"},
            {"Measure": "Bi-static range sigma, steady state", "Single look": f"{t.single_look_sigma_range_m:.2f} m", "With tracker": f"{t.tracked_sigma_range_m:.2f} m ({t.range_improvement_db:+.1f} dB)"},
            {"Measure": "Range-rate sigma, steady state", "Single look": f"{t.single_look_sigma_rate_m_s:.3f} m/s", "With tracker": f"{t.tracked_sigma_rate_m_s:.3f} m/s ({t.rate_improvement_db:+.1f} dB)"},
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(t.latency_note)

left, right = st.columns(2)
with left:
    show(plots.threshold_trade_snr(r))
with right:
    show(plots.confirmation_curves(r))

st.subheader("Simulation")
st.write("Runs the filters on detections drawn from the link budget along the target's path. The report includes the latest run.")
if st.button("Run tracker Monte Carlo"):
    bar = st.progress(0.0, "Running")
    st.session_state["tracker_mc"] = run_tracker_monte_carlo(r.analyzer, r.tracker.trade.best_pfa, on_progress=lambda f: bar.progress(f, "Running"))
    bar.empty()

mc = st.session_state.get("tracker_mc")
if mc is not None:
    a, b, c = st.columns(3)
    a.metric("Track continuity", f"{mc.continuity:.1%}")
    b.metric("Runs lost", f"{mc.lost_runs} of {mc.runs}")
    c.metric("Per-look Pfa used", f"{mc.pfa_per_cell:.1e}")
    tab_r, tab_v, tab_p = st.tabs(["Range", "Range rate", "Position"])
    with tab_r:
        show(plots.tracker_rmse(mc, "range"))
    with tab_v:
        show(plots.tracker_rmse(mc, "rate"))
    with tab_p:
        show(plots.tracker_rmse(mc, "position"))
