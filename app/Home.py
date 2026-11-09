"""Bi-static link-budget app: overview. Run with:  streamlit run app/Home.py"""

import streamlit as st

from common import Scenario, finish_page, replace_scenario, scenario_files, start_page

start_page(
    "Bi-static radar link budget",
    "Work through the pages in order: noise floor, direct path and front-end, target detection and estimation, "
    "then tracker gain. Every model states why it was used or rejected.",
)

load, name = st.columns([2, 2])
with load:
    labels = scenario_files()
    picked = st.selectbox("Load a saved or example scenario", list(labels), format_func=lambda k: labels[k], index=None, placeholder="Choose a scenario")
    if st.button("Load scenario", disabled=picked is None):
        try:
            replace_scenario(Scenario.from_yaml(picked))
            st.rerun()
        except Exception as error:
            st.error(f"Could not load {picked.name}: {error}")
with name:
    uploaded = st.file_uploader("Or load a YAML file from elsewhere", type=["yaml", "yml"])
    if uploaded is not None and st.button("Load file"):
        try:
            replace_scenario(Scenario.from_yaml(uploaded.getvalue().decode("utf-8") + "\n"))
            st.rerun()
        except Exception as error:  # a hand-edited file can fail in many ways; show why
            st.error(f"Could not load this file: {error}")

r = finish_page()

st.subheader("Where this design stands")
a, b, c, d = st.columns(4)
a.metric("1. Noise at output", f"{r.noise.noise_output_dbm_hz:.1f} dBm/Hz", f"{r.noise.dominant_source} dominates", delta_color="off")
b.metric("2. Maximum direct signal", f"{r.direct.element_maximum_dbm:.1f} dBm", f"{r.frontend.required_adc_bits:.1f} ADC bits needed", delta_color="off")
c.metric("3. Single-look Pd", f"{r.detection.pd:.3f}", f"{r.detection.margin_db:+.1f} dB margin")
d.metric("4. Track confirmation", f"{r.tracker.confirm_probability_at_relaxed_threshold:.3f}", f"{r.tracker.false_tracks_per_hour:.2g} false tracks/h", delta_color="off")

st.dataframe(
    [{"Output": k, "Value": f"{v:.4g}"} for k, v in r.metrics.items()],
    width="stretch",
    hide_index=True,
)
st.caption("To keep this scenario, use Save scenario in the sidebar on any page.")
