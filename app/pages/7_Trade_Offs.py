"""Generic trade-off explorer: any input against any output, plus waveform comparison."""

import streamlit as st

from common import finish_page, scenario, show, start_page
from bistatic_lb import plots
from bistatic_lb.pipeline import run_link_budget
from bistatic_lb.sweeps import SWEEPABLE, sweep, sweep_values

start_page("Trade-offs", "Pick one input to vary and one output to watch. The current design is unchanged.")

r = finish_page()
s = scenario()

c1, c2 = st.columns(2)
with c1:
    path = st.selectbox("Vary", list(SWEEPABLE), format_func=lambda k: SWEEPABLE[k][0])
with c2:
    metric = st.selectbox("Watch", list(r.metrics), index=list(r.metrics).index("Target SINR (dB)"))
label, default_start, default_stop, is_integer = SWEEPABLE[path]
c1, c2, c3 = st.columns(3)
step = 1 if is_integer else None
start = c1.number_input("From", value=int(default_start) if is_integer else float(default_start), step=step, key=f"start::{path}")
stop = c2.number_input("To", value=int(default_stop) if is_integer else float(default_stop), step=step, key=f"stop::{path}")
points = c3.slider("Points", 5, 41, 15)

if st.button("Run sweep", type="primary"):
    if stop <= start:
        st.error("'To' must be larger than 'From'.")
    else:
        bar = st.progress(0.0)
        x, y = sweep(s, path, sweep_values(path, start, stop, points), metric, on_progress=bar.progress)
        bar.empty()
        node = s
        for key in path.split("."):
            node = getattr(node, key)
        show(plots.sweep_line(x, y, label, metric, float(node)))
        skipped = int((y != y).sum())
        if skipped:
            st.warning(f"{skipped} point(s) were outside the model's valid input range and were skipped.")
        st.dataframe([{label: float(a), metric: float(b)} for a, b in zip(x, y)], width="stretch", hide_index=True)

st.subheader("Waveform comparison")
st.write("The same scenario with each waveform. Bandwidth settings apply to pulsed LFM and to noise-like codes.")
rows = []
names = {"pulsed_lfm": "Pulsed LFM", "phase_coded": "Phase-coded", "zadoff_chu": "Zadoff-Chu", "fm": "FM", "dab": "DAB", "dvbt": "DVB-T"}
for kind, name in names.items():
    x = run_link_budget(s.with_value("waveform.kind", kind))
    rows.append(
        {
            "Waveform": name,
            "Bandwidth (MHz)": round(x.waveform.bandwidth_hz / 1e6, 3),
            "Processing gain (dB)": round(x.waveform.processing_gain_db, 1),
            "PA saturated (W)": round(x.frontend.pa_saturated_w or 0.0),
            "Gain window (dB)": round(x.frontend.gain_window_db, 1),
            "Digital cancel needed (dB)": round(x.frontend.digital_cancellation_needed_db, 1),
            "SINR (dB)": round(x.link.sinr_db, 1),
            "Pd": round(x.detection.pd, 3),
            "Range sigma (m)": round(x.estimation.sigma_bistatic_range_m, 2),
            "Range-rate sigma (m/s)": round(x.estimation.sigma_range_rate_m_s, 4),
        }
    )
st.dataframe(rows, width="stretch", hide_index=True)
