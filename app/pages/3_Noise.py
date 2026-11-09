"""Stage 1: noise estimation."""

import streamlit as st

from common import STATE_KEY, WIDGET_PREFIX, applicability_table, choice, finish_page, number, scenario, show, start_page, toggle
from bistatic_lb import plots

start_page("Stage 1: Noise", "External noise from ITU-R P.372, integrated over the array pattern, plus the receiver's own noise.")

inputs, outputs = st.columns([1, 2])
with inputs:
    choice(
        "environment.man_made_category", "Man-made noise environment",
        {"city": "City", "residential": "Residential", "rural": "Rural", "quiet_rural": "Quiet rural"},
    )
    override_path = "environment.man_made_override_db"
    has_override = scenario().environment.man_made_override_db is not None
    use_measured = st.checkbox("Use a measured site value", value=has_override)
    if use_measured and not has_override:
        st.session_state[STATE_KEY] = scenario().with_value(override_path, 0.0).to_yaml()
    if not use_measured and has_override:
        st.session_state[STATE_KEY] = scenario().with_value(override_path, None).to_yaml()
        st.session_state.pop(WIDGET_PREFIX + override_path, None)
    if use_measured:
        number(override_path, "Measured Fa at this frequency (dB)", step=1.0)
    choice(
        "environment.noise_design_point", "Design the link for",
        {"median": "Median noise", "upper_decile": "Upper-decile noise (met about 90% of the time and locations)"},
    )
    use_deviations = st.checkbox(
        "Use measured decile deviations", value=scenario().environment.man_made_upper_decile_db is not None,
        help="P.372 Table 2 is an average for 0.3 to 250 MHz.",
    )
    for path, label in (("environment.man_made_upper_decile_db", "Upper decile deviation (dB)"), ("environment.man_made_lower_decile_db", "Lower decile deviation (dB)")):
        stored = getattr(scenario().environment, path.split(".")[1])
        if use_deviations and stored is None:
            st.session_state[STATE_KEY] = scenario().with_value(path, 3.0).to_yaml()
        if not use_deviations and stored is not None:
            st.session_state[STATE_KEY] = scenario().with_value(path, None).to_yaml()
            st.session_state.pop(WIDGET_PREFIX + path, None)
        if use_deviations:
            number(path, label, min_value=0.0, step=1.0)
    number("environment.ground_temperature_k", "Ground temperature (K)", min_value=100.0, step=10.0)
    toggle("environment.sun_in_beam", "Sun in the beam")
    st.caption("Receiver stages are on the Direct Path & Front-End page.")

r = finish_page()
with outputs:
    n = r.noise
    a, b, c, d = st.columns(4)
    a.metric("System temperature", f"{n.system_output_k:.0f} K")
    b.metric("Noise density", f"{n.noise_output_dbm_hz:.1f} dBm/Hz")
    c.metric("Receiver NF", f"{r.lineup.noise_figure_db:.2f} dB")
    d.metric("Dominant source", n.dominant_source.capitalize())
    st.caption(n.array_noise_gain_note)
    show(plots.noise_vs_frequency(r.scenario))

applicability_table(n.contributions, "Every noise source considered, including those rejected, and why.")

st.subheader("Noise spread: P.372-18 Part 7")
stats = n.statistics
det = r.detection
a, b, c, d = st.columns(4)
a.metric("Lower decile", f"{stats.system_lower_k:.0f} K")
b.metric("Sum of medians", f"{stats.system_sum_of_medians_k:.0f} K")
c.metric("Upper decile", f"{stats.system_upper_k:.0f} K", f"+{stats.upper_margin_db:.1f} dB", delta_color="off")
d.metric("Pd at upper decile", f"{det.pd_upper_decile:.3f}", f"{det.margin_upper_decile_db:+.1f} dB margin")
st.write(
    "Man-made and galactic noise vary, combined by Part 7. Receiver, ground and atmosphere noise are held constant. "
    f"Man-made deviations used: +{stats.man_made_upper_deviation_db:.1f} dB and -{stats.man_made_lower_deviation_db:.1f} dB."
)
if stats.note:
    st.warning(stats.note)
show(plots.noise_spread_vs_frequency(r.scenario))
