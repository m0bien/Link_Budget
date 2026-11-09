"""Stage 2: the direct-path signal, and what it demands of the RF front-end."""

import numpy as np
import streamlit as st

from common import STATE_KEY, applicability_table, choice, finish_page, number, requirements_table, scenario, show, start_page, toggle
from bistatic_lb import plots
from bistatic_lb.frontend.levels import levels_for
from bistatic_lb.propagation import profile_from_csv

start_page(
    "Stage 2: Direct path and RF front-end",
    "The direct signal is normally the strongest thing the receiver sees. It sets the ADC, limiter, LNA and AGC requirements.",
)

prop, chain, adc = st.columns(3)
with prop:
    st.subheader("Propagation")
    number("environment.refractivity_gradient_n_per_km", "Refractivity gradient (N/km)", step=10.0, help="-40 is standard. Below -157 forms a duct.")
    choice("environment.ground_type", "Ground", {"sea": "Sea", "wet": "Wet ground", "medium": "Medium ground", "dry": "Dry ground"})
    number("environment.roughness_rms_m", "Surface roughness, rms (m)", min_value=0.0, step=0.5)
    number("environment.rain_rate_mm_h", "Rain rate R0.01 (mm/h)", min_value=0.0, step=5.0)
    number("environment.fade_time_percent", "Time percentage for fades (%)", min_value=0.001, max_value=50.0, step=0.01, fmt="%.3f")
    number("environment.log10_k_geoclimatic", "P.530 log10(K)", step=0.5, help="From the ITU LogK.csv grid or local measurements. Not bundled here.")
    number("environment.dn75", "P.530 dN75", min_value=0.0, step=5.0, help="From the ITU dN75.csv grid. Not bundled here.")
    toggle("environment.include_ducting_in_maximum", "Allow for ducting in the maximum level")
    toggle("environment.tx_in_clutter", "Transmitter is inside clutter")
    toggle("environment.rx_in_clutter", "Receiver is inside clutter")
    number("environment.pressure_hpa", "Pressure (hPa)", min_value=100.0, step=10.0)
    number("environment.water_vapour_g_m3", "Water vapour (g/m^3)", min_value=0.0, step=1.0)

    uploaded = st.file_uploader("Terrain profile, CSV: distance_km, height_m from Tx", type=["csv", "txt"])
    if uploaded is not None:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as handle:
            handle.write(uploaded.getvalue().decode("utf-8"))
        points = profile_from_csv(handle.name)
        if len(points) >= 2 and st.button("Use this profile"):
            st.session_state[STATE_KEY] = scenario().with_value("environment.terrain_profile", points).to_yaml()
            st.rerun()
    if scenario().environment.terrain_profile and st.button("Clear profile (smooth Earth)"):
        st.session_state[STATE_KEY] = scenario().with_value("environment.terrain_profile", []).to_yaml()
        st.rerun()

with chain:
    st.subheader("Receiver channel")
    number("frontend.feed_loss_db", "Feed loss (dB)", min_value=0.0, step=0.1)
    number("frontend.limiter_loss_db", "Limiter loss (dB)", min_value=0.0, step=0.1)
    number("frontend.rf_filter_loss_db", "RF filter loss (dB)", min_value=0.0, step=0.1)
    number("frontend.lna_gain_db", "LNA gain (dB)", step=1.0)
    number("frontend.lna_nf_db", "LNA noise figure (dB)", min_value=0.0, step=0.1)
    number("frontend.lna_iip3_dbm", "LNA IIP3 (dBm)", step=1.0)
    number("frontend.lna_input_p1db_dbm", "LNA input P1dB (dBm)", step=1.0)
    number("frontend.lna_survival_dbm", "LNA survival level (dBm)", step=1.0)
    number("frontend.mixer_loss_db", "Mixer loss (dB)", min_value=0.0, step=0.5)
    number("frontend.mixer_iip3_dbm", "Mixer IIP3 (dBm)", step=1.0)
    number("frontend.if_gain_db", "IF gain (dB)", step=1.0)
    number("frontend.if_nf_db", "IF noise figure (dB)", min_value=0.0, step=0.5)
    number("frontend.if_iip3_dbm", "IF IIP3 (dBm)", step=1.0)
    number("frontend.lo_phase_noise_dbc_hz", "LO phase noise at target Doppler (dBc/Hz)", step=5.0)

with adc:
    st.subheader("ADC and cancellation")
    number("frontend.adc_bits", "ADC bits", min_value=4, max_value=24, step=1, integer=True)
    number("frontend.adc_enob_loss_bits", "Bits lost to ENOB", min_value=0.0, step=0.5)
    number("frontend.adc_full_scale_dbm", "Full scale (dBm)", step=1.0)
    number("frontend.adc_oversampling", "Sample rate / bandwidth", min_value=1.0, step=0.5)
    number("frontend.adc_headroom_db", "Headroom beyond PAPR (dB)", min_value=0.0, step=1.0)
    number("frontend.analog_canceller_db", "Analog canceller (dB), before the ADC", min_value=0.0, step=5.0)
    number("frontend.digital_canceller_limit_db", "Digital cancellation limit (dB), after the ADC", min_value=0.0, step=5.0)
    number("frontend.residual_below_noise_db", "Residual below integrated noise (dB)", min_value=0.0, step=1.0)
    choice(
        "frontend.gain_control", "Gain control",
        {"auto": "Automatic: gated for pulsed waveforms", "continuous": "Continuous: one AGC gain for all time", "gated": "Gated around the direct pulse"},
        help="Gated gain drops for the direct pulse and restores while listening. Only pulsed waveforms can use it, "
        "because continuous waveforms always carry the direct signal.",
    )
    if scenario().waveform.kind == "pulsed_lfm":
        number("frontend.gain_switch_time_s", "Gain switch settling time (us)", scale=1e-6, min_value=0.0, step=0.5,
               help="The receiver is blind for the pulse plus this time, which eclipses the closest echoes.")

r = finish_page()
if r.scenario.frontend.gain_control == "gated" and r.frontend.gain_control != "gated":
    st.warning(f"Gated gain needs a pulsed waveform. {r.waveform.name} carries the direct signal all the time, so continuous gain is used.")
st.divider()
d, f = r.direct, r.frontend
a, b, c, e = st.columns(4)
a.metric("Maximum, per element", f"{d.element_maximum_dbm:.1f} dBm")
b.metric("Nominal, per element", f"{d.element_nominal_dbm:.1f} dBm")
c.metric("Minimum, per element", f"{d.element_minimum_dbm:.1f} dBm")
e.metric("After beamforming", f"{d.output_nominal_dbm:.1f} dBm", f"{d.array_gain_toward_tx_dbi:.1f} dBi toward Tx", delta_color="off")
st.caption(d.maximum_note)

st.subheader("Noise above quantization")
a, b, c, e = st.columns(4)
a.metric("Instantaneous range", f"{f.instantaneous_dynamic_range_db:.1f} dB", f"{f.effective_bits:.1f} effective bits", delta_color="off")
b.metric("Back-off for peaks", f"{f.clipping_backoff_db:.1f} dB", "PAPR plus headroom", delta_color="off")
c.metric("Direct-to-noise at ADC", f"{f.direct_to_noise_at_adc_db:.1f} dB", "Maximum, after analog suppression", delta_color="off")
e.metric("Noise above quantization", f"{f.noise_over_quantization_db:.1f} dB", f"{f.noise_over_quantization_nominal_db:.1f} dB in operation", delta_color="off")
st.caption(
    "Computed, not entered: with the strongest direct signal plus back-off placed at full scale, noise sits "
    "range - back-off - direct-to-noise above quantization. "
    f"Quantization costs {f.sensitivity_loss_db:.2f} dB at the nominal direct level and {f.worst_sensitivity_loss_db:.2f} dB at the maximum."
)
bits_text = "no resolution is enough until the link closes" if not np.isfinite(f.required_adc_bits) else f"{f.required_adc_bits:.1f} ADC bits"
if f.shortfall_db > 0:
    st.error(
        f"Noise stays {f.shortfall_db:.1f} dB below quantization even at the highest gain that avoids clipping. Add about "
        f"{f.shortfall_db / 6.02:.1f} more effective bits, or fit {f.shortfall_db:.1f} dB of suppression before the ADC. "
        "A digital null cannot help, because the ADC sees the signal first."
    )
elif f.required_adc_bits > r.scenario.frontend.adc_bits:
    st.warning(f"Detection needs {bits_text}: quantization at the maximum direct level uses up more than the detection margin.")
elif f.listening_gain_db is not None:
    st.success(
        f"Gated gain: {f.pulse_gain_db:.1f} dB while the direct pulse is present, {f.listening_gain_db:.1f} dB while listening, "
        f"so the target sees noise {f.noise_over_quantization_nominal_db:.1f} dB above quantization. Detection needs {bits_text}. "
        f"The receiver is blind for the pulse plus {f.blind_extra_s * 1e6:.1f} us after it."
    )
else:
    st.success(
        f"Gain window {f.gain_min_db:.1f} to {f.gain_max_db:.1f} dB. Configured gain {f.configured_gain_db:.1f} dB. "
        f"Detection needs {bits_text}, with {f.allowed_quantization_loss_db:.1f} dB of margin available for quantization."
    )

requirements_table(f.requirements)

st.subheader("Levels through the receiver")
st.caption(
    "Left: the strongest direct signal and the noise after each stage, at the configured gains. "
    "The AGC is modelled as an ideal attenuator just before the ADC, so earlier stages always see the full gain and can compress. "
    "Right: levels at the ADC input once the AGC has set the gain for the nominal direct level. "
    "With gated gain (pulsed waveforms) that cut lasts only for the direct pulse: between pulses the gain returns, "
    "so the noise at the ADC rises again, shown as the diamond on the left and the extra line on the right. "
    "Clipping is checked, not simulated: a level in the shaded region means the AGC must cut further, and the extra "
    "quantization loss that causes is counted in the detection margin."
)
plot1, plot2 = st.columns(2)
with plot1:
    show(plots.lineup_levels(r))
with plot2:
    show(plots.dynamic_range_ladder(r))
levels = levels_for(r)
for p in levels.compressing:
    st.warning(
        f"{p.name} compresses: peak output {p.peak_dbm:.1f} dBm against a 1 dB compression point of {p.p1db_out_dbm:.1f} dBm"
        + (" (estimated as IIP3 - 9.6 dB)." if p.p1db_estimated else ".")
    )
if levels.agc_cut_db > 0.05:
    st.info(
        f"At the maximum direct level the AGC removes {levels.agc_cut_db:.1f} dB of the {levels.configured_gain_db:.1f} dB configured gain. "
        f"Without it the mean direct level would be {levels.overshoot_without_agc_db:.1f} dB over the clipping limit."
    )

plot1, plot2 = st.columns(2)
with plot1:
    show(plots.gain_window_vs_bits(r))
with plot2:
    show(plots.direct_levels_vs_baseline(r.scenario))

st.subheader("Receiver line-up")
st.dataframe(r.lineup.table(), width="stretch", hide_index=True)
st.write(f"Cascade NF {r.lineup.noise_figure_db:.2f} dB, gain {r.lineup.gain_db:.1f} dB, IIP3 {r.lineup.iip3_dbm:.1f} dBm.")

applicability_table(d.terms.contributions, "Every propagation mechanism considered on the direct path, and why it was used or rejected.")
