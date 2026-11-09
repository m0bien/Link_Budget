"""Receiver array design and its trade-offs."""

import numpy as np
import streamlit as st

from common import choice, finish_page, number, show_plot, start_page, toggle
from bistatic_lb.antenna.arrays import array_from_scenario

start_page("Antenna array", "A ring of dipoles at the receiver, with an optional central column for elevation.")

inputs, outputs = st.columns([1, 2])
with inputs:
    choice("array.element", "Element", {"dipole": "Half-wave dipole", "folded_dipole": "Folded dipole (wider band, same pattern)"})
    choice(
        "array.orientation", "Dipole orientation",
        {"vertical": "Vertical (matches the monopole)", "horizontal": "Horizontal (cross-polarized to the monopole)"},
        help="Horizontal dipoles reject the direct path, but also the target unless it depolarizes.",
    )
    number("array.n_circular", "Ring elements", min_value=1, max_value=128, step=1, integer=True)
    number("array.radius_wavelengths", "Ring radius (wavelengths at design frequency)", min_value=0.05, step=0.05)
    number("array.n_vertical", "Central column elements", min_value=0, max_value=64, step=1, integer=True, help="Two or more adds elevation estimation.")
    number("array.vertical_spacing_wavelengths", "Column spacing (wavelengths)", min_value=0.1, step=0.05)
    number("array.height_m", "Array height (m)", min_value=1.0, step=5.0)
    choice("array.taper", "Taper", {"uniform": "Uniform", "cardioid": "Cardioid (lower sidelobes, wider beam)"})
    choice(
        "array.digitization", "Digitization",
        {"per_element": "One ADC per element", "analog_subarray": "Analog subarray with RF null"},
        help="Only an analog null reduces the signal reaching the ADC.",
    )
    toggle("array.null_on_transmitter", "Steer a null to the transmitter")
    number("array.calibration_error_deg", "Channel calibration error, rms (deg)", min_value=0.0, step=0.5)
    number("array.analog_null_depth_db", "Analog null depth (dB)", min_value=0.0, step=5.0)
    number("array.cross_pol_isolation_db", "Cross-polarization isolation (dB)", min_value=0.0, step=5.0)
    number("array.beam_straddle_loss_db", "Beam straddle loss (dB)", min_value=0.0, step=0.5)

r = finish_page()
with outputs:
    a = r.antenna
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gain to target", f"{a.gain_toward_target_dbi:.1f} dBi")
    c2.metric("Null depth on Tx", f"{a.null_depth_db:.1f} dB")
    c3.metric("Azimuth beamwidth", f"{a.beamwidth_az_deg:.1f} deg")
    c4.metric("Peak sidelobe", f"{a.peak_sidelobe_db:.1f} dB")
    c1, c2, c3 = st.columns(3)
    c1.metric("Element spacing", f"{a.max_spacing_wavelengths:.2f} wavelengths")
    c2.metric("Usable band", f"{a.usable_band_hz[0] / 1e6:.0f} to {a.usable_band_hz[1] / 1e6:.0f} MHz")
    c3.metric("Mismatch loss", f"{a.rx_mismatch_db:.2f} dB")

    tab_az, tab_el, tab_layout = st.tabs(["Azimuth", "Elevation", "Layout"])
    with tab_az:
        show_plot("azimuth_pattern")
    with tab_el:
        show_plot("elevation_pattern")
    with tab_layout:
        show_plot("array_layout")

st.subheader("Trade-off: element count at fixed spacing")
s = r.scenario
spacing = float(st.slider("Spacing between ring elements (wavelengths)", 0.3, 0.7, 0.45, 0.05))
counts = np.array([4, 6, 8, 12, 16, 24, 32, 48])
rows = []
for n in counts:
    radius = spacing / (2.0 * np.sin(np.pi / n))
    trial = s.with_value("array.n_circular", int(n)).with_value("array.radius_wavelengths", float(radius))
    array = array_from_scenario(trial)
    look = (r.geometry.azimuth_at_rx_rad, r.geometry.elevation_at_rx_rad)
    w = array.beamformer(*look)
    rows.append(
        {
            "Elements": int(n),
            "Radius (wavelengths)": round(float(radius), 2),
            "Gain (dBi)": round(float(array.gain_dbi(w, *look)), 1),
            "Beamwidth (deg)": round(array.beamwidth_az_deg(w, *look), 1),
            "Peak sidelobe (dB)": round(array.peak_sidelobe_db(w, *look), 1),
            "ADC channels": int(n) + s.array.n_vertical,
        }
    )
st.dataframe(rows, width="stretch", hide_index=True)
st.caption("Mutual coupling is neglected, which matters most for close spacing and deep nulls.")
