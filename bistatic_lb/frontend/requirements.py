"""Stage 2b: from signal levels to RF front-end requirements.

The governing fact is the order of operations. With per-element digitization, the
direct signal reaches every ADC before any beamforming, so a digital spatial null
does nothing for ADC dynamic range. Suppression is therefore counted in two groups:

    before the ADC: analog subarray null, analog canceller (protects the ADC)
    after the ADC:  digital beamforming null, digital canceller (protects the detector)

How far thermal noise can sit above quantization noise is not a design choice; it follows
from three levels. Set the gain so the strongest direct signal, plus its peak-to-average
back-off, just reaches full scale. Then the noise lies above quantization by

    noise over quantization = instantaneous range - back-off - direct-to-noise ratio

where the direct-to-noise ratio is measured in band at the ADC input, after any analog
suppression. More gain would clip; less gain only lowers the noise toward quantization.

The gain window runs from the gain at which noise equals quantization (3 dB loss) up to
the clipping limit, so its width is the noise over quantization above. If it is negative,
no gain setting lifts the noise over quantization, and the gap is the shortfall.

Gain control. With a continuous waveform the direct signal is always present, so one AGC
gain holds for all time and the target pays the direct signal's back-off. With a pulsed
waveform the direct pulse is present only for one pulse width per repetition interval. A
gated receiver drops the gain for the pulse and restores it while listening, so the target
sees noise lifted well above quantization. The price is a blind time of one pulse width plus
the switch's settling time after each direct pulse, and stages ahead of the switch still see
the full pulse. In gated mode the gain window applies only to the blanked pulse.

How many ADC bits are needed is set by detection, not by a fixed noise-over-quantization
target: the quantization loss at the strongest direct signal must fit inside the detection
margin an ideal ADC would leave. ``with_detection_margin`` adds that requirement once the
target stage has run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..config import Scenario
from ..noise.budget import NoiseResult
from ..noise.receiver import Lineup
from ..propagation.budget import DirectPathResult
from ..units import C_M_S, db
from ..waveforms.base import Waveform
from .adc import quantization_noise_in_band_dbm, sensitivity_loss_db


CONSTANT_ENVELOPE_PAPR_DB = 0.5  # at or below this the waveform is treated as constant envelope


@dataclass(frozen=True)
class Requirement:
    item: str
    required: str
    configured: str
    met: bool | None  # None: information only
    rationale: str


@dataclass(frozen=True)
class FrontEndResult:
    requirements: list[Requirement]
    effective_bits: float
    sample_rate_hz: float
    instantaneous_dynamic_range_db: float
    noise_at_input_dbm: float
    quantization_dbm: float
    pre_adc_suppression_db: float
    gain_min_db: float
    gain_max_db: float
    gain_window_db: float
    shortfall_db: float
    noise_over_quantization_db: float  # best achievable: strongest direct signal at the clipping limit
    noise_over_quantization_nominal_db: float  # at the gain the AGC uses for the nominal direct signal
    noise_over_quantization_worst_db: float  # at the gain the AGC uses for the maximum direct signal
    direct_to_noise_at_adc_db: float  # maximum direct signal over in-band noise, after analog suppression
    clipping_backoff_db: float
    allowed_quantization_loss_db: float | None  # None until the detection margin is known
    required_effective_bits: float | None  # inf: detection fails even with an ideal ADC
    required_adc_bits: float | None
    configured_gain_db: float
    operating_gain_db: float  # with the nominal direct signal present
    gain_control: str  # "continuous" or "gated", after resolving "auto"
    listening_gain_db: float | None  # gated: gain between direct pulses, where the target is heard
    pulse_gain_db: float  # gain while the maximum direct signal is present
    blind_extra_s: float  # gated: blind time after each direct pulse, beyond the pulse itself
    agc_range_db: float
    sensitivity_loss_db: float
    worst_sensitivity_loss_db: float
    digital_cancellation_needed_db: float
    digital_cancellation_applied_db: float
    direct_residual_after_processing_dbm: float
    post_integration_noise_dbm: float
    direct_to_noise_element_db: float
    pa_saturated_w: float | None
    direct_intermod_dbc: float | None = None  # third-order distortion of the direct signal, relative to it; None if not noise-like


def compute_frontend(
    s: Scenario,
    waveform: Waveform,
    lineup: Lineup,
    noise: NoiseResult,
    direct: DirectPathResult,
    delay_s: float,
    doppler_hz: float,
) -> FrontEndResult:
    f = s.frontend
    a = s.array
    bw = waveform.bandwidth_hz
    enob = max(f.adc_bits - f.adc_enob_loss_bits, 1.0)
    fs = f.adc_oversampling * bw
    bw_db = float(db(bw))
    noise_in = noise.noise_element_dbm_hz + bw_db  # at the antenna terminals, in-band
    quant = quantization_noise_in_band_dbm(f.adc_full_scale_dbm, enob, fs, bw)
    instantaneous_dr = f.adc_full_scale_dbm - quant

    # --- Suppression that actually protects the ADC --------------------------------
    subarray_null = a.analog_null_depth_db if a.digitization == "analog_subarray" else 0.0
    pre_adc = subarray_null + f.analog_canceller_db
    backoff = waveform.papr_db + f.adc_headroom_db

    # --- Noise over quantization, from full scale, direct path and noise -------------
    direct_at_adc = direct.element_maximum_dbm - pre_adc
    direct_to_noise = direct_at_adc - noise_in
    noise_over_quant = instantaneous_dr - backoff - direct_to_noise

    # --- Gain window: from noise equal to quantization, up to clipping ---------------
    gain_min = quant - noise_in
    gain_max = f.adc_full_scale_dbm - backoff - direct_at_adc
    window = gain_max - gain_min  # equals noise_over_quant
    shortfall = max(-window, 0.0)

    # --- Operating point: AGC backs off the configured gain as needed ----------------
    configured_gain = lineup.gain_db
    gain_nominal = min(configured_gain, f.adc_full_scale_dbm - backoff - (direct.element_nominal_dbm - pre_adc))
    gain_worst = min(configured_gain, gain_max)
    gated = waveform.is_pulsed and f.gain_control in ("auto", "gated")
    if gated:
        # The target is heard between direct pulses, where only noise and echoes reach the ADC.
        listening = min(configured_gain, f.adc_full_scale_dbm - backoff - noise_in)
        noq_nominal = noq_worst = noise_in + listening - quant
        loss_nominal = loss_worst = sensitivity_loss_db(noise_in + listening, quant)
        blind_extra = f.gain_switch_time_s
        shortfall = 0.0  # an empty window only affects the blanked pulse
    else:
        listening = None
        noq_nominal = noise_in + gain_nominal - quant
        noq_worst = noise_in + gain_worst - quant
        loss_nominal = sensitivity_loss_db(noise_in + gain_nominal, quant)
        loss_worst = sensitivity_loss_db(noise_in + gain_worst, quant)
        blind_extra = 0.0
    agc_range = direct.element_maximum_dbm - direct.element_minimum_dbm

    # --- LNA and limiter ------------------------------------------------------------
    before_lna = lineup.gain_before_db("LNA")
    at_lna_max = direct.element_maximum_dbm + before_lna
    required_lna_p1db = at_lna_max + waveform.papr_db + 3.0

    # The second-stage noise must not spoil the LNA: allow 0.5 dB degradation from it.
    downstream = Lineup(tuple(st for st in lineup.stages if st.name in ("Mixer", "IF amplifier")))
    f_lna, f_rest = 10 ** (f.lna_nf_db / 10), 10 ** (downstream.noise_figure_db / 10)
    allowed = f_lna * (10 ** (0.5 / 10) - 1.0)
    required_lna_gain = float(db((f_rest - 1.0) / allowed))

    # Third-order products from the direct path should stay below in-band noise.
    required_iip3 = direct.element_maximum_dbm + (direct.element_maximum_dbm - noise_in) / 2.0
    # A constant-envelope signal's own third-order term is a scaled copy of itself (compression), not noise.
    varying_envelope = waveform.papr_db > CONSTANT_ENVELOPE_PAPR_DB
    intermod_dbc = None
    if varying_envelope and math.isfinite(lineup.iip3_dbm):
        intermod_dbc = 2.0 * (direct.element_nominal_dbm - lineup.iip3_dbm)

    # --- Direct path at the detector, after beamforming and matched filtering --------
    integration = waveform.integration_time_s
    post_noise = noise.noise_output_dbm_hz - float(db(integration))
    leakage = waveform.direct_leakage_db(delay_s, doppler_hz)
    before_cancel = direct.output_nominal_dbm - pre_adc + leakage
    needed = max(before_cancel - post_noise + f.residual_below_noise_db, 0.0)
    applied = min(needed, f.digital_canceller_limit_db)
    residual = before_cancel - applied

    # --- Other subsystems --------------------------------------------------------------
    n0_output = noise.noise_output_dbm_hz
    required_phase_noise = n0_output - direct.output_nominal_dbm - 6.0
    pa_saturated = None
    pa_row = Requirement(
        "Transmitter PA saturated power", "Not applicable", "Third-party transmitter", None,
        "The EIRP override means the PA is not part of this system.",
    )
    if s.transmitter.eirp_override_dbw is None:
        pa_saturated = s.transmitter.power_w * 10 ** (waveform.papr_db / 10.0)
        average = s.transmitter.power_w * waveform.duty_cycle
        pa_row = Requirement(
            "Transmitter PA saturated power", f"{pa_saturated:.0f} W",
            f"{s.transmitter.power_w:.0f} W mean envelope; {average:.0f} W average", None,
            f"Mean envelope power plus {waveform.papr_db:.1f} dB back-off for {waveform.name}. "
            f"Duty cycle {waveform.duty_cycle:.2f} sets the average heat load.",
        )

    window_text = (
        f"{gain_min:.1f} to {gain_max:.1f} dB ({window:.1f} dB wide)"
        if window >= 0
        else f"Empty: minimum {gain_min:.1f} dB exceeds maximum {gain_max:.1f} dB"
    )
    if gated:
        switch_step = configured_gain - gain_worst
        blind_path_m = (waveform.cfg.pulse_width_s + f.gain_switch_time_s) * C_M_S
        gain_rows = [
            Requirement(
                "Noise above quantization", "Computed",
                f"{noq_nominal:.1f} dB while listening; {noise_over_quant:.1f} dB during the pulse", None,
                f"Gated gain: between direct pulses the gain returns to {listening:.1f} dB, so noise sits {noq_nominal:.1f} dB above "
                f"quantization and costs {loss_nominal:.2f} dB, whatever the direct-path level. During the pulse it could sit at most "
                f"{noise_over_quant:.1f} dB above (range {instantaneous_dr:.1f} dB, back-off {backoff:.1f} dB, direct-to-noise "
                f"{direct_to_noise:.1f} dB), but the receiver is blanked then.",
            ),
            Requirement(
                "Receiver gain window", "Only during the blanked pulse", window_text, None,
                "From the gain that lifts noise to quantization up to the gain at which the direct pulse clips. With gated gain it "
                "applies only while the receiver is blanked, so an empty window does not cost detection.",
            ),
            Requirement(
                "Gain switching", f"Drop {max(switch_step, 0.0):.1f} dB for the direct pulse, then restore",
                f"{f.gain_switch_time_s * 1e6:.2f} us settling", None,
                f"Pulse gain {gain_worst:.1f} dB keeps the maximum direct pulse below clipping; listening gain {listening:.1f} dB. "
                f"The receiver is blind for the {waveform.cfg.pulse_width_s * 1e6:.1f} us pulse plus the settling time, so echoes "
                f"within {blind_path_m / 1e3:.2f} km of extra path are eclipsed. Stages ahead of the switch see the full pulse and "
                "must stay linear or recover within the settling time.",
            ),
            Requirement(
                "AGC range", f"{agc_range:.1f} dB", "Affects the pulse gain only", None,
                "Maximum minus minimum direct path at one element. With gated gain it changes only the gain during the pulse; "
                "the listening gain does not depend on the direct path.",
            ),
        ]
    else:
        gain_rows = [
            Requirement(
                "Noise above quantization", "Computed", f"{noise_over_quant:.1f} dB at best; {noq_nominal:.1f} dB in operation", None,
                f"Instantaneous range {instantaneous_dr:.1f} dB at {fs / 1e6:.3g} MS/s, minus {backoff:.1f} dB back-off, minus a "
                f"{direct_to_noise:.1f} dB direct-to-noise ratio at the ADC input. Costs {loss_nominal:.2f} dB of sensitivity at the "
                f"nominal direct level and {loss_worst:.2f} dB at the maximum."
                + (" Continuous gain is selected, so the direct pulse sets the gain even between pulses." if waveform.is_pulsed else ""),
            ),
            Requirement(
                "Receiver gain window", "Non-empty window", window_text, window >= 0,
                "From the gain that lifts noise to the quantization level, up to the gain at which the strongest signal clips."
                + (f" Short by {shortfall:.1f} dB." if shortfall > 0 else ""),
            ),
            Requirement(
                "Configured gain vs window", "Inside the window", f"{configured_gain:.1f} dB",
                gain_min <= configured_gain <= gain_max if window >= 0 else False,
                "Line-up gain set on the front-end page.",
            ),
            Requirement(
                "AGC range", f"{agc_range:.1f} dB", "Must be fitted if over about 3 dB", None,
                f"Maximum minus minimum direct path at one element. Backing off at the nominal level costs "
                f"{loss_nominal:.2f} dB of sensitivity; at the maximum level, {loss_worst:.2f} dB.",
            ),
        ]
    requirements = [
        *gain_rows,
        Requirement(
            "Limiter threshold", f"{at_lna_max + waveform.papr_db:.1f} to {f.lna_survival_dbm:.1f} dBm at the LNA input",
            f"LNA survives {f.lna_survival_dbm:.1f} dBm",
            at_lna_max + waveform.papr_db <= f.lna_survival_dbm,
            "Above the highest normal peak so it stays out of normal operation, and below the LNA's survival level.",
        ),
        Requirement(
            "LNA input P1dB", f"{required_lna_p1db:.1f} dBm or more", f"{f.lna_input_p1db_dbm:.1f} dBm",
            f.lna_input_p1db_dbm >= required_lna_p1db,
            f"Peak direct signal at the LNA input ({at_lna_max:.1f} dBm plus PAPR) with 3 dB margin.",
        ),
        Requirement(
            "LNA gain", f"{required_lna_gain:.1f} dB or more", f"{f.lna_gain_db:.1f} dB",
            f.lna_gain_db >= required_lna_gain,
            "Keeps later stages from degrading noise figure by more than 0.5 dB.",
        ),
        Requirement(
            "Cascade input IP3", f"{required_iip3:.1f} dBm or more", f"{lineup.iip3_dbm:.1f} dBm",
            lineup.iip3_dbm >= required_iip3,
            "Puts third-order products from the direct signal at or below in-band noise. "
            + (
                f"At the nominal level they sit {-intermod_dbc:.0f} dB below the direct signal, and that distortion is counted in the target's SINR."
                if intermod_dbc is not None
                else f"{waveform.name} has a constant envelope, so its own third-order term is not noise-like and no SINR penalty is applied; "
                "the requirement still matters when other strong signals are present."
            ),
        ),
        Requirement(
            "Suppression before the ADC", f"{max(shortfall, 0.0) + pre_adc:.1f} dB", f"{pre_adc:.1f} dB fitted",
            shortfall == 0.0,
            "Only an analog null or RF canceller can close a gain-window shortfall. A digital null cannot.",
        ),
        Requirement(
            "Digital cancellation after the ADC", f"{needed:.1f} dB", f"Limit {f.digital_canceller_limit_db:.1f} dB",
            needed <= f.digital_canceller_limit_db,
            f"Puts the direct-path leakage ({leakage:.1f} dB from {waveform.name}) in the target's cell "
            f"{f.residual_below_noise_db:.0f} dB below integrated noise. ECA-type cancellation suits continuous waveforms.",
        ),
        Requirement(
            "LO phase noise at target Doppler", f"{required_phase_noise:.1f} dBc/Hz or better",
            f"{f.lo_phase_noise_dbc_hz:.1f} dBc/Hz", f.lo_phase_noise_dbc_hz <= required_phase_noise,
            f"6 dB under receiver noise density, for the {direct.output_nominal_dbm:.1f} dBm direct signal after beamforming. "
            "Assumes one LO shared across channels, so the spatial null also helps. "
            + (
                "With a pulsed waveform the direct pulse is separated in time, so no SINR penalty is applied."
                if waveform.is_pulsed
                else "The phase-noise skirt of the direct signal is counted in the target's SINR."
            ),
        ),
        pa_row,
    ]
    return FrontEndResult(
        requirements=requirements,
        effective_bits=enob,
        sample_rate_hz=fs,
        instantaneous_dynamic_range_db=instantaneous_dr,
        noise_at_input_dbm=noise_in,
        quantization_dbm=quant,
        pre_adc_suppression_db=pre_adc,
        gain_min_db=gain_min,
        gain_max_db=gain_max,
        gain_window_db=window,
        shortfall_db=shortfall,
        noise_over_quantization_db=noise_over_quant,
        noise_over_quantization_nominal_db=noq_nominal,
        noise_over_quantization_worst_db=noq_worst,
        direct_to_noise_at_adc_db=direct_to_noise,
        clipping_backoff_db=backoff,
        allowed_quantization_loss_db=None,
        required_effective_bits=None,
        required_adc_bits=None,
        configured_gain_db=configured_gain,
        operating_gain_db=gain_nominal,
        gain_control="gated" if gated else "continuous",
        listening_gain_db=listening,
        pulse_gain_db=gain_worst,
        blind_extra_s=blind_extra,
        agc_range_db=agc_range,
        sensitivity_loss_db=loss_nominal,
        worst_sensitivity_loss_db=loss_worst,
        digital_cancellation_needed_db=needed,
        digital_cancellation_applied_db=applied,
        direct_residual_after_processing_dbm=residual,
        post_integration_noise_dbm=post_noise,
        direct_to_noise_element_db=direct.element_nominal_dbm - noise_in,
        pa_saturated_w=pa_saturated,
        direct_intermod_dbc=intermod_dbc,
    )




def _relative(level_db: float) -> str:
    """A minimum noise-over-quantization ratio in words."""
    return f"at least {level_db:.1f} dB above" if level_db >= 0.0 else f"no more than {-level_db:.1f} dB below"


def allowed_quantization_loss_db(detection_margin_db: float, quantization_loss_in_margin_db: float) -> float:
    """Quantization loss the detection margin can absorb: the margin an ideal ADC would leave."""
    return detection_margin_db + quantization_loss_in_margin_db


def noise_over_quantization_for_loss_db(loss_db: float) -> float:
    """Inverse of sensitivity_loss_db: the noise-over-quantization ratio that costs loss_db."""
    if loss_db <= 0.0:
        return math.inf
    return float(-10.0 * math.log10(10.0 ** (loss_db / 10.0) - 1.0))


def with_detection_margin(fe: FrontEndResult, s: Scenario, detection_margin_db: float) -> FrontEndResult:
    """Add the ADC resolution requirement, which depends on the target stage's detection margin.

    The margin already includes the quantization loss at the nominal direct level, so adding
    it back gives the margin of an ideal ADC. The loss at the maximum direct level, where the
    AGC has backed the gain off furthest, must fit inside that. Each effective bit lowers the
    quantization floor by 6.02 dB at a fixed full scale.
    """
    allowed = allowed_quantization_loss_db(detection_margin_db, fe.sensitivity_loss_db)
    needed_noq = noise_over_quantization_for_loss_db(allowed)
    required_enob = fe.effective_bits + (needed_noq - fe.noise_over_quantization_worst_db) / 6.02
    required_bits = required_enob + s.frontend.adc_enob_loss_bits
    reachable = math.isfinite(required_bits)
    row = Requirement(
        "ADC resolution",
        f"{required_bits:.1f} bits ({required_enob:.1f} effective)" if reachable else "Not reachable",
        f"{s.frontend.adc_bits} bits ({fe.effective_bits:.1f} effective)",
        reachable and required_bits <= s.frontend.adc_bits,
        (
            f"An ideal ADC would leave {allowed:.1f} dB of detection margin. "
            + ("While listening between pulses" if fe.gain_control == "gated" else "At the maximum direct signal")
            + ", quantization must cost "
            f"no more than that, so noise must sit {_relative(needed_noq)} quantization; this design gives "
            f"{fe.noise_over_quantization_worst_db:.1f} dB ({fe.worst_sensitivity_loss_db:.2f} dB loss)."
            + (
                f" With gated gain, raising the gain while listening by {needed_noq - fe.noise_over_quantization_worst_db:.1f} dB "
                "does the same as more bits."
                if fe.gain_control == "gated" and needed_noq > fe.noise_over_quantization_worst_db
                else ""
            )
            if reachable
            else f"Detection misses by {-allowed:.1f} dB even with an ideal ADC, so no resolution is enough. Fix the link first."
        ),
    )
    rows = list(fe.requirements)
    after_window = next((i + 1 for i, q in enumerate(rows) if q.item == "Receiver gain window"), 0)
    rows.insert(after_window, row)
    return replace(
        fe,
        requirements=rows,
        allowed_quantization_loss_db=allowed,
        required_effective_bits=required_enob,
        required_adc_bits=required_bits,
    )
