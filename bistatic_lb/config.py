"""Scenario description: every user-settable input lives here.

The models are plain pydantic classes, so inputs are range-checked on creation and
can be saved to or loaded from YAML. Nothing in this file does any physics.

Coordinates: the receiver array centre is the origin, the transmitter lies on the
+x axis, and azimuth is measured from the receiver-to-transmitter direction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TransmitterConfig(_Section):
    """Monopole transmitter."""

    power_w: float = Field(1000.0, gt=0, description="Mean envelope power while transmitting")
    height_m: float = Field(30.0, gt=0)
    design_frequency_hz: Optional[float] = Field(None, description="None: tuned to the operating frequency")
    ground_plane: Literal["ideal", "finite"] = "ideal"
    finite_ground_tilt_deg: float = Field(30.0, gt=0, le=80, description="Main-lobe elevation with a finite ground plane")
    finite_ground_horizon_loss_db: float = Field(6.0, ge=0, description="Gain reduction at the horizon with a finite ground plane")
    line_loss_db: float = Field(1.0, ge=0)
    eirp_override_dbw: Optional[float] = Field(None, description="Third-party transmitter: EIRP toward the horizon")


class ArrayConfig(_Section):
    """Receiver array: a ring of dipoles, optionally with a central vertical column."""

    element: Literal["dipole", "folded_dipole"] = "dipole"
    orientation: Literal["vertical", "horizontal"] = "vertical"
    n_circular: int = Field(16, ge=1, le=128)
    radius_wavelengths: float = Field(1.0, gt=0, description="Ring radius at the design frequency")
    n_vertical: int = Field(0, ge=0, le=64, description="Elements in the central column; 0 gives a 2-D array")
    vertical_spacing_wavelengths: float = Field(0.5, gt=0)
    design_frequency_hz: Optional[float] = Field(None, description="None: tuned to the operating frequency")
    height_m: float = Field(15.0, gt=0)
    taper: Literal["uniform", "cardioid"] = "uniform"
    digitization: Literal["per_element", "analog_subarray"] = "per_element"
    null_on_transmitter: bool = True
    calibration_error_deg: float = Field(2.0, ge=0, description="RMS channel error, limits digital null depth")
    analog_null_depth_db: float = Field(25.0, ge=0, description="Achievable RF null when using analog subarrays")
    cross_pol_isolation_db: float = Field(20.0, ge=0, description="Direct-path rejection when dipoles are cross-polarized to the Tx")
    target_depolarization_loss_db: float = Field(0.0, ge=0)
    beam_straddle_loss_db: float = Field(1.0, ge=0, description="Average loss of a target between beams")


class GeometryConfig(_Section):
    baseline_km: float = Field(20.0, gt=0)
    target_range_rx_km: float = Field(30.0, gt=0, description="Ground range from the receiver")
    target_azimuth_deg: float = Field(60.0, description="From the Rx-to-Tx direction")
    target_altitude_m: float = Field(1000.0, ge=0)
    target_speed_m_s: float = Field(100.0, ge=0)
    target_heading_deg: float = Field(150.0, description="Direction of travel, in the same azimuth frame")
    minimum_target_range_km: float = Field(1.0, gt=0, description="Closest target to Rx, for the maximum-signal check")


class EnvironmentConfig(_Section):
    man_made_category: Literal["city", "residential", "rural", "quiet_rural"] = "rural"
    man_made_override_db: Optional[float] = Field(None, description="Measured Fa at this frequency; replaces P.372")
    man_made_upper_decile_db: Optional[float] = Field(None, ge=0, description="Measured upper decile deviation; replaces P.372 Table 2")
    man_made_lower_decile_db: Optional[float] = Field(None, ge=0, description="Measured lower decile deviation; replaces P.372 Table 2")
    noise_design_point: Literal["median", "upper_decile"] = "median"
    ground_temperature_k: float = Field(290.0, gt=0)
    surface_temperature_k: float = Field(288.15, gt=200)
    pressure_hpa: float = Field(1013.25, gt=0)
    water_vapour_g_m3: float = Field(7.5, ge=0)
    refractivity_gradient_n_per_km: float = Field(-40.0, description="Lowest km; -40 gives k of about 4/3")
    rain_rate_mm_h: float = Field(0.0, ge=0, description="R0.01; zero disables rain")
    sun_in_beam: bool = False
    ground_type: Literal["sea", "wet", "medium", "dry"] = "medium"
    roughness_rms_m: float = Field(1.0, ge=0)
    terrain_profile: list[tuple[float, float]] = Field(
        default_factory=list, description="(distance from Tx km, height above sea level m), Tx to Rx"
    )
    tx_in_clutter: bool = False
    rx_in_clutter: bool = False
    clutter_location_percent: float = Field(50.0, gt=0, lt=100)
    log10_k_geoclimatic: float = Field(-6.0, description="P.530 log10(K), from LogK.csv or measurements")
    dn75: float = Field(30.0, ge=0, description="P.530 dN75, from dN75.csv")
    mean_terrain_elevation_m: float = 0.0
    fade_time_percent: float = Field(0.01, gt=0, lt=100, description="Time percentage for the minimum-signal fade")
    include_ducting_in_maximum: bool = True
    radar_clutter_enabled: bool = True
    clutter_gamma_db: float = Field(-15.0, description="Constant-gamma terrain reflectivity")


class FrontEndConfig(_Section):
    @model_validator(mode="before")
    @classmethod
    def _drop_retired_fields(cls, data):
        """Older scenario files set a noise-over-quantization target; it is now computed."""
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k != "noise_over_quantization_db"}
        return data

    """One receiver channel, in signal-flow order."""

    feed_loss_db: float = Field(0.5, ge=0)
    limiter_loss_db: float = Field(0.5, ge=0)
    rf_filter_loss_db: float = Field(1.0, ge=0)
    lna_gain_db: float = 25.0
    lna_nf_db: float = Field(1.5, ge=0)
    lna_iip3_dbm: float = 15.0
    lna_input_p1db_dbm: float = 0.0
    lna_survival_dbm: float = Field(30.0, description="Maximum safe power at the LNA input")
    mixer_loss_db: float = Field(7.0, ge=0)
    mixer_iip3_dbm: float = 20.0
    if_gain_db: float = 28.0
    if_nf_db: float = Field(6.0, ge=0)
    if_iip3_dbm: float = 15.0
    adc_bits: int = Field(16, ge=4, le=24)
    adc_enob_loss_bits: float = Field(2.5, ge=0, description="Nominal bits minus effective bits")
    adc_full_scale_dbm: float = 4.0
    adc_oversampling: float = Field(2.5, ge=1.0, description="Sample rate divided by signal bandwidth")
    adc_headroom_db: float = Field(3.0, ge=0, description="Margin below clipping, beyond PAPR")
    analog_canceller_db: float = Field(0.0, ge=0, description="RF direct-path cancellation before the ADC")
    digital_canceller_limit_db: float = Field(50.0, ge=0, description="Best achievable digital cancellation")
    residual_below_noise_db: float = Field(10.0, ge=0, description="Target level of the direct-path residual below integrated noise")
    lo_phase_noise_dbc_hz: float = Field(-110.0, description="At the target's Doppler offset")
    gain_control: Literal["auto", "continuous", "gated"] = Field(
        "auto",
        description="continuous: one AGC gain for all time, set by the direct signal. gated: low gain during the direct pulse, "
        "full gain while listening (pulsed waveforms only). auto: gated for pulsed waveforms, continuous otherwise.",
    )
    gain_switch_time_s: float = Field(2e-6, ge=0, description="Gated gain: settling or recovery time after the direct pulse, during which the receiver is blind")


class WaveformConfig(_Section):
    kind: Literal["pulsed_lfm", "phase_coded", "zadoff_chu", "fm", "dab", "dvbt"] = "pulsed_lfm"
    bandwidth_hz: float = Field(1e6, gt=0, description="LFM sweep or code chip rate; ignored by FM, DAB and DVB-T")
    cpi_s: float = Field(0.5, gt=0, description="Coherent processing interval")
    pulse_width_s: float = Field(100e-6, gt=0, description="Pulsed LFM")
    prf_hz: float = Field(2000.0, gt=0, description="Pulsed LFM")
    window_sidelobe_db: float = Field(-35.0, lt=0, description="Range-Doppler window peak sidelobe")
    window_loss_db: float = Field(1.3, ge=0)
    fm_programme: Literal["low", "typical", "high"] = "typical"
    dvbt_channel_mhz: Literal[6, 7, 8] = 8


class TargetConfig(_Section):
    rcs_dbsm: float = 10.0
    rcs_model: Literal["constant", "equivalence", "forward_scatter"] = "constant"
    forward_scatter_area_m2: float = Field(10.0, gt=0, description="Silhouette area for the forward-scatter model")
    fluctuation: Literal["swerling0", "swerling1", "swerling2", "swerling3", "swerling4", "chi2_scan", "rician"] = "swerling1"
    chi2_dof_m: float = Field(1.5, gt=0, description="Weinstock-style chi-square: m = DOF/2, slow fluctuation")
    rician_k_db: float = Field(6.0, description="Dominant-to-diffuse power ratio")


class DetectionConfig(_Section):
    false_alarms_per_s: float = Field(0.01, gt=0)
    max_bistatic_range_km: float = Field(150.0, gt=0, description="Sets the number of range cells")
    max_speed_m_s: float = Field(300.0, gt=0, description="Sets the number of Doppler cells")
    noncoherent_looks: int = Field(1, ge=1, le=256)
    cfar_reference_cells: int = Field(32, ge=0, description="0: fixed threshold, no CFAR loss")
    required_pd: float = Field(0.9, gt=0, lt=1)
    processing_loss_db: float = Field(2.0, ge=0)


class TrackerConfig(_Section):
    confirm_m: int = Field(3, ge=1)
    confirm_n: int = Field(4, ge=1, le=12)
    update_interval_s: float = Field(1.0, gt=0)
    target_confirm_probability: float = Field(0.95, gt=0, lt=1)
    allowed_false_tracks_per_hour: float = Field(1.0, gt=0)
    process_accel_m_s2: float = Field(2.0, gt=0)
    gate_sigma: float = Field(3.5, gt=0)
    association: Literal["pda", "nearest"] = "pda"

    @model_validator(mode="after")
    def _m_within_n(self):
        if self.confirm_m > self.confirm_n:
            raise ValueError("Track confirmation needs M <= N.")
        return self


class MonteCarloConfig(_Section):
    runs: int = Field(200, ge=1)
    scans: int = Field(60, ge=5)
    seed: int = 1


class Scenario(_Section):
    name: str = "Bi-static scenario"
    frequency_hz: float = Field(600e6, ge=50e6, le=18e9)
    transmitter: TransmitterConfig = Field(default_factory=TransmitterConfig)
    array: ArrayConfig = Field(default_factory=ArrayConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    frontend: FrontEndConfig = Field(default_factory=FrontEndConfig)
    waveform: WaveformConfig = Field(default_factory=WaveformConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)

    @classmethod
    def from_yaml(cls, source: str | Path) -> "Scenario":
        """Load from a YAML file path or a YAML string."""
        is_path = isinstance(source, Path) or "\n" not in str(source)
        text = Path(source).read_text(encoding="utf-8") if is_path else str(source)
        return cls.model_validate(yaml.safe_load(text))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    def with_value(self, dotted_path: str, value) -> "Scenario":
        """Return a validated copy with one field changed, e.g. 'array.n_circular'."""
        data = self.model_dump()
        node = data
        keys = dotted_path.split(".")
        for key in keys[:-1]:
            node = node[key]
        if keys[-1] not in node:
            raise KeyError(f"Unknown scenario field: {dotted_path}")
        node[keys[-1]] = value
        return Scenario.model_validate(data)
