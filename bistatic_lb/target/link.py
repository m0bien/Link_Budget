"""Stage 3: target signal, interference and detection, at any target position.

Bi-static radar equation (Willis, Bistatic Radar, ch. 4):

    Pr = Pt Gt Gr lambda^2 sigma / ((4 pi)^3 Rt^2 Rr^2 L)

All levels in the target's range-Doppler cell are compared after matched filtering:
target power Pr, noise N0 / T_int, direct-path leakage and clutter. Their ratio is the
SINR, which equals E/N0 when interference is absent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..antenna.arrays import ReceiveArray
from ..antenna.elements import MonopoleTx
from ..config import Scenario
from ..frontend.requirements import FrontEndResult
from ..geometry import BistaticGeometry, build_geometry
from ..noise.budget import NoiseResult
from ..propagation.budget import DirectPathResult, path_terms, transmit_eirp_dbw
from ..results import Contribution
from ..units import C_M_S, K_BOLTZMANN, db, undb, wavelength_m
from ..waveforms.base import Waveform
from .detection import Fluctuation, cfar_loss_db, detection_probability, make_fluctuation, required_snr_db
from .rcs import bistatic_rcs

CLUTTER_SCATTER_HEIGHT_M = 5.0
MIN_PFA, MAX_PFA = 1e-12, 0.1


@dataclass(frozen=True)
class TargetLink:
    """Everything known about the target signal at one position."""

    geometry: BistaticGeometry
    rcs_dbsm: float
    rcs_contribution: Contribution
    eirp_toward_target_dbw: float
    tx_gain_dbi: float
    rx_gain_dbi: float
    excess_tx_db: float
    excess_rx_db: float
    total_losses_db: float
    received_dbm: float
    noise_cell_dbm: float
    direct_cell_dbm: float
    clutter_cell_dbm: float
    snr_db: float
    sinr_db: float
    doppler_hz: float
    delay_s: float
    weights: np.ndarray
    contributions_tx: list[Contribution]
    contributions_rx: list[Contribution]


@dataclass(frozen=True)
class DetectionSummary:
    n_cells: float
    pfa: float
    threshold_note: str
    cfar_loss_db: float
    required_snr_db: float
    margin_db: float
    pd: float
    pd_no_interference: float
    fluctuation: Fluctuation
    max_range_product_km2: float
    sinr_upper_decile_db: float
    pd_upper_decile: float
    margin_upper_decile_db: float


class TargetAnalyzer:
    """Holds the fixed parts of the system, and evaluates the link at any target position."""

    def __init__(
        self,
        s: Scenario,
        array: ReceiveArray,
        monopole: MonopoleTx,
        waveform: Waveform,
        noise: NoiseResult,
        direct: DirectPathResult,
        frontend: FrontEndResult,
        tx_mismatch_db: float,
        rx_mismatch_db: float,
    ):
        self.s, self.array, self.monopole, self.waveform = s, array, monopole, waveform
        self.noise, self.direct, self.frontend = noise, direct, frontend
        self.tx_mismatch_db, self.rx_mismatch_db = tx_mismatch_db, rx_mismatch_db
        self.lam = wavelength_m(s.frequency_hz)

    def steering_weights(self, geometry: BistaticGeometry) -> np.ndarray:
        a = self.s.array
        look = (geometry.azimuth_at_rx_rad, geometry.elevation_at_rx_rad)
        null_dir = (0.0, geometry.direct_path_elevation_at_rx_rad) if a.null_on_transmitter and self.array.n_elements > 1 else None
        return self.array.beamformer(*look, null_dir=null_dir)

    def _clutter_cell_dbm(self, g: BistaticGeometry, weights: np.ndarray, doppler_hz: float, eirp_dbw: float) -> float:
        s = self.s
        env = s.environment
        if not env.radar_clutter_enabled:
            return -300.0
        ground = np.array([g.target[0], g.target[1], 0.0])
        rt = float(np.linalg.norm(ground - g.tx))
        rr = float(np.linalg.norm(ground - g.rx))
        grazing = max(np.arctan2(s.transmitter.height_m, rt), np.arctan2(s.array.height_m, rr))
        sigma0_db = env.clutter_gamma_db + float(db(np.sin(grazing)))
        beamwidth = np.deg2rad(self.array.beamwidth_az_deg(weights, g.azimuth_at_rx_rad, -grazing))
        cos_half = max(np.cos(g.bistatic_angle_rad / 2.0), 1e-3)
        area = (C_M_S / (2.0 * self.waveform.bandwidth_hz)) * rr * beamwidth / cos_half**2
        terms_t = path_terms(s, rt / 1e3, s.transmitter.height_m, CLUTTER_SCATTER_HEIGHT_M)
        terms_r = path_terms(s, rr / 1e3, CLUTTER_SCATTER_HEIGHT_M, s.array.height_m)
        gain = float(db(self.array.gain_linear(weights, g.azimuth_at_rx_rad, -grazing)))
        power = (
            eirp_dbw + 30.0 + gain + sigma0_db + float(db(area)) + 2.0 * float(db(self.lam)) - 3.0 * float(db(4.0 * np.pi))
            - 2.0 * float(db(rt)) - 2.0 * float(db(rr)) - terms_t.excess_loss_mean_db - terms_r.excess_loss_mean_db
            - self.rx_mismatch_db
        )
        return float(power + self.waveform.doppler_rejection_db(doppler_hz))

    def link(self, target_xyz=None) -> TargetLink:
        s = self.s
        g = build_geometry(s, target_xyz)
        wf = self.waveform
        rcs = bistatic_rcs(s.target, g.bistatic_angle_rad, self.lam)
        weights = self.steering_weights(g)
        az, el = g.azimuth_at_rx_rad, g.elevation_at_rx_rad
        rx_gain = float(self.array.gain_dbi(weights, az, el))
        eirp, tx_gain = transmit_eirp_dbw(s, self.monopole, g.elevation_at_tx_rad, self.tx_mismatch_db)

        rt_km, rr_km = g.range_tx_m / 1e3, g.range_rx_m / 1e3
        terms_t = path_terms(s, rt_km, s.transmitter.height_m, g.target[2], elevation_rad=g.elevation_at_tx_rad)
        terms_r = path_terms(s, rr_km, g.target[2], s.array.height_m, elevation_rad=g.elevation_at_rx_rad)

        doppler = g.doppler_hz(self.lam)
        delay = g.bistatic_range_m / C_M_S
        polarization = s.array.target_depolarization_loss_db
        if s.array.orientation == "horizontal":
            polarization += s.array.cross_pol_isolation_db
        losses = (
            s.detection.processing_loss_db + s.waveform.window_loss_db + s.array.beam_straddle_loss_db
            + self.frontend.sensitivity_loss_db + wf.signal_loss_db(delay, doppler, self.frontend.blind_extra_s) + polarization + self.rx_mismatch_db
        )
        received = (
            eirp + 30.0 + rx_gain + rcs.value_dbsm + 2.0 * float(db(self.lam)) - 3.0 * float(db(4.0 * np.pi))
            - 2.0 * float(db(g.range_tx_m)) - 2.0 * float(db(g.range_rx_m))
            - terms_t.excess_loss_mean_db - terms_r.excess_loss_mean_db - losses
        )
        noise_cell = self.noise.design_output_dbm_hz - float(db(wf.integration_time_s))

        # Direct-path leakage in this cell, after the beam's response toward the transmitter.
        direct_gain = float(db(self.array.gain_with_errors_linear(weights, 0.0, g.direct_path_elevation_at_rx_rad, s.array.calibration_error_deg)))
        direct_out = self.direct.output_nominal_dbm - self.direct.array_gain_toward_tx_dbi + direct_gain
        before_cancel = direct_out - self.frontend.pre_adc_suppression_db + wf.direct_leakage_db(delay, doppler)
        needed = max(before_cancel - noise_cell + s.frontend.residual_below_noise_db, 0.0)
        direct_cell = before_cancel - min(needed, s.frontend.digital_canceller_limit_db)

        clutter_cell = self._clutter_cell_dbm(g, weights, doppler, eirp)
        interference = undb(noise_cell) + undb(direct_cell) + undb(clutter_cell)

        return TargetLink(
            geometry=g,
            rcs_dbsm=rcs.value_dbsm,
            rcs_contribution=rcs.contribution,
            eirp_toward_target_dbw=eirp,
            tx_gain_dbi=tx_gain,
            rx_gain_dbi=rx_gain,
            excess_tx_db=terms_t.excess_loss_mean_db,
            excess_rx_db=terms_r.excess_loss_mean_db,
            total_losses_db=losses,
            received_dbm=float(received),
            noise_cell_dbm=noise_cell,
            direct_cell_dbm=float(direct_cell),
            clutter_cell_dbm=float(clutter_cell),
            snr_db=float(received - noise_cell),
            sinr_db=float(received - db(interference)),
            doppler_hz=doppler,
            delay_s=delay,
            weights=weights,
            contributions_tx=terms_t.contributions,
            contributions_rx=terms_r.contributions,
        )

    # --- Detection --------------------------------------------------------------------
    def number_of_search_cells(self) -> float:
        s = self.s
        max_doppler = 2.0 * s.detection.max_speed_m_s / self.lam
        cells = self.waveform.number_of_range_doppler_cells(s.detection.max_bistatic_range_km * 1e3, max_doppler)
        g = build_geometry(s)
        weights = self.steering_weights(g)
        beamwidth = max(self.array.beamwidth_az_deg(weights, g.azimuth_at_rx_rad, g.elevation_at_rx_rad), 1.0)
        beams = max(360.0 / beamwidth, 1.0)
        if self.array.has_elevation:
            beams *= max(s.array.n_vertical / 2.0, 1.0)
        return cells * beams

    def design_pfa(self) -> float:
        s = self.s
        frame = self.waveform.cpi_s * s.detection.noncoherent_looks
        return float(np.clip(s.detection.false_alarms_per_s * frame / self.number_of_search_cells(), MIN_PFA, MAX_PFA))

    def detection_at(self, sinr_db: float, pfa: float | None = None) -> float:
        s = self.s
        pfa = self.design_pfa() if pfa is None else pfa
        n = s.detection.noncoherent_looks
        model = make_fluctuation(s.target)
        loss = cfar_loss_db(s.detection.required_pd, pfa, s.detection.cfar_reference_cells)
        return detection_probability(float(undb(sinr_db - loss)), pfa, n, model)

    def detection_summary(self, link: TargetLink) -> DetectionSummary:
        s = self.s
        pfa = self.design_pfa()
        n = s.detection.noncoherent_looks
        model = make_fluctuation(s.target)
        loss = cfar_loss_db(s.detection.required_pd, pfa, s.detection.cfar_reference_cells)
        required = required_snr_db(s.detection.required_pd, pfa, n, model) + loss
        margin = link.sinr_db - required
        range_product = link.geometry.range_tx_m * link.geometry.range_rx_m / 1e6

        # The same link with noise at its upper decile, whichever design point is selected.
        upper_noise_dbm_hz = float(db(K_BOLTZMANN * self.noise.statistics.system_upper_k) + 30.0)
        upper_cell = upper_noise_dbm_hz - float(db(self.waveform.integration_time_s))
        sinr_upper = float(link.received_dbm - db(undb(upper_cell) + undb(link.direct_cell_dbm) + undb(link.clutter_cell_dbm)))
        return DetectionSummary(
            n_cells=self.number_of_search_cells(),
            pfa=pfa,
            threshold_note=f"{s.detection.false_alarms_per_s:g} false alarms per second across {self.number_of_search_cells():.3g} cells.",
            cfar_loss_db=loss,
            required_snr_db=required,
            margin_db=margin,
            pd=self.detection_at(link.sinr_db, pfa),
            pd_no_interference=self.detection_at(link.snr_db, pfa),
            fluctuation=model,
            max_range_product_km2=float(range_product * undb(margin / 2.0)),
            sinr_upper_decile_db=sinr_upper,
            pd_upper_decile=self.detection_at(sinr_upper, pfa),
            margin_upper_decile_db=sinr_upper - required,
        )
