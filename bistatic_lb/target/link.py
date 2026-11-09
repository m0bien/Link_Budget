"""Stage 3: target signal, interference and detection, at any target position.

Bi-static radar equation (Willis, Bistatic Radar, ch. 4):

    Pr = Pt Gt Gr lambda^2 sigma / ((4 pi)^3 Rt^2 Rr^2 L)

All levels in the target's range-Doppler cell are compared after matched filtering:
target power Pr, noise N0 / T_int, direct-path leakage and clutter. Their ratio is the
SINR, which equals E/N0 when interference is absent.

Two front-end effects of the direct signal also land in the target's cell as noise:

    LO phase noise    the direct signal's skirt at the target's Doppler offset, a density of
                      P_direct + L(f) dBm/Hz. The LO is shared, so the spatial null and analog
                      suppression lower it, but a linear canceller does not. Not applied to pulsed
                      waveforms, where the direct pulse is separated in time.
    Intermodulation   third-order distortion of a varying-envelope direct signal, 2 (P - IIP3) dB
                      below it per element and spread over the bandwidth. It follows the direct
                      signal's phase at each element, so the spatial null lowers it too.

Both are planning estimates; the two-tone intercept relation is used for the noise-like case.

Stationary interference. The direct signal's leakage and the ground clutter in the target's
range cell (clutter.py) are both at zero Doppler. The Doppler filter rejects them by its
response at the target's Doppler, and the digital canceller then removes what is needed to
put the rest below noise, up to its stated limit. This assumes the canceller spans the
clutter's range extent. A canceller that removes zero-Doppler signals also removes a
zero-Doppler target: it keeps 1 - (1 - 1/depth) sinc^2(x) of a target x Doppler cells from
zero, the response of projecting out the zero-Doppler component over one interval. That
notch is what sets the minimum detectable velocity.
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
from .clutter import LegLoss, ring_clutter_dbm
from .migration import Migration, migration
from .detection import Fluctuation, cfar_loss_db, detection_probability, make_fluctuation, required_snr_db
from .rcs import bistatic_rcs

CLUTTER_SCATTER_HEIGHT_M = 5.0
ABSENT_DBM = -300.0  # stands for "no such interference"
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
    clutter_zero_doppler_dbm: float  # in the range cell, before the Doppler filter and the canceller
    cancellation_needed_db: float
    cancellation_applied_db: float
    notch_loss_db: float  # what the canceller takes from the target itself
    lobing_extra_loss_db: float  # loss with the actual ground-ray phase, minus the phase-averaged loss
    phase_noise_cell_dbm: float
    intermod_cell_dbm: float
    migration: Migration
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
    min_detectable_range_rate_m_s: float = 0.0  # smallest bi-static range rate that meets the required Pd; inf if none does


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
        self._clutter_legs: tuple[LegLoss, LegLoss] | None = None
        # Transmit gain against depression angle, tabulated for the clutter integral.
        self._tx_angles = np.linspace(0.0, np.pi / 2.0, 181)
        self._tx_gain = np.array([monopole.gain_linear(float(a)) for a in self._tx_angles])

    def steering_weights(self, geometry: BistaticGeometry) -> np.ndarray:
        a = self.s.array
        look = (geometry.azimuth_at_rx_rad, geometry.elevation_at_rx_rad)
        null_dir = (0.0, geometry.direct_path_elevation_at_rx_rad) if a.null_on_transmitter and self.array.n_elements > 1 else None
        return self.array.beamformer(*look, null_dir=null_dir)

    def _clutter_zero_doppler_dbm(self, g: BistaticGeometry, weights: np.ndarray) -> float:
        """Ground clutter in the target's range cell, at zero Doppler: the ring integral of clutter.py."""
        s = self.s
        if not s.environment.radar_clutter_enabled:
            return ABSENT_DBM
        if self._clutter_legs is None:
            ht, hr = s.transmitter.height_m, s.array.height_m
            self._clutter_legs = (
                LegLoss(lambda km: path_terms(s, km, ht, CLUTTER_SCATTER_HEIGHT_M).excess_loss_mean_db),
                LegLoss(lambda km: path_terms(s, km, CLUTTER_SCATTER_HEIGHT_M, hr).excess_loss_mean_db),
            )
        if s.transmitter.eirp_override_dbw is not None:
            eirp_w = lambda grazing: np.full_like(grazing, undb(s.transmitter.eirp_override_dbw))  # noqa: E731
        else:
            fed_w = undb(float(db(s.transmitter.power_w)) - s.transmitter.line_loss_db - self.tx_mismatch_db)
            eirp_w = lambda grazing: fed_w * np.interp(grazing, self._tx_angles, self._tx_gain)  # noqa: E731
        power = ring_clutter_dbm(
            g.baseline_m, g.bistatic_range_m, self.waveform.bandwidth_hz, self.lam, s.transmitter.height_m, s.array.height_m,
            s.environment.clutter_gamma_db, eirp_w, lambda az, el: self.array.gain_linear(weights, az, el), *self._clutter_legs,
        )
        return power - self.rx_mismatch_db

    def link(self, target_xyz=None, doppler_hz: float | None = None) -> TargetLink:
        """The link at a target position. doppler_hz overrides the target's Doppler, to probe the Doppler response."""
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

        doppler = g.doppler_hz(self.lam) if doppler_hz is None else float(doppler_hz)
        delay = g.bistatic_range_m / C_M_S
        actual = s.environment.target_multipath == "actual"
        excess_t = terms_t.excess_loss_actual_db if actual else terms_t.excess_loss_mean_db
        excess_r = terms_r.excess_loss_actual_db if actual else terms_r.excess_loss_mean_db
        lobing_extra = (
            terms_t.excess_loss_actual_db - terms_t.excess_loss_mean_db + terms_r.excess_loss_actual_db - terms_r.excess_loss_mean_db
        )
        noise_cell = self.noise.design_output_dbm_hz - float(db(wf.integration_time_s))

        # Stationary interference in this cell: direct-path leakage and ground clutter. ---------
        direct_gain = float(db(self.array.gain_with_errors_linear(weights, 0.0, g.direct_path_elevation_at_rx_rad, s.array.calibration_error_deg)))
        direct_out = self.direct.output_nominal_dbm - self.direct.array_gain_toward_tx_dbi + direct_gain
        direct_before = direct_out - self.frontend.pre_adc_suppression_db + wf.direct_leakage_db(delay, doppler)
        clutter_zero = self._clutter_zero_doppler_dbm(g, weights)
        clutter_before = ABSENT_DBM if clutter_zero <= ABSENT_DBM else clutter_zero + wf.doppler_rejection_db(doppler)
        stationary = float(db(undb(direct_before) + undb(clutter_before)))
        needed = max(stationary - noise_cell + s.frontend.residual_below_noise_db, 0.0)
        applied = min(needed, s.frontend.digital_canceller_limit_db)
        direct_cell, clutter_cell = direct_before - applied, clutter_before - applied
        # The canceller cannot tell a zero-Doppler target from clutter.
        kept = 1.0 - (1.0 - float(undb(-applied))) * np.sinc(wf.doppler_cells_from_zero(doppler)) ** 2
        notch_loss = float(-db(max(kept, 1e-12)))

        polarization = s.array.target_depolarization_loss_db
        if s.array.orientation == "horizontal":
            polarization += s.array.cross_pol_isolation_db
        walk = migration(
            g.bistatic_range_rate_m_s, g.bistatic_range_accel_m_s2, wf.cpi_s, wf.bandwidth_hz, self.lam, s.waveform.migration_compensation
        )
        losses = (
            s.detection.processing_loss_db + s.waveform.window_loss_db + s.array.beam_straddle_loss_db + walk.loss_db + notch_loss
            + self.frontend.sensitivity_loss_db + wf.signal_loss_db(delay, doppler, self.frontend.blind_extra_s) + polarization + self.rx_mismatch_db
        )
        received = (
            eirp + 30.0 + rx_gain + rcs.value_dbsm + 2.0 * float(db(self.lam)) - 3.0 * float(db(4.0 * np.pi))
            - 2.0 * float(db(g.range_tx_m)) - 2.0 * float(db(g.range_rx_m))
            - excess_t - excess_r - losses
        )

        # Front-end products of the direct signal that no linear canceller removes.
        integration_db = float(db(wf.integration_time_s))
        direct_at_output = direct_out - self.frontend.pre_adc_suppression_db
        phase_noise_cell = ABSENT_DBM if wf.is_pulsed else direct_at_output + s.frontend.lo_phase_noise_dbc_hz - integration_db
        intermod_cell = ABSENT_DBM
        if self.frontend.direct_intermod_dbc is not None:
            intermod_cell = direct_out + self.frontend.direct_intermod_dbc - float(db(wf.bandwidth_hz)) - integration_db
        interference = undb(noise_cell) + undb(direct_cell) + undb(clutter_cell) + undb(phase_noise_cell) + undb(intermod_cell)

        return TargetLink(
            geometry=g,
            rcs_dbsm=rcs.value_dbsm,
            rcs_contribution=rcs.contribution,
            eirp_toward_target_dbw=eirp,
            tx_gain_dbi=tx_gain,
            rx_gain_dbi=rx_gain,
            excess_tx_db=excess_t,
            excess_rx_db=excess_r,
            total_losses_db=losses,
            received_dbm=float(received),
            noise_cell_dbm=noise_cell,
            direct_cell_dbm=float(direct_cell),
            clutter_cell_dbm=float(clutter_cell),
            clutter_zero_doppler_dbm=float(clutter_zero),
            cancellation_needed_db=float(needed),
            cancellation_applied_db=float(applied),
            notch_loss_db=notch_loss,
            lobing_extra_loss_db=float(lobing_extra),
            phase_noise_cell_dbm=float(phase_noise_cell),
            intermod_cell_dbm=float(intermod_cell),
            migration=walk,
            snr_db=float(received - noise_cell),
            sinr_db=float(received - db(interference)),
            doppler_hz=doppler,
            delay_s=delay,
            weights=weights,
            contributions_tx=terms_t.contributions,
            contributions_rx=terms_r.contributions,
        )

    def min_detectable_range_rate_m_s(self, link: TargetLink, required_sinr_db: float) -> float:
        """Smallest bi-static range rate at which this target, at this position, reaches the required SINR.

        Scans the target's Doppler outward from zero through the canceller notch and the
        Doppler filter's main lobe. Infinite when no Doppler in the scan is enough.
        """
        cell_hz = self.waveform.doppler_resolution_hz
        for cells in np.r_[0.0, np.geomspace(0.05, 8.0, 30)]:
            if self.link(link.geometry.target, doppler_hz=cells * cell_hz).sinr_db >= required_sinr_db:
                return float(cells * cell_hz * self.lam)
        return float("inf")

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
        other = undb(link.direct_cell_dbm) + undb(link.clutter_cell_dbm) + undb(link.phase_noise_cell_dbm) + undb(link.intermod_cell_dbm)
        sinr_upper = float(link.received_dbm - db(undb(upper_cell) + other))
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
            min_detectable_range_rate_m_s=self.min_detectable_range_rate_m_s(link, required),
        )
