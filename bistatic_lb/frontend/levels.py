"""Signal levels through the receiver line-up, stage by stage, for the strongest direct signal.

This shows where gain acts and where it runs into trouble:

- each stage's output level for the maximum direct path (mean and peak) and for noise,
- each active stage's output 1 dB compression point, and whether the peak passes it,
- the AGC cut needed so the mean direct level stays at or below the ADC clipping limit
  (full scale less PAPR and headroom), and how far it would overshoot without the AGC.

The AGC is modelled as an ideal variable attenuator after the IF amplifier, just before
the ADC, matching the front-end model, where it only lowers the total gain. Stages before
it see the full configured gain, so they can compress even when the ADC is protected.
With gated gain (pulsed waveforms) the cut applies only while the direct pulse is present;
between pulses the gain returns and the noise at the ADC input rises by the same amount.

Noise is the system noise at the antenna terminals (external plus receiver, referred to
the input) carried through the cumulative gain. It is exact at the ADC; at earlier points
it includes noise that later stages have not yet added, which is a small effect after a
good LNA.

Where a stage has no stated compression point, the input P1dB is estimated as IIP3 - 9.6 dB,
the relation for a memoryless cubic non-linearity. The result flags it as an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import FrontEndConfig
from ..noise.receiver import Lineup

P1DB_BELOW_IIP3_DB = 9.6


@dataclass(frozen=True)
class LevelPoint:
    name: str  # the point after this stage
    gain_db: float  # cumulative gain from the antenna terminals to here
    direct_dbm: float  # maximum direct path, mean power
    peak_dbm: float  # maximum direct path, peak power (mean plus PAPR)
    noise_dbm: float  # in-band noise
    p1db_out_dbm: float | None  # output 1 dB compression of the stage just passed
    p1db_estimated: bool
    compresses: bool


@dataclass(frozen=True)
class LineupLevels:
    points: list[LevelPoint]
    configured_gain_db: float
    agc_cut_db: float  # attenuation the AGC adds when the direct path is at its maximum
    adc_full_scale_dbm: float
    clipping_limit_dbm: float
    quantization_dbm: float
    direct_without_agc_dbm: float  # mean direct level at the ADC input with the configured gain
    listening_noise_dbm: float | None = None  # gated gain: noise at the ADC input between direct pulses

    @property
    def overshoot_without_agc_db(self) -> float:
        """How far the mean direct level would pass the clipping limit with no AGC; negative means it fits."""
        return self.direct_without_agc_dbm - self.clipping_limit_dbm

    @property
    def compressing(self) -> list[LevelPoint]:
        return [p for p in self.points if p.compresses]


def _input_p1db(stage_name: str, iip3_dbm: float, f: FrontEndConfig) -> tuple[float | None, bool]:
    if stage_name == "LNA":
        return f.lna_input_p1db_dbm, False
    if np.isfinite(iip3_dbm):
        return iip3_dbm - P1DB_BELOW_IIP3_DB, True
    return None, False


def lineup_levels(
    lineup: Lineup,
    f: FrontEndConfig,
    direct_maximum_dbm: float,
    noise_at_input_dbm: float,
    papr_db: float,
    pre_adc_suppression_db: float,
    agc_gain_db: float,
    quantization_dbm: float,
    listening_gain_db: float | None = None,
) -> LineupLevels:
    """Levels after each stage. agc_gain_db is the total gain the AGC allows at the maximum direct level.

    listening_gain_db is the gain between direct pulses when the gain is gated, otherwise None.
    """
    points = [LevelPoint("Antenna", 0.0, direct_maximum_dbm, direct_maximum_dbm + papr_db, noise_at_input_dbm, None, False, False)]
    gain, direct = 0.0, direct_maximum_dbm
    for stage in lineup.stages:
        p1db_in, estimated = _input_p1db(stage.name, stage.iip3_dbm, f)
        compresses = p1db_in is not None and direct + papr_db > p1db_in
        gain += stage.gain_db
        direct += stage.gain_db
        p1db_out = None if p1db_in is None else p1db_in + stage.gain_db - 1.0
        points.append(LevelPoint(stage.name, gain, direct, direct + papr_db, noise_at_input_dbm + gain, p1db_out, estimated, compresses))
    if pre_adc_suppression_db > 0:
        direct -= pre_adc_suppression_db
        points.append(LevelPoint("Analog suppression", gain, direct, direct + papr_db, noise_at_input_dbm + gain, None, False, False))
    direct_without_agc = direct
    cut = max(gain - agc_gain_db, 0.0)
    gain -= cut
    direct -= cut
    points.append(LevelPoint("AGC, ADC input", gain, direct, direct + papr_db, noise_at_input_dbm + gain, None, False, False))
    return LineupLevels(
        points=points,
        configured_gain_db=lineup.gain_db,
        agc_cut_db=cut,
        adc_full_scale_dbm=f.adc_full_scale_dbm,
        clipping_limit_dbm=f.adc_full_scale_dbm - papr_db - f.adc_headroom_db,
        quantization_dbm=quantization_dbm,
        direct_without_agc_dbm=direct_without_agc,
        listening_noise_dbm=None if listening_gain_db is None else noise_at_input_dbm + listening_gain_db,
    )


def levels_for(r) -> LineupLevels:
    """Line-up levels for a LinkBudgetResult."""
    fe = r.frontend
    return lineup_levels(
        r.lineup,
        r.scenario.frontend,
        r.direct.element_maximum_dbm,
        fe.noise_at_input_dbm,
        r.waveform.papr_db,
        fe.pre_adc_suppression_db,
        min(fe.configured_gain_db, fe.gain_max_db),
        fe.quantization_dbm,
        fe.listening_gain_db,
    )
