"""What limits the digital canceller that removes the direct signal and stationary clutter.

An ECA-type canceller subtracts delayed copies of a reference from the surveillance signal,
one copy (tap) per range cell it covers. Three things bound what it achieves.

1. Fidelity of the reference. The canceller can only remove what its reference contains.

       known waveform    the receiver regenerates the transmitted signal. The real emission
                         differs from that replica by the transmitter's own error (amplifier
                         distortion, filtering, noise): its modulation error ratio.
       reference beam    a beam on the transmitter supplies the reference off air. It contains
                         the transmitter's error, so that is not a limit, but it carries the
                         receiver's noise: its direct-to-noise ratio is the fidelity.

   The part of the direct signal the reference lacks is noise-like. It stays in the
   surveillance signal at P_direct / fidelity, spread over the bandwidth, and so reaches
   every range-Doppler cell at

       P_direct - fidelity - 10 log10(B T)

   whatever the cancellation depth. The same residue appears in the matched filter itself,
   so it applies even when no cancellation is needed. The spatial null lowers it, since it
   arrives from the transmitter's direction.

2. Stability and hardware. Drift of the channel, the clocks and the gains over one coherent
   interval moves energy off zero Doppler, out of reach of zero-Doppler taps. This stays a
   stated limit on the depth (``digital_canceller_limit_db``).

3. Range extent. Clutter is removed only in range cells that have a tap. Clutter sharing the
   target's range cell beyond the covered extent is left to the Doppler filter alone. The
   tap count also sets the size of the least-squares problem, and estimating K taps from
   N = B T samples costs -10 log10(1 - K / N) of signal.

Not modelled: multipath and targets in the reference beam, clutter's own residue from
reference error, and batch or Doppler-extended variants that trade a wider notch for
tolerance to drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import C_M_S


@dataclass(frozen=True)
class CancellerModel:
    reference: str  # "known_waveform" or "reference_beam", after resolving "auto"
    fidelity_db: float  # reference-to-error ratio
    fidelity_basis: str  # where that number comes from, in words
    range_extent_m: float  # bi-static extra path covered by taps
    taps: int
    estimation_loss_db: float
    depth_limit_db: float  # stability and hardware

    def covers(self, bistatic_range_m: float) -> bool:
        return bistatic_range_m <= self.range_extent_m

    def fidelity_floor_dbm(self, direct_at_output_dbm: float, bandwidth_hz: float, integration_s: float) -> float:
        """Level in any range-Doppler cell of the direct signal's part that the reference lacks."""
        return float(direct_at_output_dbm - self.fidelity_db - 10.0 * np.log10(bandwidth_hz * integration_s))


def build_canceller(
    reference: str,
    transmitter_fidelity_db: float,
    reference_beam_dnr_db: float,
    third_party_transmitter: bool,
    range_extent_m: float,
    bandwidth_hz: float,
    integration_s: float,
    depth_limit_db: float,
) -> CancellerModel:
    """Resolve the reference choice and size the canceller. 'auto' uses a reference beam for a third-party transmitter."""
    if reference == "auto":
        reference = "reference_beam" if third_party_transmitter else "known_waveform"
    if reference == "reference_beam":
        fidelity, basis = reference_beam_dnr_db, "direct-to-noise ratio of a beam on the transmitter"
    else:
        fidelity, basis = transmitter_fidelity_db, "transmitter fidelity (modulation error ratio)"
    taps = max(int(np.ceil(range_extent_m / (C_M_S / bandwidth_hz))), 1)
    samples = max(bandwidth_hz * integration_s, 1.0)
    used = min(taps / samples, 0.999)
    return CancellerModel(
        reference=reference,
        fidelity_db=float(fidelity),
        fidelity_basis=basis,
        range_extent_m=float(range_extent_m),
        taps=taps,
        estimation_loss_db=float(-10.0 * np.log10(1.0 - used)),
        depth_limit_db=float(depth_limit_db),
    )
