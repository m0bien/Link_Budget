"""Bibliography, and how far each standard was actually checked.

``status`` says what was done, so that a report never overstates it:

* ``checked``  - equations were read from this edition's PDF while coding.
* ``older``    - equations were read from the older edition named here. The
                 in-force edition could not be retrieved; re-check before relying
                 on it for a formal design.
* ``library``  - implemented by the ITU-Rpy package at the edition it implements.
* ``values``   - only representative values are used, not the Recommendation's
                 full method.
* ``textbook`` - general reference; formulas are standard results.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reference:
    key: str
    citation: str
    status: str
    note: str = ""


_ALL = [
    Reference("P.372", "Rec. ITU-R P.372-18 (09/2026), Radio noise", "checked",
              "Fig. 3 values are read from the curves, to about 1 dB. Man-made noise above 250 MHz is an estimate. "
              "The clear-air Tmr data file is not bundled. Part 7 is applied to man-made and galactic noise only."),
    Reference("P.525", "Rec. ITU-R P.525-5 (11/2024), Calculation of free-space attenuation",
              "textbook", "Standard Friis result."),
    Reference("P.526", "Rec. ITU-R P.526-16 (11/2025), Propagation by diffraction", "checked"),
    Reference("P.530", "Rec. ITU-R P.530-19 (09/2025), Propagation data and prediction methods "
              "for terrestrial line-of-sight systems", "checked",
              "The LogK.csv and dN75.csv grids are not bundled, so K and dN75 are user inputs."),
    Reference("P.2108", "Rec. ITU-R P.2108-1 (09/2021), Prediction of clutter loss", "checked"),
    Reference("P.676", "Rec. ITU-R P.676, Attenuation by atmospheric gases", "library",
              "ITU-Rpy implements P.676-12."),
    Reference("P.838", "Rec. ITU-R P.838-3, Specific attenuation model for rain", "library"),
    Reference("P.453", "Rec. ITU-R P.453-14 (08/2019), The radio refractive index", "values",
              "Only the k = 157/(157 + dN) relation and the duct threshold dN/dh < -157 N/km."),
    Reference("P.527", "Rec. ITU-R P.527-6 (09/2021), Electrical characteristics of the surface "
              "of the Earth", "values",
              "Representative permittivity and conductivity per ground class, not the "
              "frequency-dependent soil model of P.527-6."),
    Reference("Willis", "N. J. Willis, Bistatic Radar, 2nd ed., SciTech, 2005", "textbook"),
    Reference("Griffiths", "H. D. Griffiths and C. J. Baker, An Introduction to Passive Radar, "
              "Artech House, 2017", "textbook"),
    Reference("Richards", "M. A. Richards, Fundamentals of Radar Signal Processing, 2nd ed., "
              "McGraw-Hill, 2014", "textbook"),
    Reference("Shnidman", "D. A. Shnidman, Determination of required SNR values, IEEE Trans. AES "
              "38(3), 2002", "textbook"),
    Reference("Albersheim", "W. J. Albersheim, A closed-form approximation to Robertson's "
              "detection characteristics, Proc. IEEE 69(7), 1981", "textbook"),
    Reference("Swerling", "P. Swerling, Probability of detection for fluctuating targets, "
              "IRE Trans. IT-6, 1960", "textbook"),
    Reference("VanTrees", "H. L. Van Trees, Detection, Estimation and Modulation Theory, "
              "Parts I and IV, Wiley", "textbook"),
    Reference("BarShalom", "Y. Bar-Shalom, X. R. Li and T. Kirubarajan, Estimation with "
              "Applications to Tracking and Navigation, Wiley, 2001", "textbook"),
    Reference("Kalata", "P. R. Kalata, The tracking index, IEEE Trans. AES 20(2), 1984", "textbook"),
    Reference("Hernandez", "M. L. Hernandez, A. Farina and B. Ristic, A PCRLB for tracking in "
              "cluttered environments, IEEE Trans. AES 42(3), 2006", "textbook"),
    Reference("Balanis", "C. A. Balanis, Antenna Theory, 4th ed., Wiley, 2016", "textbook"),
]

REFERENCES: dict[str, Reference] = {r.key: r for r in _ALL}
