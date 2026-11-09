# Bi-static radar link budget

Takes one Tx-Rx bi-static radar design from physics to finalized control parameters:

**antennas -> noise floor -> direct path and RF front-end -> target detection and estimation -> tracker gain -> report**

Every model states why it was used or rejected, and the report records how far each standard was actually checked.

## Set up and run

Use a short folder path on Windows, since some Python packages have very long internal paths.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\pip install -e ".[dev]"
```

Start the app:

```bash
.venv\Scripts\streamlit run app/Home.py
```

The running app picks up edits to the library and to the pages by itself; no restart is needed.

Run the tests:

```bash
.venv\Scripts\pytest
```

## Using it

| You want to | Go to |
|---|---|
| Explore a design and see trade-offs | The app: pages follow the stages in order |
| Save a scenario and reopen it later | Save scenario in the app's sidebar writes `scenarios/saved/<name>.yaml`; load it from the Home page |
| Understand the theory and checks | `notebooks/01` to `06`, saved with their outputs |
| Script it | `run_link_budget(Scenario.from_yaml("scenarios/dvbt_uhf.yaml"))` |
| Produce a report | App's Report page, or `build_report`, `write_docx` and `write_markdown` |

An example report is in [reports/example_dvbt.md](reports/example_dvbt.md).

```python
from bistatic_lb import Scenario, run_link_budget

s = Scenario.from_yaml("scenarios/dvbt_uhf.yaml").with_value("array.n_circular", 24)
r = run_link_budget(s)
print(r.detection.pd, r.frontend.gain_window_db, r.estimation.position_rms_m)
```

## How the code is organized

| Folder | Holds |
|---|---|
| `bistatic_lb/config.py` | Every input, range-checked; saves and loads YAML |
| `bistatic_lb/pipeline.py` | `run_link_budget`: the one entry point, connecting the stages |
| `bistatic_lb/antenna/` | Monopole, dipole, folded dipole; ring and column array; beamforming |
| `bistatic_lb/noise/` | P.372 external noise, pattern integration, receiver line-up |
| `bistatic_lb/propagation/` | P.525, P.526, P.676, P.838, P.530, P.2108, terrain sources |
| `bistatic_lb/frontend/` | ADC, gain window, derived requirements |
| `bistatic_lb/waveforms/` | Pulsed LFM, phase code, Zadoff-Chu, FM, DAB, DVB-T |
| `bistatic_lb/target/` | RCS, detection, link budget, estimation bounds, Monte Carlo checks |
| `bistatic_lb/tracker/` | M-of-N, filters, PCRLB, simulation |
| `bistatic_lb/plots/`, `report/` | Shared figures; Word and Markdown output |
| `app/` | Streamlit pages: inputs plus library calls, no physics |

See [EXTENDING.md](EXTENDING.md) for how to add a waveform, antenna, propagation model, DEM terrain or a receiver.

## What was checked, and what wasn't

**Standards:**

| Source | Status |
|---|---|
| P.372-18, P.526-16, P.530-19, P.2108-1 | Equations read from these editions. Some P.372 values are read from its figures |
| P.676, P.838 | Through the ITU-Rpy package, which implements P.676-12 and P.838-3. P.676-13 exists |
| P.453, P.527 | Only simple relations and representative ground values, not their full models |

**Inputs you should set for your site**, because they have no reliable default here:

- P.530 geoclimatic factor `log10_k_geoclimatic` and `dn75`: the ITU data grids aren't bundled.
- Man-made noise above 250 MHz: P.372-18 eq. 17 stops there. The tool continues at the slope of Fig. 3, which is for city noise and stops near 900 MHz. This is flagged as an estimate, so enter a measurement if it matters.

**Modelling limits:**

- Noise spread uses P.372-18 Part 7 for man-made and galactic noise only, with other sources held constant. Table 2 deviations are averages for 0.3 to 250 MHz and may be pessimistic above that, so enter measured deviations if you have them.
- A target that crosses range or Doppler cells in one coherent interval loses integration gain, using closed forms for an unweighted flat spectrum and straight, constant-speed flight. A manoeuvre adds Doppler walk that is not included. Set the migration compensation if your processor follows the target.
- LO phase noise and third-order distortion of the direct signal are counted in the target's SINR as planning estimates. Stage compression is not modelled: the tool marks the results as not valid instead.
- Ground clutter is summed over the ring that shares the target's bi-static delay, on flat ground with one reflectivity and the smooth-Earth propagation loss. Discrete scatterers, wind-driven spread and range-ambiguous clutter are not included.
- The digital canceller is assumed to reach the depth needed, up to its stated limit, for both the direct signal and clutter. Its zero-Doppler notch sets the minimum detectable range rate.
- Ground reflection on the target paths is phase-averaged by default. Choose the actual phase to see the lobing; the detection map samples it coarsely.
- The tracker simulation starts each track at the target's first detection and applies the M-of-N test. Tracks started by false alarms are not simulated (their rate is analytic), and a lost track is not restarted.
- Without elevation measurement the position fix assumes an altitude. Set the altitude uncertainty to include its effect.
- Mutual coupling is neglected in the array. The tool warns when this could matter.
- The finite ground-plane correction, the off-tune mismatch model and the waveforms' structured sidelobe levels are planning estimates.
- The maximum direct-path level is an engineering bound, not a statistic.
- Accuracy results reflect noise only. Timing, drift and target extent add error floors in a real system.
- Forward-scatter RCS is an approximation for compact shapes.
- One Tx-Rx pair only. DEM lookup and multi-static operation are extension points.

**Tests:** model values are checked against published numbers, Monte Carlo simulation and generated waveforms.
