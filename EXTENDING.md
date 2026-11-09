# Extending the tool

Each extension follows the same pattern: add one file, and register it in one place. Physics never lives in the app.

## Add a waveform

1. Subclass `Waveform` in a new file in `bistatic_lb/waveforms/`. Define `bandwidth_hz`, `papr_db`, `duty_cycle`, `direct_leakage_db` and `generate`.
2. Add it to `WAVEFORMS` in `bistatic_lb/waveforms/__init__.py`.
3. Add its key to `WaveformConfig.kind` in `config.py`, and to the list on `app/pages/1_Scenario.py`.

`tests/test_pipeline.py::test_every_waveform_runs_end_to_end` picks it up automatically.

## Add a target fluctuation model

Add an entry to `FLUCTUATION_MODELS` in `bistatic_lb/target/detection.py`, or a branch in `make_fluctuation` if it has parameters. Any distribution of total signal power works in `detection_probability`, because the detector depends on the signal only through that total.

## Add a propagation mechanism

1. Write the model in a new file in `bistatic_lb/propagation/`.
2. In `path_terms` in `propagation/budget.py`, evaluate it and append a `Contribution` with its value, whether it applies, and why. Add its value to `PathTerms`.

The app and the report show the new row with no other change. P.452, P.528 and P.1546 are natural candidates.

## Add DEM terrain lookup

Implement `DemTerrain.profile` in `bistatic_lb/propagation/terrain.py`: sample terrain height along the great circle between the terminals, and return distances in km and heights above sea level in m. Then return it from `terrain_from_scenario`, and add coordinates to `EnvironmentConfig`. The diffraction code already takes any `TerrainSource`.

## Use measured or EM-simulated antenna patterns

Subclass `ReceiveArray` and override `element_gain(u)`, which returns each element's linear gain toward unit vectors `u`. Using active-element patterns here is also how to bring in mutual coupling.

## Add an array layout

Build a `ReceiveArray` with your own `positions_m` and `axes`. Everything downstream, including noise integration and the angle bound, works from those.

## Add a receiver (multi-static)

The stages take a geometry and return results, so the natural route is:

1. Run `run_link_budget` once per Tx-Rx pair.
2. Fuse the position estimates using each pair's covariance from `position_covariance`.
3. Add a fused track filter in `bistatic_lb/tracker/`, using `GaussianTrack` with a stacked measurement function.

## Add a figure or report section

Add a function in `bistatic_lb/plots/figures.py` that returns a Matplotlib figure. Export it from `plots/__init__.py`, then call it from an app page, or add it as a `Figure` block in `report/builder.py`.

## Add an output format

Write a function that walks `report.blocks`, as `report/markdown.py` does. Each block is a `Heading`, `Paragraph`, `Bullets`, `Table` or `Figure`.

## Add a sweepable input

Add a line to `SWEEPABLE` in `bistatic_lb/sweeps.py`. Any field is reachable from code with `sweep(s, "section.field", values, metric)`.
