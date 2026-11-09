"""Figures shared by the app, the notebooks and the report."""

from .figures import (
    array_layout,
    azimuth_pattern,
    confirmation_curves,
    detection_map,
    direct_levels_vs_baseline,
    dynamic_range_ladder,
    elevation_pattern,
    gain_window_vs_bits,
    lineup_levels,
    noise_spread_vs_frequency,
    noise_vs_frequency,
    pd_vs_range,
    pd_vs_snr,
    range_accuracy_vs_snr,
    sweep_line,
    threshold_trade_snr,
    tracker_rmse,
)

__all__ = [
    "array_layout", "azimuth_pattern", "confirmation_curves", "detection_map", "direct_levels_vs_baseline",
    "dynamic_range_ladder", "elevation_pattern", "gain_window_vs_bits", "lineup_levels", "noise_spread_vs_frequency", "noise_vs_frequency", "pd_vs_range",
    "pd_vs_snr", "range_accuracy_vs_snr", "sweep_line", "threshold_trade_snr", "tracker_rmse",
]
