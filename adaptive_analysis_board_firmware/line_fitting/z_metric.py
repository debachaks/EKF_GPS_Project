"""Canonical definition of the z metric, per the whiteboard design: this
REPLACES the old z_stat = max(|z_i|) (combined_detection.py's original
windowed_z) with a rolling MEAN, and replaces the old hierarchical
per-trial-max threshold with a flat-pooled percentile. Every script that
defines or uses the z metric should import from here rather than
reimplementing it, so there's exactly one definition to keep in sync.

    z_bar(t, W) = mean(z_i)   for i in the W-sample window ending at t

Unlike z_stat=max (immune to dilution) or D (which fits a line), the mean
blends every sample in the window together -- a sustained shift stays
visible, but a single-point spike gets diluted by the rest of the window.

THRESHOLD -- deliberately built differently from h_D/h_V:
    Pool every RAW per-iteration z value (not windowed) from every normal
    trial for a counter -- 20 seeds x n_iters raw z_t values -- into one
    flat list, sort it, and take the 95th percentile of THAT pooled
    distribution:
        Q95 = 95th percentile of {z_1, z_2, ..., z_(20 x n_iters)}
    This is NOT the same as h_D/h_V's "95th percentile of each trial's
    own max, across trials" -- it's a single flat pool across every
    sample of every trial. Note this makes Q95 independent of window
    size W: it's built from the raw z series, never from z_bar itself.

Detection rule: |z_bar(t, W)| > Q95 => level-deviation / attack
indication.
"""

import numpy as np
import pandas as pd

PERCENTILE = 95


def windowed_zmean(z_values, window, step=1):
    """z_values: raw z for one run. Returns a list of (window_start,
    z_bar) for every full, NaN-free window of the given size."""
    n = len(z_values)
    results = []
    for start in range(0, n - window + 1, step):
        w = z_values[start:start + window]
        if np.any(np.isnan(w)):
            continue
        results.append((start, w.mean()))
    return results


def build_zmean_threshold(ts, counters=None):
    """ts: the long-format zscore_timeseries.csv dataframe (columns
    counter, mode, seed, iter, raw_value, z). Returns a per-counter Q95,
    the 95th percentile of ALL raw z values pooled across every normal
    trial's every iteration -- independent of window size, so it's
    computed once and reused by every W variant of the z metric.
    """
    normal = ts[ts["mode"] == "normal"]
    if counters is not None:
        normal = normal[normal["counter"].isin(counters)]
    thresholds = (
        normal.groupby("counter")["z"]
        .apply(lambda s: np.percentile(np.abs(s.to_numpy()), PERCENTILE))
        .reset_index(name="h_z")
    )
    return thresholds
