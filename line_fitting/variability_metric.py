"""Variability detection: V_{t,W} = log(s^2_{t,W} + 1), a sliding-window
(size W=10, step=1) variance-based metric on the RAW z-score - a companion
to trend_score_windowed.py's D(t,W) (slope-based), using the window's own
variance instead of a line fit. Fit directly to raw z, not the
rolling-mean-smoothed rolling_z_w10 - a rolling mean suppresses variance by
construction (a W-sample mean of near-independent values has roughly 1/W
the variance of the raw series), so computing variance on top of it would
mostly measure how much the already-smoothed trend wobbles, not how noisy
the raw signal actually is.

V = log(variance + 1) is 0 exactly when a window has zero variance and
increases monotonically as variance grows - no epsilon hack, no U-shape
(unlike an earlier |log(variance + eps)| version).

Detection threshold h_V is a ground truth built PER COUNTER from the 20
normal trials, the same construction as trend_score's h_D:
    for each normal trial j: M_j = max over all window positions of V_{t,W}
    h_V = 95th percentile of {M_1, ..., M_20}

Every window of every attack-mode (jump/drift/replay) run is then compared
against its own counter's h_V; V_{t,W} > h_V flags that window as a
variability-deviation / attack indication.

Reads line_fitting_timeseries.csv (raw z per counter/mode/seed/
sample_index, from line_fitting_analysis.py).
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TIMESERIES_PATH = os.path.join(SCRIPT_DIR, "line_fitting_timeseries.csv")
VARIATION_OUT_PATH = os.path.join(SCRIPT_DIR, "variation_windowed_results.csv")
THRESHOLD_OUT_PATH = os.path.join(SCRIPT_DIR, "variation_threshold_by_counter.csv")
FLAGGED_OUT_PATH = os.path.join(SCRIPT_DIR, "variation_attack_flags.csv")

WINDOW = 10
STEP = 1
PERCENTILE = 95
ATTACK_MODES = ["drift", "jump", "replay"]


def compute_variation(z_values):
    """z_values: rolling_z_w10 for one run (leading entries may be NaN).
    Returns a list of (window_start, variance, V) for every full,
    NaN-free window."""
    results = []
    n = len(z_values)
    for start in range(0, n - WINDOW + 1, STEP):
        window = z_values[start:start + WINDOW]
        if np.any(np.isnan(window)):
            continue
        variance = np.var(window, ddof=1)
        v = np.log1p(variance)
        results.append((start, variance, v))
    return results


def compute_all_windows(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("sample_index")
        z = group["z"].to_numpy(dtype=float)
        sample_index = group["sample_index"].to_numpy()
        for start, variance, v in compute_variation(z):
            window_end = sample_index[start + WINDOW - 1]
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_sample_index": window_end,
                "variance": variance, "V": v,
            })
    return pd.DataFrame(rows)


def build_thresholds(variation_df):
    """h_V per counter: 95th percentile of each normal trial's own max V."""
    normal = variation_df[variation_df["mode"] == "normal"]
    per_trial_max = normal.groupby(["counter", "seed"])["V"].max().reset_index(name="M_j")
    thresholds = (
        per_trial_max.groupby("counter")["M_j"]
        .quantile(PERCENTILE / 100)
        .reset_index(name="h_V")
    )
    return thresholds, per_trial_max


def main():
    ts = pd.read_csv(TIMESERIES_PATH)

    variation_df = compute_all_windows(ts)
    variation_df.to_csv(VARIATION_OUT_PATH, index=False)
    print(f"Saved {VARIATION_OUT_PATH} ({len(variation_df)} rows)")

    thresholds, _ = build_thresholds(variation_df)
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print(f"\nh_V (95th percentile of 20 normal trials' max V), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack = variation_df[variation_df["mode"].isin(ATTACK_MODES)].merge(thresholds, on="counter")
    attack["flagged"] = attack["V"] > attack["h_V"]
    attack.to_csv(FLAGGED_OUT_PATH, index=False)
    print(f"\nSaved {FLAGGED_OUT_PATH} ({len(attack)} rows)")

    print("\nPer-run detection summary (a run counts as detected if ANY window exceeds h_V):")
    run_detected = attack.groupby(["counter", "mode", "seed"])["flagged"].any().reset_index(name="detected")
    summary = run_detected.groupby(["counter", "mode"])["detected"].agg(n_detected="sum", n_runs="count")
    print(summary.to_string())


if __name__ == "__main__":
    main()
