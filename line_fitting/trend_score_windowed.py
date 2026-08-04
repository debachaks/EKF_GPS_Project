"""Windowed trend score D(t,W) - sliding window (size W=10, step=1), fit
DIRECTLY to the raw z-score (not the rolling-mean-smoothed rolling_z_w10),
giving a D(t,W) time series per run. One window in, one D out - no window
fit on top of an already-windowed/smoothed series (see trend_score.py for
an earlier single-scalar-per-run version, superseded by this window-wise
one).

For each run (every counter/mode/seed combo), slides a window of W=10
consecutive RAW z values (step=1) across the run. Within each window
position, fits a local line

    z_i ~ alpha + beta * i        (i = 0..W-1, local index)

Shifting the x-axis by a constant doesn't change the OLS slope, only the
intercept, so using a local 0..9 index instead of the run's global sample
index has no effect on beta or D - it only changes alpha, which isn't used
here.

    e_i         = z_i - fitted_i
    sigma_eps^2 = sum(e_i^2) / (W - 2)
    SE(beta)    = sqrt(sigma_eps^2 / sum((i - i_bar)^2))
    D           = |beta / SE(beta)|

Windows containing any NaN z value are skipped (raw z has none in
practice, since the shared elapsed-time grid is built to avoid
extrapolation - unlike rolling_z_w10, which has NaN for the first 9
samples of every run before the rolling mean has enough history).

Detection threshold h_D is a ground truth built PER COUNTER from the 20
normal trials:
    for each normal trial j: M_j = max over all window positions of D(t,W)
    h_D = 95th percentile of {M_1, ..., M_20}

Every window of every attack-mode (jump/drift/replay) run is then compared
against its own counter's h_D; D(t,W) > h_D flags that window as a
trend-deviation / attack indication.

Reads line_fitting_timeseries.csv (already has raw z per
counter/mode/seed/sample_index from line_fitting_analysis.py).
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "line_fitting_timeseries.csv")
D_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_windowed_results.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_threshold_by_counter.csv")
FLAGGED_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_attack_flags.csv")

WINDOW = 10
STEP = 1
PERCENTILE = 95
ATTACK_MODES = ["drift", "jump", "replay"]

_S = np.arange(WINDOW, dtype=float)
_S_BAR = _S.mean()
_S_DEV = _S - _S_BAR
_SS_S = np.sum(_S_DEV**2)


def windowed_d(z_values):
    """z_values: rolling_z_w10 for one run (leading entries may be NaN).
    Returns a list of (window_start, beta, D) for every full, NaN-free
    window."""
    n = len(z_values)
    results = []
    for start in range(0, n - WINDOW + 1, STEP):
        window = z_values[start:start + WINDOW]
        if np.any(np.isnan(window)):
            continue
        beta = np.sum(_S_DEV * window) / _SS_S
        alpha = window.mean() - beta * _S_BAR
        fitted = alpha + beta * _S
        resid = window - fitted
        sigma_eps2 = np.sum(resid**2) / (WINDOW - 2)
        se_beta = np.sqrt(sigma_eps2 / _SS_S)
        d = abs(beta / se_beta) if se_beta > 0 else np.nan
        results.append((start, beta, d))
    return results


def compute_all_windows(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("sample_index")
        z_values = group["z"].to_numpy()
        sample_indices = group["sample_index"].to_numpy()
        for start, beta, d in windowed_d(z_values):
            window_end_t = sample_indices[start + WINDOW - 1]
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_sample_index": window_end_t,
                "beta_local": beta, "D": d,
            })
    return pd.DataFrame(rows)


def build_thresholds(d_df):
    """h_D per counter: 95th percentile of each normal trial's own max D."""
    normal = d_df[d_df["mode"] == "normal"]
    per_trial_max = normal.groupby(["counter", "seed"])["D"].max().reset_index(name="M_j")
    thresholds = (
        per_trial_max.groupby("counter")["M_j"]
        .quantile(PERCENTILE / 100)
        .reset_index(name="h_D")
    )
    return thresholds, per_trial_max


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = pd.read_csv(TIMESERIES_PATH)

    d_df = compute_all_windows(ts)
    d_df.to_csv(D_OUT_PATH, index=False)
    print(f"Saved {D_OUT_PATH} ({len(d_df)} rows)")

    thresholds, _ = build_thresholds(d_df)
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print(f"\nh_D (95th percentile of 20 normal trials' max D), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack = d_df[d_df["mode"].isin(ATTACK_MODES)].merge(thresholds, on="counter")
    attack["flagged"] = attack["D"] > attack["h_D"]
    attack.to_csv(FLAGGED_OUT_PATH, index=False)
    print(f"\nSaved {FLAGGED_OUT_PATH} ({len(attack)} rows)")

    print("\nPer-run detection summary (a run counts as detected if ANY window exceeds h_D):")
    run_detected = attack.groupby(["counter", "mode", "seed"])["flagged"].any().reset_index(name="detected")
    summary = run_detected.groupby(["counter", "mode"])["detected"].agg(n_detected="sum", n_runs="count")
    print(summary.to_string())


if __name__ == "__main__":
    main()
