"""Windowed trend score D(t,W) for the newMapping dataset -- direct port of
line_fitting/trend_score_windowed.py's math (sliding-window OLS slope
t-statistic), just reading zscore_baseline.py's raw z-score output instead
of the original's interpolated-grid version.

For each run (every counter/mode/seed), slides a window of W=10 consecutive
RAW z values (step=1). Within each window position, fits a local line
z_i ~ alpha + beta*i (i = 0..W-1, local index) and computes

    D = |beta / SE(beta)|

Detection threshold h_D is built PER COUNTER from the normal-mode baseline
runs only:
    for each normal run j: M_j = max over all window positions of D
    h_D = 95th percentile of {M_1, ..., M_n}

Every window of every attack-mode run present (whatever modes exist in
zscore_timeseries.csv besides "normal" -- currently jump/drift on
seed2/seed3, more seeds to follow as they're added) is compared against its
own counter's h_D; D > h_D flags that window as a trend-deviation.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")
D_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_windowed_results.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_threshold_by_counter.csv")
FLAGGED_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_attack_flags.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_detection_summary.csv")

WINDOW = 10
STEP = 1
PERCENTILE = 95
EPS = 1e-20   # avoids exact 0/0 (NaN) when a window is perfectly flat
              # (sigma_eps2==0, beta==0 too in every observed case in this
              # dataset -- so D naturally comes out 0, not undefined).
              # Negligible next to any real nonzero variance: the smallest
              # observed genuine (nonzero) se_beta in this data corresponds
              # to sigma_eps2 ~ 1e-36 * SS_S, and typical values are
              # ~1e-7 * SS_S -- EPS=1e-20 sits far below both, so it can
              # only ever matter for windows that were already exactly 0.

_S = np.arange(WINDOW, dtype=float)
_S_BAR = _S.mean()
_S_DEV = _S - _S_BAR
_SS_S = np.sum(_S_DEV**2)


def windowed_d(z_values):
    """z_values: raw z for one run. Returns a list of (window_start, beta, D)
    for every full, NaN-free window."""
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
        se_beta = np.sqrt(sigma_eps2 / _SS_S + EPS)
        d = abs(beta / se_beta)
        results.append((start, beta, d))
    return results


def compute_all_windows(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z_values = group["z"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, beta, d in windowed_d(z_values):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + WINDOW - 1],
                "beta_local": beta, "D": d,
            })
    return pd.DataFrame(rows)


def build_thresholds(d_df):
    """h_D per counter: 95th percentile of each normal run's own max D."""
    normal = d_df[d_df["mode"] == "normal"]
    per_run_max = normal.groupby(["counter", "seed"])["D"].max().reset_index(name="M_j")
    thresholds = (
        per_run_max.groupby("counter")["M_j"]
        .quantile(PERCENTILE / 100)
        .reset_index(name="h_D")
    )
    return thresholds


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = pd.read_csv(TIMESERIES_PATH)

    d_df = compute_all_windows(ts)
    d_df.to_csv(D_OUT_PATH, index=False)
    print(f"Saved {D_OUT_PATH} ({len(d_df)} rows)")

    thresholds = build_thresholds(d_df)
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print("\nh_D (95th percentile of normal runs' max D), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack_modes = sorted(ts.loc[ts["mode"] != "normal", "mode"].unique())
    if not attack_modes:
        print("\nNo attack-mode runs present yet -- skipping flagging step.")
        return

    attack = d_df[d_df["mode"].isin(attack_modes)].merge(thresholds, on="counter")
    attack["flagged"] = attack["D"] > attack["h_D"]
    attack.to_csv(FLAGGED_OUT_PATH, index=False)
    print(f"\nSaved {FLAGGED_OUT_PATH} ({len(attack)} rows)")

    run_detected = attack.groupby(["counter", "mode", "seed"])["flagged"].any().reset_index(name="detected")
    summary = run_detected.groupby(["counter", "mode"])["detected"].agg(n_detected="sum", n_runs="count").reset_index()
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print("\nPer-run detection summary (a run counts as detected if ANY window exceeds h_D):")
    print(summary.to_string(index=False))
    print(f"Saved {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
