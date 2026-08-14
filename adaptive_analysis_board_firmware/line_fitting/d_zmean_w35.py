"""Corrected version of d_z_w35.py's "z" metric: the windowed MEAN of the
raw z-score (rolling_z, same idea as the original line_fitting_analysis.py's
rolling_z(window=W).mean()), NOT z_stat=max(|z|) from combined_detection.py.

Since mean blends every sample in the window together (unlike max, which
is immune to dilution -- see the earlier max-vs-average discussion), this
version of "z" IS expected to be sensitive to window size the same way D
is, just via smoothing/dilution of the underlying z trace instead of
dilution of a fitted slope.

    z_mean(window) = mean(z_i)  for i in the window

Threshold h_zmean per counter = z_metric.py's flat-pooled Q95 (95th
percentile of every raw z value pooled across every normal trial's every
iteration) -- NOT the per-trial-max-then-percentile method still used for
h_D. See z_metric.py's docstring for why z is deliberately calibrated
differently from D. h_zmean is independent of WINDOW (built from raw,
unwindowed z), unlike h_D.
"""

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from z_metric import build_zmean_threshold  # noqa: E402

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")

WINDOW = 35
STEP = 1
PERCENTILE = 95

D_OUT_PATH = os.path.join(RESULTS_DIR, f"d_zmean_windowed_w{WINDOW}.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, f"d_zmean_thresholds_w{WINDOW}.csv")
FLAGGED_OUT_PATH = os.path.join(RESULTS_DIR, f"d_zmean_attack_flags_w{WINDOW}.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, f"d_zmean_detection_summary_w{WINDOW}.csv")

EPS = 1e-20

_S = np.arange(WINDOW, dtype=float)
_S_BAR = _S.mean()
_S_DEV = _S - _S_BAR
_SS_S = np.sum(_S_DEV**2)


def windowed_d_and_zmean(z_values):
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
        z_mean = window.mean()
        results.append((start, d, z_mean))
    return results


def compute_all_windows(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z_values = group["z"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, d, z_mean in windowed_d_and_zmean(z_values):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + WINDOW - 1],
                "D": d, "z_mean": z_mean,
            })
    return pd.DataFrame(rows)


def build_thresholds(df, ts):
    normal = df[df["mode"] == "normal"]
    per_run_max = normal.groupby(["counter", "seed"])["D"].max().reset_index()
    thresholds = (
        per_run_max.groupby("counter")["D"]
        .quantile(PERCENTILE / 100)
        .reset_index(name="h_D")
    )
    h_zmean = build_zmean_threshold(ts, counters=thresholds["counter"].unique()).rename(
        columns={"h_z": "h_zmean"}
    )
    thresholds = thresholds.merge(h_zmean, on="counter")[["counter", "h_zmean", "h_D"]]
    return thresholds


def main():
    ts = pd.read_csv(TIMESERIES_PATH)

    df = compute_all_windows(ts)
    df.to_csv(D_OUT_PATH, index=False)
    print(f"Saved {D_OUT_PATH} ({len(df)} rows)")

    thresholds = build_thresholds(df, ts)
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print(f"\nThresholds (W={WINDOW}), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack = df[df["mode"].isin(["jump", "drift"])].merge(thresholds, on="counter")
    attack["d_flag"] = attack["D"] > attack["h_D"]
    attack["zmean_flag"] = attack["z_mean"].abs() > attack["h_zmean"]
    attack["flag_or"] = attack["d_flag"] | attack["zmean_flag"]
    attack.to_csv(FLAGGED_OUT_PATH, index=False)
    print(f"\nSaved {FLAGGED_OUT_PATH} ({len(attack)} rows)")

    run_detected = attack.groupby(["counter", "mode", "seed"])[["d_flag", "zmean_flag", "flag_or"]].any().reset_index()
    summary = run_detected.groupby(["counter", "mode"]).agg(
        n_runs=("seed", "count"),
        n_D=("d_flag", "sum"),
        n_zmean=("zmean_flag", "sum"),
        n_or=("flag_or", "sum"),
    ).reset_index()
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print(f"\nPer-run detection summary (W={WINDOW}):")
    print(summary.to_string(index=False))
    print(f"Saved {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
