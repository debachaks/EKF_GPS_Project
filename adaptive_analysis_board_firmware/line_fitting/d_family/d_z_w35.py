"""D and z_stat only (no V this time), computed at WINDOW=35 -- continuing
the W=5/10/20 window-size sweep for trend score D, plus a check on z_stat
at yet another window size (expected to be a no-op per the max-invariance
property already confirmed at W=5/10/20).

Same math as trend_score_windowed.py / combined_detection.py's windowed_z,
just at W=35, with its own freshly-calibrated thresholds (never reuse a
threshold computed at a different window size -- see prior discussion).
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")

D_OUT_PATH = os.path.join(RESULTS_DIR, "d_z_windowed_w35.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, "d_z_thresholds_w35.csv")
FLAGGED_OUT_PATH = os.path.join(RESULTS_DIR, "d_z_attack_flags_w35.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, "d_z_detection_summary_w35.csv")

WINDOW = 35
STEP = 1
PERCENTILE = 95
EPS = 1e-20

_S = np.arange(WINDOW, dtype=float)
_S_BAR = _S.mean()
_S_DEV = _S - _S_BAR
_SS_S = np.sum(_S_DEV**2)


def windowed_d_and_z(z_values):
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
        z_stat = np.max(np.abs(window))
        results.append((start, d, z_stat))
    return results


def compute_all_windows(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z_values = group["z"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, d, z_stat in windowed_d_and_z(z_values):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + WINDOW - 1],
                "D": d, "z_stat": z_stat,
            })
    return pd.DataFrame(rows)


def build_thresholds(df):
    normal = df[df["mode"] == "normal"]
    per_run_max = normal.groupby(["counter", "seed"])[["D", "z_stat"]].max().reset_index()
    thresholds = (
        per_run_max.groupby("counter")[["D", "z_stat"]]
        .quantile(PERCENTILE / 100)
        .rename(columns={"D": "h_D", "z_stat": "h_z"})
        .reset_index()
    )
    return thresholds


def main():
    ts = pd.read_csv(TIMESERIES_PATH)

    df = compute_all_windows(ts)
    df.to_csv(D_OUT_PATH, index=False)
    print(f"Saved {D_OUT_PATH} ({len(df)} rows)")

    thresholds = build_thresholds(df)
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print("\nThresholds (W=35), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack = df[df["mode"].isin(["jump", "drift"])].merge(thresholds, on="counter")
    attack["d_flag"] = attack["D"] > attack["h_D"]
    attack["z_flag"] = attack["z_stat"] > attack["h_z"]
    attack["flag_or"] = attack["d_flag"] | attack["z_flag"]
    attack.to_csv(FLAGGED_OUT_PATH, index=False)
    print(f"\nSaved {FLAGGED_OUT_PATH} ({len(attack)} rows)")

    run_detected = attack.groupby(["counter", "mode", "seed"])[["d_flag", "z_flag", "flag_or"]].any().reset_index()
    summary = run_detected.groupby(["counter", "mode"]).agg(
        n_runs=("seed", "count"),
        n_D=("d_flag", "sum"),
        n_z=("z_flag", "sum"),
        n_or=("flag_or", "sum"),
    ).reset_index()
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print("\nPer-run detection summary (W=35):")
    print(summary.to_string(index=False))
    print(f"Saved {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
