"""Same as trend_score_centered.py (D = |beta - mean_beta_normal(t)| /
SE(beta)), but with WINDOW=20 instead of 10 -- does widening the window
help the centered version the same way it did (or didn't) for the
original D? Outputs go to separate _w20-suffixed files; the W=10 centered
version is untouched.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")

D_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_centered_windowed_w20.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_centered_threshold_w20.csv")
FLAGGED_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_centered_attack_flags_w20.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_centered_detection_summary_w20.csv")

WINDOW = 20
STEP = 1
PERCENTILE = 95
EPS = 1e-20

_S = np.arange(WINDOW, dtype=float)
_S_BAR = _S.mean()
_S_DEV = _S - _S_BAR
_SS_S = np.sum(_S_DEV**2)


def windowed_beta_se(z_values):
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
        results.append((start, beta, se_beta))
    return results


def compute_all_windows(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z_values = group["z"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, beta, se_beta in windowed_beta_se(z_values):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + WINDOW - 1],
                "beta": beta, "se_beta": se_beta,
            })
    return pd.DataFrame(rows)


def main():
    ts = pd.read_csv(TIMESERIES_PATH)

    df = compute_all_windows(ts)

    normal = df[df["mode"] == "normal"]
    mean_beta_normal = (
        normal.groupby(["counter", "window_end_iter"])["beta"]
        .mean()
        .reset_index(name="mean_beta_normal")
    )

    df = df.merge(mean_beta_normal, on=["counter", "window_end_iter"], how="left")
    df["D"] = (df["beta"] - df["mean_beta_normal"]).abs() / df["se_beta"]
    df.to_csv(D_OUT_PATH, index=False)
    print(f"Saved {D_OUT_PATH} ({len(df)} rows)")

    normal_d = df[df["mode"] == "normal"]
    per_run_max = normal_d.groupby(["counter", "seed"])["D"].max().reset_index(name="M_j")
    thresholds = (
        per_run_max.groupby("counter")["M_j"]
        .quantile(PERCENTILE / 100)
        .reset_index(name="h_D")
    )
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print(f"\nh_D (centered D, W={WINDOW}) (95th percentile of normal runs' max D), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack = df[df["mode"].isin(["jump", "drift"])].merge(thresholds, on="counter")
    attack["flagged"] = attack["D"] > attack["h_D"]
    attack.to_csv(FLAGGED_OUT_PATH, index=False)
    print(f"\nSaved {FLAGGED_OUT_PATH} ({len(attack)} rows)")

    run_detected = attack.groupby(["counter", "mode", "seed"])["flagged"].any().reset_index(name="detected")
    summary = run_detected.groupby(["counter", "mode"])["detected"].agg(n_detected="sum", n_runs="count").reset_index()
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print(f"\nPer-run detection summary (centered D, W={WINDOW}):")
    print(summary.to_string(index=False))
    print(f"Saved {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
