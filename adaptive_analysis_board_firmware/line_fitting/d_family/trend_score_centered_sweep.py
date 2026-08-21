"""Window-size sweep for the CENTERED D metric (trend_score_centered.py's
D = |beta - mean_beta_normal(t)| / SE(beta)), trying every odd window
size from 5 to 29 in steps of 2 (5, 7, 9, ..., 29), to find which window
size performs best for drift detection.

For each window size W:
    1. Compute (beta, se_beta) per window position, per run.
    2. mean_beta_normal(counter, window_end_iter) from the 20 normal
       trials, same as trend_score_centered.py.
    3. D = |beta - mean_beta_normal| / se_beta.
    4. h_D = 95th percentile of {per-run max D, across normal trials}
       (same calibration convention as trend_score_centered.py).
    5. Detection count for drift/jump on the 5 strong counters
       (hpmcounter3/4/5/8/10).

Prints one row per window size so the sweep can be read off directly.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")
SWEEP_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_centered_sweep.csv")

STRONG = ["hpmcounter3", "hpmcounter4", "hpmcounter5", "hpmcounter8", "hpmcounter10"]
WINDOWS = list(range(5, 30, 2))  # 5, 7, 9, ..., 29
STEP = 1
PERCENTILE = 95
EPS = 1e-20


def windowed_beta_se(z_values, window):
    n = len(z_values)
    s = np.arange(window, dtype=float)
    s_bar = s.mean()
    s_dev = s - s_bar
    ss_s = np.sum(s_dev**2)

    results = []
    for start in range(0, n - window + 1, STEP):
        w = z_values[start:start + window]
        if np.any(np.isnan(w)):
            continue
        beta = np.sum(s_dev * w) / ss_s
        alpha = w.mean() - beta * s_bar
        fitted = alpha + beta * s
        resid = w - fitted
        sigma_eps2 = np.sum(resid**2) / (window - 2)
        se_beta = np.sqrt(sigma_eps2 / ss_s + EPS)
        results.append((start, beta, se_beta))
    return results


def run_one_window(ts, window):
    rows = []
    for (counter, mode, seed), group in ts[ts["counter"].isin(STRONG)].groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z_values = group["z"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, beta, se_beta in windowed_beta_se(z_values, window):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + window - 1],
                "beta": beta, "se_beta": se_beta,
            })
    df = pd.DataFrame(rows)

    normal = df[df["mode"] == "normal"]
    mean_beta_normal = (
        normal.groupby(["counter", "window_end_iter"])["beta"]
        .mean()
        .reset_index(name="mean_beta_normal")
    )
    df = df.merge(mean_beta_normal, on=["counter", "window_end_iter"], how="left")
    df["D"] = (df["beta"] - df["mean_beta_normal"]).abs() / df["se_beta"]

    normal_d = df[df["mode"] == "normal"]
    per_run_max = normal_d.groupby(["counter", "seed"])["D"].max().reset_index(name="M_j")
    thresholds = (
        per_run_max.groupby("counter")["M_j"]
        .quantile(PERCENTILE / 100)
        .reset_index(name="h_D")
    )

    attack = df[df["mode"].isin(["jump", "drift"])].merge(thresholds, on="counter")
    attack["flagged"] = attack["D"] > attack["h_D"]

    run_detected = attack.groupby(["counter", "mode", "seed"])["flagged"].any().reset_index(name="detected")
    summary = run_detected.groupby(["counter", "mode"])["detected"].sum().reset_index()
    return summary


def main():
    ts = pd.read_csv(TIMESERIES_PATH)

    sweep_rows = []
    print(f"{'W':>3s}  " + "  ".join(f"{c.replace('hpmcounter','c')}_drift" for c in STRONG) + "   " +
          "  ".join(f"{c.replace('hpmcounter','c')}_jump" for c in STRONG))
    for w in WINDOWS:
        summary = run_one_window(ts, w)
        drift_counts = {c: 0 for c in STRONG}
        jump_counts = {c: 0 for c in STRONG}
        for _, row in summary.iterrows():
            if row["mode"] == "drift":
                drift_counts[row["counter"]] = int(row["detected"])
            else:
                jump_counts[row["counter"]] = int(row["detected"])

        for c in STRONG:
            sweep_rows.append({
                "window": w, "counter": c,
                "drift_detected": drift_counts[c], "jump_detected": jump_counts[c],
            })

        drift_str = "  ".join(f"{drift_counts[c]:8d}" for c in STRONG)
        jump_str = "  ".join(f"{jump_counts[c]:7d}" for c in STRONG)
        drift_avg = sum(drift_counts.values()) / len(STRONG)
        print(f"{w:3d}  {drift_str}   {jump_str}   (drift avg={drift_avg:.1f})")

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(SWEEP_OUT_PATH, index=False)
    print(f"\nSaved {SWEEP_OUT_PATH}")

    best = sweep_df.groupby("window")["drift_detected"].mean().sort_values(ascending=False)
    print("\nBest window sizes by average drift detection across the 5 strong counters:")
    print(best.head(5).to_string())


if __name__ == "__main__":
    main()
