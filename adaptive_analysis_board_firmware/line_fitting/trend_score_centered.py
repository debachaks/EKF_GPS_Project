"""Centered D metric: instead of D = |beta / SE(beta)| (implicitly
assuming normal-mode beta is ~0 at every window position), this computes

    D_centered = |beta - mean_beta_normal(t)| / SE(beta)

where mean_beta_normal(t) is the average beta from the 20 normal-mode
trials AT THE SAME window position t (same per-iteration-baseline idea
used throughout this pipeline for the raw counter's z-score and for V),
rather than assuming the "expected" normal slope is zero everywhere.

This is a NEW, standalone script -- trend_score_windowed.py (the original
D = |beta/SE(beta)|) is untouched.

Threshold calibration: per-trial-max-then-percentile (same as the
original trend_score_windowed.py) -- h_D = 95th percentile of {max over
window positions of D_centered, per normal run}.

This was investigated and deliberately kept, not just left over:
  - A flat-pooled threshold (z_metric.py's convention: 95th percentile
    of every normal run's D_centered value, pooled across all 20 trials
    and window positions) was tried first, since the old z metric's
    max-based threshold turned out to compound two extreme-value
    operations into an effectively ~99th-percentile cutoff. But D is
    far noisier per-window than z's rolling mean (se_beta can be tiny
    on near-flat windows, spiking D), so flat-pooling + "any window
    over threshold" fires on ~20/20 normal runs on most counters --
    useless for discrimination.
  - Requiring >=K windows over the flat-pooled threshold (K up to 10,
    out of ~150 post-attack windows) still left normal runs firing at
    13-16/20 on most counters -- doesn't fix it.
  - Lowering the per-trial-max method's percentile (0.70-0.85 instead
    of 0.95) isn't a structural fix either: it's a pure slider that
    raises attack detection and normal false positives together (e.g.
    at 0.85, normal FP rises to 2-4/20 per counter), with hpmcounter9
    staying completely dead at 0/20 for both attack modes regardless.
  - The original 0.95 per-trial-max threshold was the only one of these
    that kept normal-run false positives near 0, so it's kept as-is.

WINDOW=10, matching the original trend_score_windowed.py.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")

D_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_centered_windowed.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_centered_threshold.csv")
FLAGGED_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_centered_attack_flags.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_centered_detection_summary.csv")

WINDOW = 10
STEP = 1
PERCENTILE = 95
EPS = 1e-20

_S = np.arange(WINDOW, dtype=float)
_S_BAR = _S.mean()
_S_DEV = _S - _S_BAR
_SS_S = np.sum(_S_DEV**2)


def windowed_beta_se(z_values):
    """Returns (window_start, beta, se_beta) for every full, NaN-free
    window -- the raw ingredients for D, before combining/centering."""
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

    # mean_beta_normal(counter, window_end_iter): average beta across the
    # 20 normal trials, at each window position
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
    print("\nh_D (centered D, W=10) (95th percentile of normal runs' max D), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack = df[df["mode"].isin(["jump", "drift"])].merge(thresholds, on="counter")
    attack["flagged"] = attack["D"] > attack["h_D"]
    attack.to_csv(FLAGGED_OUT_PATH, index=False)
    print(f"\nSaved {FLAGGED_OUT_PATH} ({len(attack)} rows)")

    run_detected = attack.groupby(["counter", "mode", "seed"])["flagged"].any().reset_index(name="detected")
    summary = run_detected.groupby(["counter", "mode"])["detected"].agg(n_detected="sum", n_runs="count").reset_index()
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print("\nPer-run detection summary (centered D) (a run counts as detected if ANY window exceeds h_D):")
    print(summary.to_string(index=False))
    print(f"Saved {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
