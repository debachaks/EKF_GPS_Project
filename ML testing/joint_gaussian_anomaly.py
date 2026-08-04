"""Joint anomaly model over hpmcounter3 + hpmcounter5, learned on NORMAL runs
only, then scored against drift/jump/replay to see if modeling the two
counters jointly (their correlation/ratio) catches anything the per-counter
z/D/V thresholds in line_fitting/ miss.

Why these two counters, why joint, why Gaussian:
- hpmcounter3/5 are cumulative monotonic counters (not reset at board boot,
  see line_fitting/rerun_comparison.py), so we work on their RATE (first
  difference on a shared elapsed-time grid), not the raw cumulative value.
- The existing line_fitting pipeline (trend_score_windowed.py,
  variability_metric.py, combined_detection.py) already thresholds each
  counter independently (z_stat, D-slope, V-variability per counter). What
  none of those do is model the *relationship between* counter3 and
  counter5 -- e.g. their correlation drifting even if neither individual
  counter's own rate/slope crosses its own threshold. That's the joint
  Gaussian (Mahalanobis distance) model here.
- Mahalanobis distance under a fitted Gaussian is exactly the same math as
  the EKF's own NIS check in board_firmware/main_ekf.c (inn' * S^-1 * inn) --
  just applied to a window of HPC-counter-rate features instead of GPS
  innovations.

Validation: leave-one-seed-out (LOSO). With only 20 normal seeds, fitting
and testing the threshold on the same 20 would be optimistic. For each held
-out seed, the Gaussian is fit on the other 19 normal seeds' windows, the
threshold is the 95th percentile of THOSE 19 seeds' per-trial max Mahalanobis
distance (same convention as combined_detection.py's build_thresholds), and
only then is the held-out seed's own windows (normal + drift/jump/replay)
scored against it.

Outputs (ML testing/results/):
    joint_features_windowed.csv - one row per (mode, seed, window) with the
                                   5 features and the LOSO Mahalanobis score.
    joint_run_summary.csv       - one row per (mode, seed): max Mahalanobis
                                   score, the threshold applied, flagged.
    joint_detection_summary.csv - per-mode detection rate (out of 20 runs).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
while not os.path.isdir(os.path.join(PROJECT_ROOT, "original_pipeline")):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from data_preprocessing import hex_to_int  # noqa: E402

CLEAN_ROOT = os.path.join(PROJECT_ROOT, "seed_old", "CLEAN_HPC_TEST_SEED")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

MODES = ["normal", "drift", "jump", "replay"]
COUNTERS = ["hpmcounter3", "hpmcounter5"]
GRID_POINTS = 300
WINDOW = 10
STEP = 1
PERCENTILE = 95


def find_seed_names():
    dirs = sorted(d for d in glob.glob(os.path.join(CLEAN_ROOT, "test_seed_[0-9]*")) if os.path.isdir(d))
    return [os.path.basename(d) for d in dirs]


def raw_trace(seed_name, mode, counter):
    path = os.path.join(CLEAN_ROOT, seed_name, f"ekf_{mode}_hpc.csv")
    df = pd.read_csv(path)
    ts = df["timestamp_ms"].map(hex_to_int)
    elapsed = (ts - ts.iloc[0]).to_numpy()
    values = df[counter].map(hex_to_int).to_numpy()
    return elapsed, values


def build_global_grid(seed_names):
    max_elapsed = min(
        raw_trace(seed_name, mode, counter)[0].max()
        for seed_name in seed_names
        for mode in MODES
        for counter in COUNTERS
    )
    return np.linspace(0, max_elapsed, GRID_POINTS)


def rate_series(seed_name, mode, counter, grid):
    """Counter value interpolated onto the shared grid, then first-differenced
    -> rate of increase per grid step. Length GRID_POINTS - 1."""
    elapsed, values = raw_trace(seed_name, mode, counter)
    x_t = np.interp(grid, elapsed, values)
    return np.diff(x_t)


def onset_grid_index(seed_name, mode, grid):
    """Attack onset is at ATTACK_START=150 of ~300 total EKF steps (see
    ekf_config.h) -- i.e. the midpoint of the run, confirmed empirically in
    line_fitting/results/onset_grid_index_by_run.csv (onset_fraction_of_own_run
    = 0.5 for every seed/mode there). Recomputed here on THIS script's own
    grid (built from hpmcounter3/5 only, not all 8 counters) rather than
    reusing that file directly, since the two grids' max_elapsed can differ."""
    elapsed, _ = raw_trace(seed_name, mode, "hpmcounter3")
    onset_elapsed_ms = elapsed.max() * 0.5
    return int(np.searchsorted(grid, onset_elapsed_ms))


def windowed_features(rate_c3, rate_c5):
    """Per WINDOW-sample window (step STEP): mean rate and slope (D) for each
    counter separately, plus their within-window Pearson correlation -- the
    joint feature the per-counter pipeline doesn't have."""
    n = len(rate_c3)
    idx = np.arange(WINDOW)
    rows = []
    for start in range(0, n - WINDOW + 1, STEP):
        w3 = rate_c3[start:start + WINDOW]
        w5 = rate_c5[start:start + WINDOW]

        slope3 = np.polyfit(idx, w3, 1)[0]
        slope5 = np.polyfit(idx, w5, 1)[0]

        if w3.std() < 1e-9 or w5.std() < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(w3, w5)[0, 1]

        rows.append({
            "window_start": start,
            "mean_c3": w3.mean(), "mean_c5": w5.mean(),
            "slope_c3": slope3, "slope_c5": slope5,
            "corr_c3_c5": corr,
        })
    return pd.DataFrame(rows)


FEATURE_COLS = ["mean_c3", "mean_c5", "slope_c3", "slope_c5", "corr_c3_c5"]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    seed_names = find_seed_names()
    grid = build_global_grid(seed_names)
    print(f"{len(seed_names)} seeds, grid up to {grid[-1]:.0f} ms, {WINDOW}-sample windows")

    all_windows = []
    for seed_name in seed_names:
        for mode in MODES:
            rate_c3 = rate_series(seed_name, mode, "hpmcounter3", grid)
            rate_c5 = rate_series(seed_name, mode, "hpmcounter5", grid)
            feats = windowed_features(rate_c3, rate_c5)
            feats["seed"] = seed_name
            feats["mode"] = mode
            feats["onset_idx"] = onset_grid_index(seed_name, mode, grid)
            all_windows.append(feats)

    windows = pd.concat(all_windows, ignore_index=True)
    print(f"Total windows: {len(windows)} ({len(windows) // len(seed_names) // len(MODES)} per run)")

    # Only score windows at/after each run's own attack-onset point (the
    # run's own midpoint, see onset_grid_index docstring) -- same rule
    # applied uniformly to normal too, so the normal false-positive rate is
    # measured over the same relative time region as the attack detections,
    # not an unfair whole-run-vs-second-half comparison.
    post_onset = windows["window_start"] >= windows["onset_idx"]

    windows["mahalanobis"] = np.nan
    run_rows = []

    for held_out in seed_names:
        train_seeds = [s for s in seed_names if s != held_out]

        train_normal = windows[(windows["mode"] == "normal") & (windows["seed"].isin(train_seeds))]
        cov = EmpiricalCovariance().fit(train_normal[FEATURE_COLS].to_numpy())

        train_normal_post = train_normal[post_onset.loc[train_normal.index]]
        per_trial_max = (
            train_normal_post.assign(_d2=cov.mahalanobis(train_normal_post[FEATURE_COLS].to_numpy()))
            .groupby("seed")["_d2"].max()
        )
        threshold = np.percentile(per_trial_max, PERCENTILE)

        held_mask = windows["seed"] == held_out
        held_scores = cov.mahalanobis(windows.loc[held_mask, FEATURE_COLS].to_numpy())
        windows.loc[held_mask, "mahalanobis"] = held_scores

        held = windows.loc[held_mask & post_onset, ["mode", "seed", "mahalanobis"]]
        for mode, group in held.groupby("mode"):
            max_d2 = group["mahalanobis"].max()
            run_rows.append({
                "mode": mode, "seed": held_out,
                "max_mahalanobis": max_d2,
                "threshold": threshold,
                "flagged": bool(max_d2 > threshold),
            })

    run_summary = pd.DataFrame(run_rows).sort_values(["mode", "seed"])
    detection = (
        run_summary.groupby("mode")["flagged"]
        .agg(n_flagged="sum", n_runs="count")
        .reset_index()
    )
    detection["detection_rate"] = detection["n_flagged"] / detection["n_runs"]

    windows.to_csv(os.path.join(RESULTS_DIR, "joint_features_windowed.csv"), index=False)
    run_summary.to_csv(os.path.join(RESULTS_DIR, "joint_run_summary.csv"), index=False)
    detection.to_csv(os.path.join(RESULTS_DIR, "joint_detection_summary.csv"), index=False)

    print("\nPer-mode detection rate (LOSO Gaussian, hpmcounter3+5 joint):")
    print(detection.to_string(index=False))
    print(f"\nSaved windowed features, run summary, and detection summary -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
