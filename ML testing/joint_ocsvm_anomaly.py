"""Same joint hpmcounter3+hpmcounter5 windowed-rate features as
joint_gaussian_anomaly.py, but scored with a One-Class SVM instead of a
fitted Gaussian/Mahalanobis distance -- to check whether the flat result
there (0% detection on drift/jump/replay) was a limitation of assuming a
Gaussian relationship between the two counters, or whether there's just no
usable joint signal in these two counters at all regardless of model.

Same LOSO (leave-one-seed-out) protocol as joint_gaussian_anomaly.py, so the
two are directly comparable:
  - StandardScaler + OneClassSVM(kernel="rbf") fit on the OTHER 19 normal
    seeds' windows (scaling matters here since raw feature scales differ a
    lot: mean_c3/mean_c5 are O(1-10), slope_* are small, corr is in [-1,1],
    and RBF distance is scale-sensitive).
  - anomaly_score = -decision_function(x) (higher = more anomalous, same
    sign convention as Mahalanobis distance).
  - threshold = 95th percentile of the 19 training seeds' own per-trial max
    anomaly_score (not the OCSVM's built-in nu boundary at 0), so both
    models are calibrated the same way and the comparison is fair.
  - held-out seed's windows (normal + drift/jump/replay) scored against
    that fold's fitted scaler+model; a run is flagged if any window's score
    exceeds the threshold.

Outputs (ML testing/results/):
    joint_ocsvm_features_windowed.csv
    joint_ocsvm_run_summary.csv
    joint_ocsvm_detection_summary.csv
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

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
NU = 0.05
GAMMA = "scale"


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
    # measured over the same relative time region as the attack detections.
    post_onset = windows["window_start"] >= windows["onset_idx"]

    windows["anomaly_score"] = np.nan
    run_rows = []

    for held_out in seed_names:
        train_seeds = [s for s in seed_names if s != held_out]

        train_normal = windows[(windows["mode"] == "normal") & (windows["seed"].isin(train_seeds))]
        X_train = train_normal[FEATURE_COLS].to_numpy()

        scaler = StandardScaler().fit(X_train)
        model = OneClassSVM(kernel="rbf", nu=NU, gamma=GAMMA).fit(scaler.transform(X_train))

        train_normal_post = train_normal[post_onset.loc[train_normal.index]]
        train_post_scores = -model.decision_function(scaler.transform(train_normal_post[FEATURE_COLS].to_numpy()))
        per_trial_max = (
            train_normal_post.assign(_s=train_post_scores).groupby("seed")["_s"].max()
        )
        threshold = np.percentile(per_trial_max, PERCENTILE)

        held_mask = windows["seed"] == held_out
        X_held = scaler.transform(windows.loc[held_mask, FEATURE_COLS].to_numpy())
        held_scores = -model.decision_function(X_held)
        windows.loc[held_mask, "anomaly_score"] = held_scores

        held = windows.loc[held_mask & post_onset, ["mode", "seed", "anomaly_score"]]
        for mode, group in held.groupby("mode"):
            max_score = group["anomaly_score"].max()
            run_rows.append({
                "mode": mode, "seed": held_out,
                "max_anomaly_score": max_score,
                "threshold": threshold,
                "flagged": bool(max_score > threshold),
            })

    run_summary = pd.DataFrame(run_rows).sort_values(["mode", "seed"])
    detection = (
        run_summary.groupby("mode")["flagged"]
        .agg(n_flagged="sum", n_runs="count")
        .reset_index()
    )
    detection["detection_rate"] = detection["n_flagged"] / detection["n_runs"]

    windows.to_csv(os.path.join(RESULTS_DIR, "joint_ocsvm_features_windowed.csv"), index=False)
    run_summary.to_csv(os.path.join(RESULTS_DIR, "joint_ocsvm_run_summary.csv"), index=False)
    detection.to_csv(os.path.join(RESULTS_DIR, "joint_ocsvm_detection_summary.csv"), index=False)

    print("\nPer-mode detection rate (LOSO One-Class SVM, hpmcounter3+5 joint):")
    print(detection.to_string(index=False))
    print(f"\nSaved windowed features, run summary, and detection summary -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
