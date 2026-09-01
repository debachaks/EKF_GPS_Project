"""Variant of ml_fusion_classifier.py's one-class evaluation that makes
fuller use of the available data, in two ways:

1. TRAINING: the one-class models are fit on normal-mode runs' FULL
   window range (no post-onset restriction) instead of just their
   t>=150 windows. Normal runs never have an attack at all, so there is
   no oracle-timing concern in using their whole run -- restricting them
   to t>=150 was previously just a side effect of applying one shared
   filter to every row, not something normal-mode data specifically
   needed. Using the full 289-window range gives a richer basis for
   estimating the "normal" cluster shape, which matters given the
   already-small n=19-per-fold training set.

2. TESTING: each jump/drift run now contributes TWO test cases instead
   of one -- its post-onset segment (ground_truth=1, the actual attack
   window) AND its pre-onset segment (ground_truth=0, genuinely normal
   behavior, since nothing has happened yet at that point in the run).
   Previously the pre-onset ~140 windows of all 40 attack runs
   contributed nothing to training OR testing; here they become 40
   additional "should be normal" test cases, on top of the 20 genuine
   normal runs -- 60 total ground_truth=0 test cases instead of 20,
   which also directly tightens the false-alarm-rate confidence
   interval (see the earlier Wilson-CI discussion: FA computed from n=20
   is quite wide; n=60 is much tighter).

Only the genuine normal-mode runs are ever used to FIT the one-class
model -- a jump/drift run's pre-onset segment is scored as a test case
with ground_truth=0, but never used to build the "normal" reference
shape itself, keeping training data uncontaminated by attack-run
provenance even when that run's early segment looks benign.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")

sys.path.insert(0, SCRIPT_DIR)
from ml_fusion_classifier import ONSET_ITER, USABLE_COUNTERS, confusion_stats  # noqa: E402

METRICS = [
    ("D", "d", "d_final_dscore.csv"),
    ("G", "g", "g_final_w5_gscore.csv"),
    ("V", "v", "v_final_w5_vscore.csv"),
]
ATTACK_MODES = ["jump", "drift"]


def build_expanded_feature_table():
    per_metric = {}
    for short, col, path in METRICS:
        df = pd.read_csv(os.path.join(RESULTS_DIR, path))
        df = df[df["counter"].isin(USABLE_COUNTERS) & (~df["sigma_fragile"])]
        per_metric[short] = df

    seeds = sorted(per_metric["D"]["seed"].unique(), key=lambda s: int(s.replace("seed", "")))

    def feature_row(mode, seed, iter_mask_fn):
        row = {}
        for short, col, _ in METRICS:
            df = per_metric[short]
            for counter in USABLE_COUNTERS:
                sub = df[(df["counter"] == counter) & (df["mode"] == mode) & (df["seed"] == seed)]
                sub = sub[iter_mask_fn(sub["window_end_iter"])]
                row[f"{short}_{counter}"] = sub[col].abs().max() if len(sub) else float("nan")
        return row

    rows = []
    for seed in seeds:
        rows.append({
            "row_type": "normal_full", "seed": seed, "ground_truth": 0, "train_eligible": True,
            **feature_row("normal", seed, lambda it: it >= 0),  # full range, no restriction
        })
        for mode in ATTACK_MODES:
            rows.append({
                "row_type": f"{mode}_post", "seed": seed, "ground_truth": 1, "train_eligible": False,
                **feature_row(mode, seed, lambda it: it >= ONSET_ITER),
            })
            rows.append({
                "row_type": f"{mode}_pre", "seed": seed, "ground_truth": 0, "train_eligible": False,
                **feature_row(mode, seed, lambda it: it < ONSET_ITER),
            })

    table = pd.DataFrame(rows)
    feature_cols = [c for c in table.columns if c not in ("row_type", "seed", "ground_truth", "train_eligible")]
    table[feature_cols] = table[feature_cols].fillna(table[feature_cols].median())
    return table, feature_cols


def run_one_class_loso(table, feature_cols, percentile=95):
    X = table[feature_cols].to_numpy()
    y = table["ground_truth"].to_numpy()
    groups = table["seed"].to_numpy()
    train_eligible = table["train_eligible"].to_numpy()

    logo = LeaveOneGroupOut()
    y_pred = np.zeros_like(y)
    for train_idx, test_idx in logo.split(X, y, groups):
        fit_idx = train_idx[train_eligible[train_idx]]

        scaler = StandardScaler().fit(X[fit_idx])
        X_fit = scaler.transform(X[fit_idx])
        X_test = scaler.transform(X[test_idx])

        cov = LedoitWolf().fit(X_fit)
        threshold = np.percentile(cov.mahalanobis(X_fit), percentile)
        y_pred[test_idx] = (cov.mahalanobis(X_test) > threshold).astype(int)

    return confusion_stats(y, y_pred)


def run_lof_loso(table, feature_cols, n_neighbors=10, percentile=5):
    X = table[feature_cols].to_numpy()
    y = table["ground_truth"].to_numpy()
    groups = table["seed"].to_numpy()
    train_eligible = table["train_eligible"].to_numpy()

    logo = LeaveOneGroupOut()
    y_pred = np.zeros_like(y)
    for train_idx, test_idx in logo.split(X, y, groups):
        fit_idx = train_idx[train_eligible[train_idx]]

        scaler = StandardScaler().fit(X[fit_idx])
        X_fit = scaler.transform(X[fit_idx])
        X_test = scaler.transform(X[test_idx])

        k = min(n_neighbors, len(fit_idx) - 1)
        lof = LocalOutlierFactor(n_neighbors=k, novelty=True)
        lof.fit(X_fit)

        threshold = np.percentile(lof.score_samples(X_fit), percentile)
        y_pred[test_idx] = (lof.score_samples(X_test) < threshold).astype(int)

    return confusion_stats(y, y_pred)


def main():
    table, feature_cols = build_expanded_feature_table()
    print(f"Feature table: {table.shape[0]} rows x {len(feature_cols)} features")
    print(f"  row_type counts: {table['row_type'].value_counts().to_dict()}")
    print(f"  ground_truth counts: {table['ground_truth'].value_counts().to_dict()}")

    results = {
        "mahalanobis (full-window normal train, pre-onset-as-normal test)": run_one_class_loso(table, feature_cols),
        "lof (full-window normal train, pre-onset-as-normal test)": run_lof_loso(table, feature_cols),
    }

    rows = [{"method": name, **stats_} for name, stats_ in results.items()]
    out = pd.DataFrame(rows)

    out_path = os.path.join(RESULTS_DIR, "ml_fusion_expanded_normal_loso.csv")
    out.to_csv(out_path, index=False)

    print(f"\n=== LOSO-CV results, expanded normal/pre-onset evaluation (n={table.shape[0]}) ===")
    print(out.to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
