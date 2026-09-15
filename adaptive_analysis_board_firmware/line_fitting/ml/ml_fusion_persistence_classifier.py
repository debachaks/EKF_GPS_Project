"""Option 1 from the ml_fusion_expanded_normal.py post-mortem: replace
each counter/metric's feature -- max(|score|), a single static number
that loses all temporal structure -- with a PERSISTENCE feature:
the longest run of CONSECUTIVE windows where |score| exceeds that
metric's own threshold H. Same idea as the rule-based fix in
combined/ensemble_persistence_false_alarm.py (require >=3 consecutive
fires, not just one), but built into the ML feature itself rather than
applied as a rule after the fact -- max()/mean() already erase the
time axis before the one-class model ever sees the data, so a
persistence rule can't be bolted on afterward; it has to be computed
before that collapse happens.

A single noisy window gives a run-length of 1. A real, sustained
attack signature gives a much longer run. The hypothesis: this should
suppress the jump_pre/drift_pre over-flagging found in
ml_fusion_expanded_normal.py (85%/70%, vs. 20% for genuine normal
trials) the same way the 3-consecutive-window rule suppressed it for
the rule-based ensemble (75%->15%, 55%->15%), while preserving
detection recall.

Builds the SAME expanded test design as ml_fusion_expanded_normal.py
(normal_full trained on full window range; jump/drift split into
_post [ground_truth=1] and _pre [ground_truth=0] test cases) so the
before/after comparison is apples-to-apples.
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
    ("D", "d", "H_d", "d_final_dscore.csv", "d_final_thresholds.csv"),
    ("G", "g", "H_g", "g_final_w5_gscore.csv", "g_final_w5_thresholds.csv"),
    ("V", "v", "H_v", "v_final_w5_vscore.csv", "v_final_w5_thresholds.csv"),
]
ATTACK_MODES = ["jump", "drift"]


def longest_run(bool_arr):
    longest = current = 0
    for v in bool_arr:
        current = current + 1 if v else 0
        longest = max(longest, current)
    return longest


def build_persistence_feature_table():
    per_metric = {}
    for short, col, thresh_col, score_path, thresh_path in METRICS:
        df = pd.read_csv(os.path.join(RESULTS_DIR, score_path))
        thresh = pd.read_csv(os.path.join(RESULTS_DIR, thresh_path)).set_index("counter")[thresh_col]
        df = df[df["counter"].isin(USABLE_COUNTERS) & (~df["sigma_fragile"])].copy()
        df["h"] = df["counter"].map(thresh)
        df["flagged"] = df[col].abs() > df["h"]
        per_metric[short] = df

    seeds = sorted(per_metric["D"]["seed"].unique(), key=lambda s: int(s.replace("seed", "")))

    def feature_row(mode, seed, mask_fn):
        row = {}
        for short, col, thresh_col, _, _ in METRICS:
            df = per_metric[short]
            for counter in USABLE_COUNTERS:
                sub = df[(df["counter"] == counter) & (df["mode"] == mode) & (df["seed"] == seed)]
                sub = sub[mask_fn(sub["window_end_iter"])].sort_values("window_end_iter")
                row[f"{short}_{counter}"] = longest_run(sub["flagged"].to_numpy()) if len(sub) else 0
        return row

    rows = []
    for seed in seeds:
        rows.append({
            "row_type": "normal_full", "seed": seed, "ground_truth": 0, "train_eligible": True,
            **feature_row("normal", seed, lambda it: it >= 0),
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
        X_fit, X_test = scaler.transform(X[fit_idx]), scaler.transform(X[test_idx])

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
        X_fit, X_test = scaler.transform(X[fit_idx]), scaler.transform(X[test_idx])

        k = min(n_neighbors, len(fit_idx) - 1)
        lof = LocalOutlierFactor(n_neighbors=k, novelty=True)
        lof.fit(X_fit)
        threshold = np.percentile(lof.score_samples(X_fit), percentile)
        y_pred[test_idx] = (lof.score_samples(X_test) < threshold).astype(int)

    return confusion_stats(y, y_pred)


def main():
    table, feature_cols = build_persistence_feature_table()
    print(f"Feature table: {table.shape[0]} rows x {len(feature_cols)} features (persistence = longest consecutive run)")
    print(f"  row_type counts: {table['row_type'].value_counts().to_dict()}")

    results = {
        "mahalanobis (persistence feature, expanded test)": run_one_class_loso(table, feature_cols),
        "lof (persistence feature, expanded test)": run_lof_loso(table, feature_cols),
    }

    for name, stats_ in results.items():
        print(f"\n--- {name} ---")
        print(stats_)

    out = pd.DataFrame([{"method": k, **v} for k, v in results.items()])
    out_path = os.path.join(RESULTS_DIR, "ml_fusion_persistence_expanded_loso.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    # breakdown: where do the false positives land?
    for method, fn in [("mahalanobis", run_one_class_loso), ("lof", run_lof_loso)]:
        X = table[feature_cols].to_numpy()
        y = table["ground_truth"].to_numpy()
        groups = table["seed"].to_numpy()
        train_eligible = table["train_eligible"].to_numpy()
        logo = LeaveOneGroupOut()
        y_pred = np.zeros_like(y)
        for train_idx, test_idx in logo.split(X, y, groups):
            fit_idx = train_idx[train_eligible[train_idx]]
            scaler = StandardScaler().fit(X[fit_idx])
            X_fit, X_test = scaler.transform(X[fit_idx]), scaler.transform(X[test_idx])
            if method == "mahalanobis":
                cov = LedoitWolf().fit(X_fit)
                th = np.percentile(cov.mahalanobis(X_fit), 95)
                y_pred[test_idx] = (cov.mahalanobis(X_test) > th).astype(int)
            else:
                k = min(10, len(fit_idx) - 1)
                lof = LocalOutlierFactor(n_neighbors=k, novelty=True)
                lof.fit(X_fit)
                th = np.percentile(lof.score_samples(X_fit), 5)
                y_pred[test_idx] = (lof.score_samples(X_test) < th).astype(int)

        table_copy = table.copy()
        table_copy["pred"] = y_pred
        breakdown = table_copy[table_copy.ground_truth == 0].groupby("row_type")["pred"].agg(["sum", "count"])
        print(f"\n=== {method}: false positives by row_type (ground_truth=0 rows) ===")
        print(breakdown)


if __name__ == "__main__":
    main()
