"""ML fusion classifier for the ICC submission: replaces the per-counter
Q95-threshold + hand-designed OR/majority/AND voting (combined/ensemble_
detection.py) with a single model that jointly learns from all 5 usable
counters at once.

Two approaches, both evaluated with leave-one-seed-out (LOSO) cross-
validation grouped by seed (so a seed's normal/jump/drift runs never
split across train/test within a fold -- avoids leaking the shared
pre-onset noise-stream quirks documented in pre_onset_audit.py into an
inflated CV score):

  1. SUPERVISED (logistic regression, random forest): trained on labeled
     normal/jump/drift runs. Strongest expected numbers, but the claim
     must be scoped carefully -- LOSO-CV here tests generalization
     across SEEDS of the two known attack shapes (jump, drift), not
     generalization to unseen attack types.

  2. ONE-CLASS (Mahalanobis distance over the 15-d feature space,
     fit on normal runs only): does not require labeled attacks at
     train time, closer to a realistic deployment assumption, and a
     multivariate generalization of the existing per-counter Q95
     threshold. Weaker expected numbers, but a fairer claim: "fusing
     counters helps even without attack labels."

Features: for each run (mode, seed), for each of the 5 usable counters
(hpmcounter9 excluded -- established near-negative-control) x 3 metrics
(D_final W=10, G_final W=5, V_final W=5), take max(|score|) over
post-onset windows (window_end_iter >= ONSET_ITER), excluding
sigma_fragile positions -- mirrors the per-trial-max used to calibrate
each metric's own Q95 threshold (d_final_metric.py etc.), just fed to a
learned boundary instead of a fixed percentile cutoff.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")

ONSET_ITER = 150
USABLE_COUNTERS = ["hpmcounter3", "hpmcounter4", "hpmcounter5", "hpmcounter8", "hpmcounter10"]
MODES = ["normal", "jump", "drift"]

METRICS = [
    ("D", "d", "d_final_dscore.csv"),
    ("G", "g", "g_final_w5_gscore.csv"),
    ("V", "v", "v_final_w5_vscore.csv"),
]


def build_feature_table():
    """One row per (mode, seed): 15 features (5 counters x 3 metrics),
    each = max(|score|) over non-fragile post-onset windows."""
    per_metric_max = {}
    for short, col, path in METRICS:
        df = pd.read_csv(os.path.join(RESULTS_DIR, path))
        df = df[df["counter"].isin(USABLE_COUNTERS) & (df["window_end_iter"] >= ONSET_ITER) & (~df["sigma_fragile"])]
        agg = df.groupby(["counter", "mode", "seed"])[col].apply(lambda s: s.abs().max())
        per_metric_max[short] = agg

    seeds = sorted(per_metric_max["D"].index.get_level_values("seed").unique(),
                    key=lambda s: int(s.replace("seed", "")))

    rows = []
    for mode in MODES:
        for seed in seeds:
            row = {"mode": mode, "seed": seed}
            for short, _, _ in METRICS:
                for counter in USABLE_COUNTERS:
                    key = (counter, mode, seed)
                    row[f"{short}_{counter}"] = per_metric_max[short].get(key, np.nan)
            rows.append(row)

    table = pd.DataFrame(rows)
    feature_cols = [c for c in table.columns if c not in ("mode", "seed")]
    table[feature_cols] = table[feature_cols].fillna(table[feature_cols].median())
    return table, feature_cols


def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def confusion_stats(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    fa = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fa_ci = wilson_ci(fp, fp + tn)
    rec_ci = wilson_ci(tp, tp + fn)
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "recall_ci_lo": round(rec_ci[0], 3), "recall_ci_hi": round(rec_ci[1], 3),
        "f1": round(f1, 3),
        "false_alarm": round(fa, 3),
        "fa_ci_lo": round(fa_ci[0], 3), "fa_ci_hi": round(fa_ci[1], 3),
    }


def run_supervised_loso(table, feature_cols, model_name):
    X = table[feature_cols].to_numpy()
    y = (table["mode"] != "normal").astype(int).to_numpy()  # attack=1, normal=0
    groups = table["seed"].to_numpy()

    logo = LeaveOneGroupOut()
    y_pred = np.zeros_like(y)
    for train_idx, test_idx in logo.split(X, y, groups):
        scaler = StandardScaler().fit(X[train_idx])
        X_train, X_test = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])

        if model_name == "logreg":
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        elif model_name == "rf":
            clf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0, class_weight="balanced")
        else:
            raise ValueError(model_name)

        clf.fit(X_train, y[train_idx])
        y_pred[test_idx] = clf.predict(X_test)

    return confusion_stats(y, y_pred)


def run_one_class_loso(table, feature_cols, percentile=95):
    """Mahalanobis distance from the normal-run centroid, fit fresh each
    fold on ONLY that fold's training normal runs -- no attack labels
    used to build the detector, only to evaluate it afterward. Threshold
    calibrated per-fold from the training normal runs' own max distance,
    mirroring the Q95-over-normal-trials convention used everywhere else
    in this project (d_final_metric.py etc.). Covariance uses Ledoit-Wolf
    shrinkage rather than the raw empirical covariance: with only 19
    training points in a 15-d feature space (n approx p), the empirical
    covariance is near-singular and produces wildly unstable distances
    for any held-out point -- shrinkage regularizes exactly this regime."""
    X = table[feature_cols].to_numpy()
    y = (table["mode"] != "normal").astype(int).to_numpy()
    groups = table["seed"].to_numpy()
    is_normal = (table["mode"] == "normal").to_numpy()

    logo = LeaveOneGroupOut()
    y_pred = np.zeros_like(y)
    for train_idx, test_idx in logo.split(X, y, groups):
        train_normal_idx = train_idx[is_normal[train_idx]]

        scaler = StandardScaler().fit(X[train_normal_idx])
        X_train_normal = scaler.transform(X[train_normal_idx])
        X_test = scaler.transform(X[test_idx])

        cov = LedoitWolf().fit(X_train_normal)
        train_dist = cov.mahalanobis(X_train_normal)
        threshold = np.percentile(train_dist, percentile)

        test_dist = cov.mahalanobis(X_test)
        y_pred[test_idx] = (test_dist > threshold).astype(int)

    return confusion_stats(y, y_pred)


def main():
    table, feature_cols = build_feature_table()
    print(f"Feature table: {table.shape[0]} runs x {len(feature_cols)} features")
    print(f"  mode counts: {table['mode'].value_counts().to_dict()}")

    results = {
        "logreg (supervised)": run_supervised_loso(table, feature_cols, "logreg"),
        "random_forest (supervised)": run_supervised_loso(table, feature_cols, "rf"),
        "mahalanobis (one-class, normal-only)": run_one_class_loso(table, feature_cols),
    }

    rows = [{"method": name, **stats_} for name, stats_ in results.items()]
    out = pd.DataFrame(rows)

    out_path = os.path.join(RESULTS_DIR, "ml_fusion_classifier_loso.csv")
    out.to_csv(out_path, index=False)

    print("\n=== LOSO-CV results, all 5 usable counters fused (attack=1, normal=0; n=60) ===")
    print(out.to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
