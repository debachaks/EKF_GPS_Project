"""Tests three MORE one-class model architectures on the RAW, unmodified
D/G/V features (ml_fusion_classifier.py's original 15-feature set: 5
counters x 3 metrics, each metric's own independent max(|score|) --
NOT the agreement/scenario-derived features).

Motivation: ml_fusion_pca_classifier.py already showed that better
downstream modeling of the SAME raw features (Mahalanobis + PCA) barely
moves the pre-onset false-alarm problem (0.583->0.467), which pointed to
a feature-construction issue (destroyed timing information), not a
model-choice issue. This script checks that conclusion more thoroughly
before accepting it -- trying model families structurally different from
Mahalanobis (single global Gaussian) and LOF (local density ratio):

  1. Isolation Forest -- tree-based partitioning, no covariance/density
     estimate at all. (Already tried once, very early in this project,
     on a different feature set and only the standard evaluation, with
     default settings -- collapsed to F1=0.000. Retried here properly,
     on THIS feature set, both evaluations, with contamination tuned to
     the same 5%/10% convention used elsewhere.)
  2. One-Class SVM (RBF kernel) -- a nonlinear boundary around the dense
     region of the normal training cluster, unlike Mahalanobis's
     elliptical assumption.
  3. k-NN distance -- average distance to the k nearest normal training
     points; a simpler, non-local-density cousin of LOF (LOF normalizes
     by neighbors' own density, k-NN distance doesn't).

Same LOSO-CV protocol as ml_fusion_classifier.py / ml_fusion_pca_classifier.py:
fit fresh each fold on ONLY that fold's training normal-mode runs (19 of
them), never on attack data. Evaluated both ways:
  (a) standard: post-onset-only features, normal/jump/drift as 3 classes.
  (b) expanded/pre-onset stress test: normal trained on its full window
      range; jump/drift each contribute a post-onset (ground_truth=1)
      AND pre-onset (ground_truth=0) test row -- mirrors
      ml_fusion_expanded_normal.py exactly, for direct comparability
      against Mahalanobis/LOF's already-reported 0.583/0.500 raw-feature
      baseline and PCA's 0.467/0.433.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")

sys.path.insert(0, SCRIPT_DIR)
from ml_fusion_classifier import build_feature_table, confusion_stats  # noqa: E402
from ml_fusion_expanded_normal import build_expanded_feature_table  # noqa: E402

K_NEIGHBORS = 10


def _loso_loop(table, feature_cols, y, is_normal_fn, fit_and_score_fn, percentile):
    """Shared LOSO scaffold: fit_and_score_fn(X_train_normal, X_test) ->
    (train_scores, test_scores), where HIGHER score = more anomalous."""
    X = table[feature_cols].to_numpy()
    groups = table["seed"].to_numpy()
    is_normal = is_normal_fn(table)

    logo = LeaveOneGroupOut()
    y_pred = np.zeros_like(y)
    for train_idx, test_idx in logo.split(X, y, groups):
        train_normal_idx = train_idx[is_normal[train_idx]]

        scaler = StandardScaler().fit(X[train_normal_idx])
        X_train_normal = scaler.transform(X[train_normal_idx])
        X_test = scaler.transform(X[test_idx])

        train_scores, test_scores = fit_and_score_fn(X_train_normal, X_test)
        threshold = np.percentile(train_scores, percentile)
        y_pred[test_idx] = (test_scores > threshold).astype(int)

    return confusion_stats(y, y_pred)


def run_isoforest(table, feature_cols, y, is_normal_fn, contamination=0.05):
    def fit_and_score(X_train_normal, X_test):
        clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=0)
        clf.fit(X_train_normal)
        # score_samples: higher = more normal, so negate for "higher = more anomalous"
        return -clf.score_samples(X_train_normal), -clf.score_samples(X_test)

    return _loso_loop(table, feature_cols, y, is_normal_fn, fit_and_score, percentile=100 * (1 - contamination))


def run_ocsvm(table, feature_cols, y, is_normal_fn, nu=0.1, gamma="scale"):
    def fit_and_score(X_train_normal, X_test):
        clf = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
        clf.fit(X_train_normal)
        # decision_function: higher = more normal, so negate
        return -clf.decision_function(X_train_normal), -clf.decision_function(X_test)

    return _loso_loop(table, feature_cols, y, is_normal_fn, fit_and_score, percentile=100 * (1 - nu))


def run_knn_distance(table, feature_cols, y, is_normal_fn, k=K_NEIGHBORS, percentile=95):
    def fit_and_score(X_train_normal, X_test):
        kk = min(k, X_train_normal.shape[0] - 1)
        nn = NearestNeighbors(n_neighbors=kk).fit(X_train_normal)

        train_dist, _ = nn.kneighbors(X_train_normal, n_neighbors=kk + 1)
        train_scores = train_dist[:, 1:].mean(axis=1)  # exclude self (distance 0)

        test_dist, _ = nn.kneighbors(X_test, n_neighbors=kk)
        test_scores = test_dist.mean(axis=1)
        return train_scores, test_scores

    return _loso_loop(table, feature_cols, y, is_normal_fn, fit_and_score, percentile=percentile)


def main():
    table, feature_cols = build_feature_table()
    y_standard = (table["mode"] != "normal").astype(int).to_numpy()
    is_normal_standard = lambda t: (t["mode"] == "normal").to_numpy()  # noqa: E731

    print(f"Standard feature table: {table.shape[0]} runs x {len(feature_cols)} raw features")
    standard_results = {
        "isolation_forest (raw features, normal-only)": run_isoforest(table, feature_cols, y_standard, is_normal_standard),
        "one_class_svm (raw features, normal-only)": run_ocsvm(table, feature_cols, y_standard, is_normal_standard),
        f"knn_distance_k{K_NEIGHBORS} (raw features, normal-only)": run_knn_distance(table, feature_cols, y_standard, is_normal_standard),
    }
    standard_out = pd.DataFrame([{"method": name, **s} for name, s in standard_results.items()])
    standard_path = os.path.join(RESULTS_DIR, "ml_fusion_more_oneclass_classifier_loso.csv")
    standard_out.to_csv(standard_path, index=False)
    print("\n=== Standard LOSO-CV, raw features, additional one-class models (n=60) ===")
    print(standard_out.to_string(index=False))
    print(f"Saved {standard_path}")

    expanded_table, expanded_cols = build_expanded_feature_table()
    y_expanded = expanded_table["ground_truth"].to_numpy()
    is_normal_expanded = lambda t: t["train_eligible"].to_numpy()  # noqa: E731

    print(f"\nExpanded feature table: {expanded_table.shape[0]} rows x {len(expanded_cols)} raw features")
    expanded_results = {
        "isolation_forest (raw features, pre-onset-as-normal test)": run_isoforest(expanded_table, expanded_cols, y_expanded, is_normal_expanded),
        "one_class_svm (raw features, pre-onset-as-normal test)": run_ocsvm(expanded_table, expanded_cols, y_expanded, is_normal_expanded),
        f"knn_distance_k{K_NEIGHBORS} (raw features, pre-onset-as-normal test)": run_knn_distance(expanded_table, expanded_cols, y_expanded, is_normal_expanded),
    }
    expanded_out = pd.DataFrame([{"method": name, **s} for name, s in expanded_results.items()])
    expanded_path = os.path.join(RESULTS_DIR, "ml_fusion_more_oneclass_expanded_normal_loso.csv")
    expanded_out.to_csv(expanded_path, index=False)
    print(f"\n=== Expanded/pre-onset LOSO-CV, raw features, additional one-class models (n={expanded_table.shape[0]}) ===")
    print(expanded_out.to_string(index=False))
    print(f"Saved {expanded_path}")


if __name__ == "__main__":
    main()
