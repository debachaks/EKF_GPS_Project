"""One-class detection on the ORIGINAL 15 raw D/G/V features (5 counters
x 3 metrics, each metric's own max(|score|) -- ml_fusion_classifier.py's
feature set, unmodified), with PCA dimensionality reduction added before
Mahalanobis/LOF instead of ml_fusion_agreement_classifier.py's hand-built
"how many metrics crossed their own threshold" agreement count.

Why this exists: the agreement-count features fixed the pre-onset
false-alarm problem (FA 0.583->0.067 for Mahalanobis on the expanded/
pre-onset stress test -- see ml_fusion_agreement_classifier.py), but
they do so by pre-computing each metric's own rule-based flag (score >
its Q95 threshold) and handing the model a count of how many fired --
that is hand-designed ensemble voting wearing an ML costume, not the
model learning the joint structure of the raw scores itself.

This script instead asks: can a one-class model work directly with the
RAW continuous scores (no thresholding, no hand-built agreement signal)
and still suppress the pre-onset false alarms, purely by better
exploiting the CORRELATION structure between the 15 raw features? The
raw-feature Mahalanobis/LOF result (FA=0.583/0.500 on the pre-onset
stress test) already had access to all the joint information needed in
principle -- Mahalanobis distance is defined by the full covariance
matrix, which encodes exactly "do these features tend to move together"
-- but with only 19 training points in 15 dimensions (n approx p), the
covariance estimate is too noisy to reliably capture that structure,
even with Ledoit-Wolf shrinkage (shrinkage regularizes the estimate
without fixing the fundamental sample-size problem).

PCA fit fresh each LOSO fold, on ONLY that fold's training normal data
(scaled first), keeping N_COMPONENTS directions -- a purely data-driven,
unsupervised dimensionality reduction (no attack labels, no per-metric
thresholds), same "no hand-built rule" standard as the original raw
features, just projected onto the directions of highest normal-mode
variance before distance/density scoring.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")

sys.path.insert(0, SCRIPT_DIR)
from ml_fusion_classifier import build_feature_table, confusion_stats  # noqa: E402
from ml_fusion_expanded_normal import build_expanded_feature_table  # noqa: E402

N_COMPONENTS = 5  # same dimensionality as the agreement-feature version, for a fair comparison


def run_one_class_pca_loso(table, feature_cols, mode_col, is_normal_fn, n_components=N_COMPONENTS, percentile=95):
    X = table[feature_cols].to_numpy()
    y = mode_col
    groups = table["seed"].to_numpy()
    is_normal = is_normal_fn(table)

    logo = LeaveOneGroupOut()
    y_pred = np.zeros_like(y)
    for train_idx, test_idx in logo.split(X, y, groups):
        train_normal_idx = train_idx[is_normal[train_idx]]

        scaler = StandardScaler().fit(X[train_normal_idx])
        X_train_normal = scaler.transform(X[train_normal_idx])
        X_test = scaler.transform(X[test_idx])

        k = min(n_components, X_train_normal.shape[0] - 1, X_train_normal.shape[1])
        pca = PCA(n_components=k, random_state=0).fit(X_train_normal)
        Z_train = pca.transform(X_train_normal)
        Z_test = pca.transform(X_test)

        cov = LedoitWolf().fit(Z_train)
        threshold = np.percentile(cov.mahalanobis(Z_train), percentile)
        y_pred[test_idx] = (cov.mahalanobis(Z_test) > threshold).astype(int)

    return confusion_stats(y, y_pred)


def run_lof_pca_loso(table, feature_cols, mode_col, is_normal_fn, n_components=N_COMPONENTS,
                      n_neighbors=10, percentile=5):
    X = table[feature_cols].to_numpy()
    y = mode_col
    groups = table["seed"].to_numpy()
    is_normal = is_normal_fn(table)

    logo = LeaveOneGroupOut()
    y_pred = np.zeros_like(y)
    for train_idx, test_idx in logo.split(X, y, groups):
        train_normal_idx = train_idx[is_normal[train_idx]]

        scaler = StandardScaler().fit(X[train_normal_idx])
        X_train_normal = scaler.transform(X[train_normal_idx])
        X_test = scaler.transform(X[test_idx])

        k_pca = min(n_components, X_train_normal.shape[0] - 1, X_train_normal.shape[1])
        pca = PCA(n_components=k_pca, random_state=0).fit(X_train_normal)
        Z_train = pca.transform(X_train_normal)
        Z_test = pca.transform(X_test)

        k_nn = min(n_neighbors, len(train_normal_idx) - 1)
        lof = LocalOutlierFactor(n_neighbors=k_nn, novelty=True)
        lof.fit(Z_train)

        threshold = np.percentile(lof.score_samples(Z_train), percentile)
        y_pred[test_idx] = (lof.score_samples(Z_test) < threshold).astype(int)

    return confusion_stats(y, y_pred)


def main():
    # --- standard evaluation (raw 15 features, PCA-reduced) ---
    table, feature_cols = build_feature_table()
    y_standard = (table["mode"] != "normal").astype(int).to_numpy()
    is_normal_standard = lambda t: (t["mode"] == "normal").to_numpy()  # noqa: E731

    print(f"Standard feature table: {table.shape[0]} runs x {len(feature_cols)} raw features -> PCA({N_COMPONENTS})")

    standard_results = {
        f"mahalanobis+PCA{N_COMPONENTS} (raw features, normal-only)":
            run_one_class_pca_loso(table, feature_cols, y_standard, is_normal_standard),
        f"lof+PCA{N_COMPONENTS} (raw features, normal-only)":
            run_lof_pca_loso(table, feature_cols, y_standard, is_normal_standard),
    }
    standard_out = pd.DataFrame([{"method": name, **s} for name, s in standard_results.items()])
    standard_path = os.path.join(RESULTS_DIR, "ml_fusion_pca_classifier_loso.csv")
    standard_out.to_csv(standard_path, index=False)
    print("\n=== Standard LOSO-CV, raw features + PCA (n=60) ===")
    print(standard_out.to_string(index=False))
    print(f"Saved {standard_path}")

    # --- expanded / pre-onset stress test (raw 15 features, PCA-reduced) ---
    expanded_table, expanded_cols = build_expanded_feature_table()
    y_expanded = expanded_table["ground_truth"].to_numpy()
    is_normal_expanded = lambda t: t["train_eligible"].to_numpy()  # noqa: E731

    print(f"\nExpanded feature table: {expanded_table.shape[0]} rows x {len(expanded_cols)} raw features -> PCA({N_COMPONENTS})")

    expanded_results = {
        f"mahalanobis+PCA{N_COMPONENTS} (raw features, pre-onset-as-normal test)":
            run_one_class_pca_loso(expanded_table, expanded_cols, y_expanded, is_normal_expanded),
        f"lof+PCA{N_COMPONENTS} (raw features, pre-onset-as-normal test)":
            run_lof_pca_loso(expanded_table, expanded_cols, y_expanded, is_normal_expanded),
    }
    expanded_out = pd.DataFrame([{"method": name, **s} for name, s in expanded_results.items()])
    expanded_path = os.path.join(RESULTS_DIR, "ml_fusion_pca_expanded_normal_loso.csv")
    expanded_out.to_csv(expanded_path, index=False)
    print(f"\n=== Expanded/pre-onset LOSO-CV, raw features + PCA (n={expanded_table.shape[0]}) ===")
    print(expanded_out.to_string(index=False))
    print(f"Saved {expanded_path}")


if __name__ == "__main__":
    main()
