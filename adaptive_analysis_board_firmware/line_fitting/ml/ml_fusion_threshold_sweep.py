"""Threshold sensitivity sweep for the final ML fusion config (persistence
feature, hpmcounter3 excluded, stress test -- normal_full + jump/drift_post
+ jump/drift_pre). Both one-class methods pick their anomaly threshold as a
percentile of the TRAINING (fit) distribution:

  - Mahalanobis: flag test point if its distance exceeds the P-th percentile
    of training distances (we use P=95 as the operating point).
  - LOF: flag test point if its score falls below the P-th percentile of
    training scores (we use P=5).

This sweeps P across a range for each method (LOSO-CV at every point, same
protocol as ml_fusion_persistence_classifier.py) and plots FA and Recall
against P, marking the chosen operating point -- shows the chosen threshold
sits in a stable plateau rather than at a cliff edge, which is the question
a reviewer would ask about an unmotivated 95/5 choice.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
PLOTS_DIR = os.path.join(LINE_FITTING_DIR, "plots_heatmap")

sys.path.insert(0, SCRIPT_DIR)
from ml_fusion_classifier import confusion_stats  # noqa: E402
from ml_fusion_persistence_classifier import build_persistence_feature_table  # noqa: E402

MAHALANOBIS_PERCENTILES = [80, 85, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 99.5]
LOF_PERCENTILES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20]
CHOSEN_MAHALANOBIS = 95
CHOSEN_LOF = 5


def drop_hpmcounter3(feature_cols):
    return [c for c in feature_cols if not c.endswith("_hpmcounter3")]


def sweep_mahalanobis(table, feature_cols):
    X = table[feature_cols].to_numpy()
    y = table["ground_truth"].to_numpy()
    groups = table["seed"].to_numpy()
    train_eligible = table["train_eligible"].to_numpy()
    logo = LeaveOneGroupOut()

    rows = []
    for p in MAHALANOBIS_PERCENTILES:
        y_pred = np.zeros_like(y)
        for train_idx, test_idx in logo.split(X, y, groups):
            fit_idx = train_idx[train_eligible[train_idx]]
            scaler = StandardScaler().fit(X[fit_idx])
            X_fit, X_test = scaler.transform(X[fit_idx]), scaler.transform(X[test_idx])
            cov = LedoitWolf().fit(X_fit)
            threshold = np.percentile(cov.mahalanobis(X_fit), p)
            y_pred[test_idx] = (cov.mahalanobis(X_test) > threshold).astype(int)
        stats_ = confusion_stats(y, y_pred)
        rows.append({"percentile": p, **stats_})
    return pd.DataFrame(rows)


def sweep_lof(table, feature_cols):
    X = table[feature_cols].to_numpy()
    y = table["ground_truth"].to_numpy()
    groups = table["seed"].to_numpy()
    train_eligible = table["train_eligible"].to_numpy()
    logo = LeaveOneGroupOut()

    rows = []
    for p in LOF_PERCENTILES:
        y_pred = np.zeros_like(y)
        for train_idx, test_idx in logo.split(X, y, groups):
            fit_idx = train_idx[train_eligible[train_idx]]
            scaler = StandardScaler().fit(X[fit_idx])
            X_fit, X_test = scaler.transform(X[fit_idx]), scaler.transform(X[test_idx])
            k = min(10, len(fit_idx) - 1)
            lof = LocalOutlierFactor(n_neighbors=k, novelty=True)
            lof.fit(X_fit)
            threshold = np.percentile(lof.score_samples(X_fit), p)
            y_pred[test_idx] = (lof.score_samples(X_test) < threshold).astype(int)
        stats_ = confusion_stats(y, y_pred)
        rows.append({"percentile": p, **stats_})
    return pd.DataFrame(rows)


def make_plot(maha_df, lof_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, df, title, chosen, xlabel in [
        (axes[0], maha_df, "Mahalanobis", CHOSEN_MAHALANOBIS, "Threshold percentile P\n(flag if distance > P-th pct of training)"),
        (axes[1], lof_df, "LOF", CHOSEN_LOF, "Threshold percentile P\n(flag if score < P-th pct of training)"),
    ]:
        ax.plot(df["percentile"], df["false_alarm"], marker="o", color="tab:red", label="False alarm")
        ax.plot(df["percentile"], df["recall"], marker="s", color="tab:blue", label="Recall")
        ax.axvline(chosen, color="gray", linestyle="--", linewidth=1, label=f"Chosen P={chosen}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Rate")
    axes[0].legend(loc="center right", fontsize=8)
    fig.suptitle("Threshold sensitivity (persistence feature, hpmcounter3 excluded, stress test)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    table, feature_cols = build_persistence_feature_table()
    feature_cols = drop_hpmcounter3(feature_cols)
    print(f"Feature table: {table.shape[0]} rows x {len(feature_cols)} features (hpmcounter3 excluded)")

    maha_df = sweep_mahalanobis(table, feature_cols)
    lof_df = sweep_lof(table, feature_cols)

    print("\n--- Mahalanobis sweep ---")
    print(maha_df[["percentile", "recall", "false_alarm", "f1"]].to_string(index=False))
    print("\n--- LOF sweep ---")
    print(lof_df[["percentile", "recall", "false_alarm", "f1"]].to_string(index=False))

    maha_df.to_csv(os.path.join(RESULTS_DIR, "ml_fusion_threshold_sweep_mahalanobis.csv"), index=False)
    lof_df.to_csv(os.path.join(RESULTS_DIR, "ml_fusion_threshold_sweep_lof.csv"), index=False)

    make_plot(maha_df, lof_df, os.path.join(PLOTS_DIR, "ml_fusion_threshold_sweep.png"))


if __name__ == "__main__":
    main()
