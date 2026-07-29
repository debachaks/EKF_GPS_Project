"""Multivariate (joint-counter) anomaly detection, Mahalanobis-distance
style - same approach as the JAMS paper (Section IV-B3): fit a mean vector
and covariance matrix on normal-only data, then score every run (held-out
normal and every attack type) by how far it sits from that normal cluster,
accounting for how the counters co-vary rather than checking each one in
isolation.

Feature vector per run: per-run median rate (counter delta / mcycle delta)
for each of the 8 surviving hpmcounters - the same per-run summary used
throughout test_seed_run_level_rate_analysis.py, just combined into one
8-dimensional vector per run instead of tested one counter at a time.

Normal-only scores are computed via leave-one-out (fit on the other 19
normal runs, score the held-out one) to avoid evaluating on the same data
used to fit the reference distribution. The final reference (for scoring
attack runs) is fit on all 20 normal runs. Threshold = mean + 3*std of the
held-out normal scores, matching the paper's three-sigma rule.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from data_preprocessing import hex_to_int  # noqa: E402

CLEAN_ROOT = os.path.join(SCRIPT_DIR, "CLEAN_HPC_TEST_SEED")
ANOMALY_TYPES = ["drift", "jump", "replay"]
ALL_TYPES = ["normal"] + ANOMALY_TYPES
REG_EPSILON = 1e-6


def find_seed_dirs():
    return sorted(d for d in glob.glob(os.path.join(CLEAN_ROOT, "test_seed_[0-9]*")) if os.path.isdir(d))


def find_counters(seed_dirs):
    common = None
    for seed_dir in seed_dirs:
        for run_type in ALL_TYPES:
            path = os.path.join(seed_dir, f"ekf_{run_type}_hpc.csv")
            cols = frozenset(c for c in pd.read_csv(path, nrows=1).columns if c.startswith("hpmcounter"))
            common = cols if common is None else (common & cols)
    return sorted(common, key=lambda c: int(c.replace("hpmcounter", "")))


def per_run_rate_median(seed_dir, run_type, counters):
    path = os.path.join(seed_dir, f"ekf_{run_type}_hpc.csv")
    df = pd.read_csv(path)
    mcycle_delta = df["mcycle"].map(hex_to_int).diff()
    counter_df = df[counters].apply(lambda col: col.map(hex_to_int))
    rates = counter_df.diff().div(mcycle_delta, axis=0).iloc[1:].replace([np.inf, -np.inf], np.nan)
    return rates.median()


def build_feature_table(seed_dirs, counters):
    rows = []
    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        for run_type in ALL_TYPES:
            medians = per_run_rate_median(seed_dir, run_type, counters)
            row = {"seed": seed_name, "run_type": run_type}
            row.update(medians.to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def standardize(x, mean, std):
    return (x - mean) / std


def mahalanobis(x, mean, cov_inv):
    diff = x - mean
    return np.sqrt(diff @ cov_inv @ diff)


def fit_reference(feature_matrix):
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std[std < 1e-12] = 1.0
    z = standardize(feature_matrix, mean, std)
    cov = np.cov(z, rowvar=False) + np.eye(z.shape[1]) * REG_EPSILON
    cov_inv = np.linalg.inv(cov)
    return mean, std, cov_inv


def score_matrix(feature_matrix, mean, std, cov_inv):
    z = standardize(feature_matrix, mean, std)
    return np.array([mahalanobis(row, np.zeros_like(mean), cov_inv) for row in z])


def leave_one_out_normal_scores(normal_matrix):
    n = normal_matrix.shape[0]
    scores = np.zeros(n)
    for i in range(n):
        train = np.delete(normal_matrix, i, axis=0)
        mean, std, cov_inv = fit_reference(train)
        held_out = normal_matrix[i]
        z = standardize(held_out, mean, std)
        scores[i] = mahalanobis(z, np.zeros_like(mean), cov_inv)
    return scores


def main():
    seed_dirs = find_seed_dirs()
    if not seed_dirs:
        print(f"No cleaned test_seed_N folders found under {CLEAN_ROOT} - run test_seed_clean_hpc.py first")
        return
    print(f"Found {len(seed_dirs)} seeds")

    counters = find_counters(seed_dirs)
    print(f"Feature vector: {counters} ({len(counters)}-dimensional)\n")

    table = build_feature_table(seed_dirs, counters)
    normal_matrix = table[table.run_type == "normal"][counters].to_numpy()

    print("Computing leave-one-out normal scores (train on other 19, score the held-out one)...")
    normal_scores = leave_one_out_normal_scores(normal_matrix)
    threshold = normal_scores.mean() + 3 * normal_scores.std()
    print(f"Held-out normal scores: mean={normal_scores.mean():.3f} std={normal_scores.std():.3f}")
    print(f"Threshold (mean + 3*std): {threshold:.3f}")
    n_normal_fp = (normal_scores > threshold).sum()
    print(f"Held-out normal runs exceeding threshold (false positives): {n_normal_fp}/{len(normal_scores)}\n")

    print("Fitting final reference on all 20 normal runs, scoring each attack type...")
    mean, std, cov_inv = fit_reference(normal_matrix)

    results = []
    for run_type in ANOMALY_TYPES:
        attack_matrix = table[table.run_type == run_type][counters].to_numpy()
        scores = score_matrix(attack_matrix, mean, std, cov_inv)
        n_flagged = (scores > threshold).sum()
        for seed, score in zip(table[table.run_type == run_type]["seed"], scores):
            results.append({"run_type": run_type, "seed": seed, "score": score, "flagged": score > threshold})
        print(f"{run_type}: {n_flagged}/{len(scores)} runs flagged "
              f"(scores: mean={scores.mean():.3f} min={scores.min():.3f} max={scores.max():.3f})")

    out_path = os.path.join(SCRIPT_DIR, "test_seed_multivariate_mahalanobis_scores.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
