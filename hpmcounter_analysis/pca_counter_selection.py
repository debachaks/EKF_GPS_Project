"""PCA-based counter selection for the new candidate HPM event sets.

Mirrors the methodology from the JAMS paper (Section IV-B1): collect
benign-only traces, standardize each event's rate, run PCA, and rank
events by their contribution to the leading components (sum of squared
loadings weighted by each component's explained-variance share). Applied
per set (set_1/set_2/set_3), pooling rows across all 5 normal-only seeds.

Uses the RATE metric (counter delta / mcycle delta) rather than the raw
accumulated counter value. Raw hpmcounter values grow roughly linearly
over a run (see seed_old/test_seed_normal_pairwise_diff_distribution.py),
so PCA on raw values would mostly rediscover "how far into the run is
this row" as the dominant pattern for every counter, rather than genuine
differences in event-rate behavior between counters.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from data_preprocessing import hex_to_int  # noqa: E402

COUNTERS = [f"hpmcounter{i}" for i in range(3, 11)]

SET_LABELS = {
    "set_1": {
        "hpmcounter3": "FP interlock cycles (M1: FP latency - PRIMARY)",
        "hpmcounter4": "FP div/sqrt retired (M1: count control)",
        "hpmcounter5": "cond-branch mispredictions (M2: PD-check + alarm branches)",
        "hpmcounter6": "long-latency interlock (M1: alt FP-stall observable)",
        "hpmcounter7": "addr-generation interlock (M3: AGU timing - low hope)",
        "hpmcounter8": "int arithmetic retired (activity reference)",
        "hpmcounter9": "pipeline flushes (NEGATIVE control - must stay flat)",
        "hpmcounter10": "exceptions taken (should be 0; nonzero = FP traps)",
    },
    "set_2": {
        "hpmcounter3": "FP add retired (mix check)",
        "hpmcounter4": "FP multiply retired (mix check)",
        "hpmcounter5": "FP fused-multiply-add retired (mix check)",
        "hpmcounter6": "FP load retired (mix check)",
        "hpmcounter7": "FP store retired (mix check)",
        "hpmcounter8": "other FP retired (mix check)",
        "hpmcounter9": "int arithmetic retired (ANCHOR: activity reference)",
        "hpmcounter10": "pipeline flushes (ANCHOR: negative control)",
    },
    "set_3": {
        "hpmcounter3": "d-cache misses (M3: memory)",
        "hpmcounter4": "d-cache blocked cycles (M3: memory stalls)",
        "hpmcounter5": "d-cache writeback requests (M3: memory)",
        "hpmcounter6": "mul/div interlock (M4: expect ~0, pure FP)",
        "hpmcounter7": "long-latency interlock (ANCHOR: cross-check vs set1)",
        "hpmcounter8": "cond-branch mispredictions (ANCHOR: cross-check vs set1)",
        "hpmcounter9": "int arithmetic retired (ANCHOR: activity reference)",
        "hpmcounter10": "pipeline flushes (ANCHOR: negative control)",
    },
}

ANCHOR_GROUPS = {
    "int arithmetic retired": [("set_1", "hpmcounter8"), ("set_2", "hpmcounter9"), ("set_3", "hpmcounter9")],
    "pipeline flushes": [("set_1", "hpmcounter9"), ("set_2", "hpmcounter10"), ("set_3", "hpmcounter10")],
    "cond-branch mispredictions": [("set_1", "hpmcounter5"), ("set_3", "hpmcounter8")],
    "long-latency interlock": [("set_1", "hpmcounter6"), ("set_3", "hpmcounter7")],
}


def find_seed_dirs(set_name):
    return sorted(glob.glob(os.path.join(SRC_DIR, "seed_*", set_name)))


def load_rate_matrix(set_name):
    """Row-wise rate (counter delta / mcycle delta) for all 8 counters,
    pooled across every seed's normal run for this set."""
    rows = []
    for seed_dir in find_seed_dirs(set_name):
        path = os.path.join(seed_dir, "ekf_normal_hpc.csv")
        df = pd.read_csv(path)
        mcycle_delta = df["mcycle"].map(hex_to_int).diff()
        counter_df = df[COUNTERS].apply(lambda col: col.map(hex_to_int))
        rates = counter_df.diff().div(mcycle_delta, axis=0).iloc[1:]
        rates = rates.replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(rates)
    return pd.concat(rows, ignore_index=True)


def load_raw_matrix(set_name):
    """Raw counter value, pooled across every seed's normal run - kept for
    reference/comparison only (see module docstring for why rate is the
    primary metric)."""
    rows = []
    for seed_dir in find_seed_dirs(set_name):
        path = os.path.join(seed_dir, "ekf_normal_hpc.csv")
        df = pd.read_csv(path)
        rows.append(df[COUNTERS].apply(lambda col: col.map(hex_to_int)))
    return pd.concat(rows, ignore_index=True)


VARIANCE_CAPTURED_TARGET = 0.90


def pca_importance(matrix, counters):
    """Standardize, run PCA, and return a ranking table: importance =
    sum over the LEADING components (enough to capture
    VARIANCE_CAPTURED_TARGET of total variance) of
    (explained_variance_ratio * loading^2).

    Summing over ALL components instead of a leading subset is a
    mathematical dead end: for standardized features, sum_k(loading[k,j]^2
    * evr[k]) over the FULL set of components is provably 1/n_features for
    every feature j, regardless of the data (it's just the reconstructed
    correlation matrix's diagonal, normalized) - every counter would score
    identically no matter how it actually behaves. The JAMS paper avoids
    this by truncating to a limited number of leading components ("10 PCA
    components, which capture over 96% of the benign data variance"); this
    does the same, keeping only enough components to reach
    VARIANCE_CAPTURED_TARGET.
    """
    std = matrix.std()
    zero_variance = std[std < 1e-12].index.tolist()

    usable = [c for c in counters if c not in zero_variance]
    scaler = StandardScaler()
    z = scaler.fit_transform(matrix[usable])

    pca = PCA()
    pca.fit(z)

    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)
    n_components = int(np.searchsorted(cumulative, VARIANCE_CAPTURED_TARGET) + 1)
    n_components = min(n_components, len(evr))

    loadings = pca.components_[:n_components]
    evr_used = evr[:n_components]
    importance = (loadings**2 * evr_used[:, None]).sum(axis=0)

    result = pd.DataFrame({
        "counter": usable,
        "importance": importance,
        "std_of_rate": [matrix[c].std() for c in usable],
        "mean_of_rate": [matrix[c].mean() for c in usable],
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    for c in zero_variance:
        result.loc[len(result)] = [c, 0.0, 0.0, matrix[c].mean()]

    return result, evr, n_components


def print_set_report(set_name):
    print(f"\n{'='*70}\n{set_name}\n{'='*70}")
    labels = SET_LABELS[set_name]

    rate_matrix = load_rate_matrix(set_name)
    print(f"Pooled rows (5 seeds): {len(rate_matrix)}")

    ranking, evr, n_components = pca_importance(rate_matrix, COUNTERS)
    print(f"PCA explained variance ratio per component: {np.round(evr, 3).tolist()}")
    print(f"Using top {n_components} component(s) to reach {VARIANCE_CAPTURED_TARGET:.0%} variance captured")
    print()
    print(f"{'rank':<5}{'counter':<14}{'importance':<12}{'std_of_rate':<14}{'mean_of_rate':<14}event")
    for i, row in ranking.iterrows():
        print(f"{i+1:<5}{row['counter']:<14}{row['importance']:<12.4f}"
              f"{row['std_of_rate']:<14.6g}{row['mean_of_rate']:<14.6g}{labels[row['counter']]}")

    out_path = os.path.join(SRC_DIR, f"{set_name}_pca_ranking.csv")
    ranking["event"] = ranking["counter"].map(labels)
    ranking.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def print_anchor_consistency():
    print(f"\n{'='*70}\nAnchor counter cross-check (same event, measured in multiple sets)\n{'='*70}")
    for event_name, locations in ANCHOR_GROUPS.items():
        print(f"\n{event_name}:")
        for set_name, counter in locations:
            rate_matrix = load_rate_matrix(set_name)
            s = rate_matrix[counter]
            print(f"  {set_name}/{counter}: mean={s.mean():.6g} std={s.std():.6g} "
                  f"(relative std={s.std()/abs(s.mean()) if s.mean() else float('nan'):.3f})")


def main():
    for set_name in ["set_1", "set_2", "set_3"]:
        print_set_report(set_name)
    print_anchor_consistency()


if __name__ == "__main__":
    main()
