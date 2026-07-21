"""Compare hpmcounter3-hpmcounter10 between each anomaly type and normal.

For each counter and each anomaly type (drift/jump/replay) runs a Mann-Whitney U
test against the normal run, plus Cliff's delta as an effect size. P-values are
corrected for multiple comparisons with Benjamini-Hochberg (FDR).
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from data_preprocessing import CLEAN_DIR, hex_to_int

COUNTERS = [f"hpmcounter{i}" for i in range(3, 11)]
ANOMALY_TYPES = ["drift", "jump", "replay"]
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hpmcounter_stats.csv")


def load_counters(name):
    path = os.path.join(CLEAN_DIR, f"ekf_{name}_hpc.csv")
    df = pd.read_csv(path)
    return df[COUNTERS].apply(lambda col: col.map(hex_to_int))


def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    diff = x[:, None] - y[None, :]
    return (np.sign(diff).sum()) / (len(x) * len(y))


def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted


def main():
    normal = load_counters("normal")

    rows = []
    for anomaly in ANOMALY_TYPES:
        anomaly_df = load_counters(anomaly)
        for counter in COUNTERS:
            x = anomaly_df[counter].to_numpy()
            y = normal[counter].to_numpy()
            u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
            delta = cliffs_delta(x, y)
            rows.append({
                "counter": counter,
                "anomaly_type": anomaly,
                "median_anomaly": np.median(x),
                "median_normal": np.median(y),
                "U": u_stat,
                "p_value": p_value,
                "cliffs_delta": delta,
            })

    results = pd.DataFrame(rows)
    results["p_value_fdr"] = benjamini_hochberg(results["p_value"].to_numpy())
    results["significant_fdr_0.05"] = results["p_value_fdr"] < 0.05
    results = results.sort_values("p_value_fdr").reset_index(drop=True)

    results.to_csv(OUT_PATH, index=False)
    print(results.to_string(index=False))
    print(f"\nSaved full results to {OUT_PATH}")


if __name__ == "__main__":
    main()
