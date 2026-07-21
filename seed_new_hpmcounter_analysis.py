"""Row-level hpmcounter3-hpmcounter10 comparison for the seed_new/ run.

seed_new/ monitors a different set of hardware events on hpmcounter3-10 than
the original seed_N_data/ runs, and (unlike those) is only a single run per
anomaly type so far - not enough independent runs for the run-level test in
run_level_hpmcounter_analysis.py. This mirrors the original single-run
hpmcounter_analysis.py instead: same Mann-Whitney U + Cliff's delta + BH-FDR
approach, applied directly to seed_new/'s raw ekf_*_hpc.csv files.

Caveat: rows within one run are an autocorrelated time series, not
independent samples, so p-values here are optimistic. Treat any signal as
exploratory - a candidate counter/event worth confirming across multiple
seed_new-style runs, not a confirmed result.
"""

import os

import numpy as np
import pandas as pd

from data_preprocessing import hex_to_int
from hpmcounter_analysis import ANOMALY_TYPES, COUNTERS, benjamini_hochberg, cliffs_delta
from scipy.stats import mannwhitneyu

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SEED_DIR = os.path.join(SRC_DIR, "seed_new")
OUT_PATH = os.path.join(SRC_DIR, "seed_new_hpmcounter_stats.csv")


def load_counters(name):
    path = os.path.join(SEED_DIR, f"ekf_{name}_hpc.csv")
    df = pd.read_csv(path)
    return df[COUNTERS].apply(lambda col: col.map(hex_to_int))


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
    print(
        "\nCaveat: this is a single run per type (row-level, autocorrelated "
        "within-run), same as the original one-run analysis - treat any "
        "'significant' counter here as exploratory, to confirm with more "
        "seed_new-style runs before trusting it."
    )


if __name__ == "__main__":
    main()
