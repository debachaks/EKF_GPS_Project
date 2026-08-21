"""Decomposes the spread of "normal" behavior into two sources, using the
20 normal-mode runs:

    within-seed noise  - how much a single run's own rate
                          (counter-delta / mcycle-delta, row to row) wobbles
                          around ITS OWN mean - a proxy for run-level/
                          microarchitectural jitter, since it's measured
                          entirely within one execution.
    between-seed noise - how much the 20 seeds' own mean rates differ from
                          EACH OTHER - reflects genuine differences in each
                          seed's injected GPS sensor-noise realization
                          (different noise -> different EKF workload ->
                          different average rate), not measurement jitter.

If between-seed >> within-seed, most of what the mean+-std normal band
captures is legitimate seed-to-seed diversity (not reducible without
changing what "normal" means), so tightening the band would just make the
threshold blind to real normal variation. If within-seed is comparable or
larger, a meaningful chunk of the spread is run-level jitter, which could
plausibly be reduced (longer averaging windows, tighter experimental
control, etc.).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
while not os.path.isdir(os.path.join(PROJECT_ROOT, "original_pipeline")):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from data_preprocessing import hex_to_int  # noqa: E402

CLEAN_ROOT = os.path.join(PROJECT_ROOT, "seed_old", "CLEAN_HPC_TEST_SEED")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def find_seed_names():
    dirs = sorted(d for d in glob.glob(os.path.join(CLEAN_ROOT, "test_seed_[0-9]*")) if os.path.isdir(d))
    return [os.path.basename(d) for d in dirs]


def find_counters(seed_names):
    common = None
    for seed_name in seed_names:
        path = os.path.join(CLEAN_ROOT, seed_name, "ekf_normal_hpc.csv")
        cols = frozenset(c for c in pd.read_csv(path, nrows=1).columns if c.startswith("hpmcounter"))
        common = cols if common is None else (common & cols)
    return sorted(common, key=lambda c: int(c.replace("hpmcounter", "")))


def row_rate_series(seed_name, counter):
    path = os.path.join(CLEAN_ROOT, seed_name, "ekf_normal_hpc.csv")
    df = pd.read_csv(path)
    mcycle_delta = df["mcycle"].map(hex_to_int).diff()
    counter_delta = df[counter].map(hex_to_int).diff()
    rate = (counter_delta / mcycle_delta).iloc[1:].replace([np.inf, -np.inf], np.nan).dropna()
    return rate.to_numpy()


def main():
    seed_names = find_seed_names()
    counters = find_counters(seed_names)

    rows = []
    for counter in counters:
        within_stds = []
        seed_means = []
        for seed_name in seed_names:
            rate = row_rate_series(seed_name, counter)
            within_stds.append(rate.std(ddof=1))
            seed_means.append(rate.mean())

        within_seed_std = np.mean(within_stds)
        between_seed_std = np.std(seed_means, ddof=1)
        ratio = between_seed_std / within_seed_std if within_seed_std > 0 else np.nan

        rows.append({
            "counter": counter,
            "within_seed_std (avg)": within_seed_std,
            "between_seed_std": between_seed_std,
            "between/within ratio": ratio,
            "dominant_source": "between-seed (real GPS-noise diversity)" if ratio > 1 else "within-seed (run jitter)",
        })

    result = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(result.to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "variance_decomposition_results.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
