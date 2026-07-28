"""Mean-based variant of test_seed_run_level_rate_analysis.py.

The median-based whole-run rate test is deliberately blind to a brief,
sharp spike within a run - it only reports the "typical" value and ignores
outliers unless they make up more than half the run's samples. If an
attack's real effect is a short burst rather than a sustained shift, mean
(which does get pulled toward outliers) might catch something median
missed.

Same run-level design as test_seed_run_level_rate_analysis.py (each seed =
one independent run, compared via Mann-Whitney U + Cliff's delta, across
all 8 surviving hpmcounters and all 20 seeds) - the only change is that the
per-run summary statistic is the mean of the row-wise rate (counter delta /
mcycle delta) instead of the median.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from data_preprocessing import hex_to_int  # noqa: E402
from hpmcounter_analysis import benjamini_hochberg, cliffs_delta  # noqa: E402

CLEAN_ROOT = os.path.join(SCRIPT_DIR, "CLEAN_HPC_TEST_SEED")
SUMMARY_OUT = os.path.join(SCRIPT_DIR, "test_seed_run_level_rate_summary_mean.csv")
STATS_OUT = os.path.join(SCRIPT_DIR, "test_seed_run_level_rate_hpmcounter_stats_mean.csv")

ANOMALY_TYPES = ["drift", "jump", "replay"]
ALL_TYPES = ["normal"] + ANOMALY_TYPES


def find_seed_dirs():
    return sorted(
        d for d in glob.glob(os.path.join(CLEAN_ROOT, "test_seed_[0-9]*")) if os.path.isdir(d)
    )


def find_counters(seed_dirs):
    common = None
    for seed_dir in seed_dirs:
        for run_type in ALL_TYPES:
            path = os.path.join(seed_dir, f"ekf_{run_type}_hpc.csv")
            cols = frozenset(c for c in pd.read_csv(path, nrows=1).columns if c.startswith("hpmcounter"))
            common = cols if common is None else (common & cols)
    return sorted(common, key=lambda c: int(c.replace("hpmcounter", "")))


def per_run_rate_means(seed_dir, run_type, counters):
    path = os.path.join(seed_dir, f"ekf_{run_type}_hpc.csv")
    df = pd.read_csv(path)

    mcycle_delta = df["mcycle"].map(hex_to_int).diff()
    counter_df = df[counters].apply(lambda col: col.map(hex_to_int))
    counter_deltas = counter_df.diff()

    rates = counter_deltas.div(mcycle_delta, axis=0)
    rates = rates.iloc[1:].replace([np.inf, -np.inf], np.nan)
    return rates.mean()


def build_run_summary(seed_dirs, counters):
    rows = []
    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        for run_type in ALL_TYPES:
            means = per_run_rate_means(seed_dir, run_type, counters)
            row = {"seed": seed_name, "run_type": run_type}
            row.update(means.to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def run_tests(summary, counters):
    normal = summary[summary["run_type"] == "normal"]

    rows = []
    for anomaly in ANOMALY_TYPES:
        anomaly_df = summary[summary["run_type"] == anomaly]
        for counter in counters:
            x = anomaly_df[counter].to_numpy()
            y = normal[counter].to_numpy()
            u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
            delta = cliffs_delta(x, y)
            rows.append({
                "counter": counter,
                "anomaly_type": anomaly,
                "n_anomaly_runs": len(x),
                "n_normal_runs": len(y),
                "mean_of_run_mean_rates_anomaly": np.mean(x),
                "mean_of_run_mean_rates_normal": np.mean(y),
                "U": u_stat,
                "p_value": p_value,
                "cliffs_delta": delta,
            })

    results = pd.DataFrame(rows)
    results["p_value_fdr"] = benjamini_hochberg(results["p_value"].to_numpy())
    results["significant_fdr_0.05"] = results["p_value_fdr"] < 0.05
    return results.sort_values("p_value_fdr").reset_index(drop=True)


def main():
    seed_dirs = find_seed_dirs()
    if not seed_dirs:
        print(f"No cleaned test_seed_N folders found under {CLEAN_ROOT} - run test_seed_clean_hpc.py first")
        return
    print(f"Found {len(seed_dirs)} seeds: {[os.path.basename(d) for d in seed_dirs]}")

    counters = find_counters(seed_dirs)
    print(f"hpmcounters present after cleaning: {counters}")

    summary = build_run_summary(seed_dirs, counters)
    summary.to_csv(SUMMARY_OUT, index=False)
    print(f"Saved per-run mean rates to {SUMMARY_OUT}")

    results = run_tests(summary, counters)
    results.to_csv(STATS_OUT, index=False)
    print(results.to_string(index=False))
    print(f"\nSaved run-level mean-rate test results to {STATS_OUT}")

    n_seeds = len(seed_dirs)
    if n_seeds < 15:
        print(
            f"\nCaveat: only {n_seeds} run(s) per type. Treat any 'significant' "
            "result above as a pilot-stage hypothesis, not a firm conclusion "
            "(recommended 20-30 runs/type for stable power)."
        )


if __name__ == "__main__":
    main()
