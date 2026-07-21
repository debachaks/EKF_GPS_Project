"""Run-level comparison of hpmcounter3-hpmcounter10 for the new-HPM-mapping
runs (seed_new_1..5), using the cleaned files from CLEAN_HPC_NEW_HPM/.

Same methodology as run_level_hpmcounter_analysis.py: each run is reduced to
a per-run median per counter first (rows within a run are an autocorrelated
time series, not independent samples), then the N=5 normal-run medians are
compared against the N=5 anomaly-run medians per counter. Kept fully
separate from the original seed_1_data..seed_5_data pipeline since these
hpmcounter columns count a different set of hardware events here.
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from data_preprocessing import hex_to_int  # noqa: E402
from hpmcounter_analysis import benjamini_hochberg, cliffs_delta  # noqa: E402

CLEAN_ROOT = os.path.join(SCRIPT_DIR, "CLEAN_HPC_NEW_HPM")
SUMMARY_OUT = os.path.join(SCRIPT_DIR, "new_hpm_run_level_summary.csv")
STATS_OUT = os.path.join(SCRIPT_DIR, "new_hpm_run_level_hpmcounter_stats.csv")
PLOT_DIR = os.path.join(SCRIPT_DIR, "new_hpm_run_level_plots")

ANOMALY_TYPES = ["drift", "jump", "replay"]
ALL_TYPES = ["normal"] + ANOMALY_TYPES


def find_seed_dirs():
    return sorted(d for d in glob.glob(os.path.join(CLEAN_ROOT, "seed_new_[0-9]*")) if os.path.isdir(d))


def find_counters(seed_dirs):
    """hpmcounter columns present (post-cleaning) in every seed/mode file."""
    common = None
    for seed_dir in seed_dirs:
        for run_type in ALL_TYPES:
            path = os.path.join(seed_dir, f"ekf_{run_type}_hpc.csv")
            cols = frozenset(c for c in pd.read_csv(path, nrows=1).columns if c.startswith("hpmcounter"))
            common = cols if common is None else (common & cols)
    return sorted(common, key=lambda c: int(c.replace("hpmcounter", "")))


def per_run_medians(seed_dir, run_type, counters):
    path = os.path.join(seed_dir, f"ekf_{run_type}_hpc.csv")
    df = pd.read_csv(path)
    counter_df = df[counters].apply(lambda col: col.map(hex_to_int))
    return counter_df.median()


def build_run_summary(seed_dirs, counters):
    rows = []
    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        for run_type in ALL_TYPES:
            medians = per_run_medians(seed_dir, run_type, counters)
            row = {"seed": seed_name, "run_type": run_type}
            row.update(medians.to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def plot_run_level_dots(summary, counters):
    os.makedirs(PLOT_DIR, exist_ok=True)
    normal = summary[summary["run_type"] == "normal"]

    for counter in counters:
        plt.figure(figsize=(6, 4))
        groups = ALL_TYPES
        for i, run_type in enumerate(groups):
            data = normal if run_type == "normal" else summary[summary["run_type"] == run_type]
            ys = data[counter].to_numpy()
            xs = np.full(len(ys), i) + np.random.uniform(-0.05, 0.05, len(ys))
            plt.scatter(xs, ys, label=run_type)
        plt.xticks(range(len(groups)), groups)
        plt.ylabel(f"{counter} (per-run median)")
        plt.title(f"{counter}: per-run medians by run type (seed_new_1..5)")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(PLOT_DIR, f"{counter}_run_level.png")
        plt.savefig(out_path)
        plt.close()
    print(f"Saved per-counter dot plots to {PLOT_DIR}")


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
                "median_of_run_medians_anomaly": np.median(x),
                "median_of_run_medians_normal": np.median(y),
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
        print(f"No cleaned seed_new_N folders found under {CLEAN_ROOT} - run new_hpm_clean_hpc.py first")
        return
    print(f"Found {len(seed_dirs)} seeds: {[os.path.basename(d) for d in seed_dirs]}")

    counters = find_counters(seed_dirs)
    print(f"hpmcounters present after cleaning: {counters}")

    summary = build_run_summary(seed_dirs, counters)
    summary.to_csv(SUMMARY_OUT, index=False)
    print(f"Saved per-run medians to {SUMMARY_OUT}")

    results = run_tests(summary, counters)
    results.to_csv(STATS_OUT, index=False)
    print(results.to_string(index=False))
    print(f"\nSaved run-level test results to {STATS_OUT}")

    plot_run_level_dots(summary, counters)

    n_seeds = len(seed_dirs)
    if n_seeds < 15:
        print(
            f"\nCaveat: only {n_seeds} run(s) per type. Treat any 'significant' "
            "result above as a pilot-stage hypothesis, not a firm conclusion "
            "(recommended 20-30 runs/type for stable power)."
        )


if __name__ == "__main__":
    main()
