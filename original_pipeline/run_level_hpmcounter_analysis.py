"""Run-level comparison of hpmcounter3-hpmcounter10 across anomaly types.

Unlike hpmcounter_analysis.py (which compares individual timestamped rows
within a single run), this script treats each seed_N_data/ run as ONE
independent sample: every counter is reduced to its per-run median first,
then normal-run medians are compared against anomaly-run medians across
seeds. This is the correct unit of replication since rows within a run are
an autocorrelated time series, not independent observations.

With only a handful of seeds, treat any "significant" result here as a
pilot-stage hypothesis, not a firm conclusion (see README/session notes on
required run counts for real power).
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from data_preprocessing import hex_to_int

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SEED_GLOB = os.path.join(SRC_DIR, "seed_*_data")
PLOT_DIR = os.path.join(SRC_DIR, "run_level_plots")
SUMMARY_OUT = os.path.join(SRC_DIR, "run_level_summary.csv")
STATS_OUT = os.path.join(SRC_DIR, "run_level_hpmcounter_stats.csv")

COUNTERS = [f"hpmcounter{i}" for i in range(3, 11)]
ANOMALY_TYPES = ["drift", "jump", "replay"]
ALL_TYPES = ["normal"] + ANOMALY_TYPES


def find_seed_dirs():
    return sorted(d for d in glob.glob(SEED_GLOB) if os.path.isdir(d))


def per_run_medians(seed_dir, run_type):
    path = os.path.join(seed_dir, f"ekf_{run_type}_hpc.csv")
    df = pd.read_csv(path)
    counters = df[COUNTERS].apply(lambda col: col.map(hex_to_int))
    return counters.median()


def build_run_summary():
    rows = []
    for seed_dir in find_seed_dirs():
        seed_name = os.path.basename(seed_dir)
        for run_type in ALL_TYPES:
            medians = per_run_medians(seed_dir, run_type)
            row = {"seed": seed_name, "run_type": run_type}
            row.update(medians.to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    diff = x[:, None] - y[None, :]
    return np.sign(diff).sum() / (len(x) * len(y))


def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted


def run_tests(summary):
    normal = summary[summary["run_type"] == "normal"]

    rows = []
    for anomaly in ANOMALY_TYPES:
        anomaly_df = summary[summary["run_type"] == anomaly]
        for counter in COUNTERS:
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


def plot_run_level_dots(summary):
    os.makedirs(PLOT_DIR, exist_ok=True)
    normal = summary[summary["run_type"] == "normal"]

    for counter in COUNTERS:
        plt.figure(figsize=(6, 4))
        groups = ["normal"] + ANOMALY_TYPES
        for i, run_type in enumerate(groups):
            data = normal if run_type == "normal" else summary[summary["run_type"] == run_type]
            ys = data[counter].to_numpy()
            xs = np.full(len(ys), i) + np.random.uniform(-0.05, 0.05, len(ys))
            plt.scatter(xs, ys, label=run_type)
        plt.xticks(range(len(groups)), groups)
        plt.ylabel(f"{counter} (per-run median)")
        plt.title(f"{counter}: per-run medians by run type")
        plt.tight_layout()
        out_path = os.path.join(PLOT_DIR, f"{counter}_run_level.png")
        plt.savefig(out_path)
        plt.close()


def main():
    seed_dirs = find_seed_dirs()
    if not seed_dirs:
        print(f"No seed_*_data folders found matching {SEED_GLOB}")
        return
    print(f"Found {len(seed_dirs)} seeds: {[os.path.basename(d) for d in seed_dirs]}")

    summary = build_run_summary()
    summary.to_csv(SUMMARY_OUT, index=False)
    print(f"Saved per-run medians to {SUMMARY_OUT}")

    results = run_tests(summary)
    results.to_csv(STATS_OUT, index=False)
    print(results.to_string(index=False))
    print(f"\nSaved run-level test results to {STATS_OUT}")

    n_seeds = len(seed_dirs)
    if n_seeds < 15:
        print(
            f"\nCaveat: only {n_seeds} run(s) per type. Treat any 'significant' "
            "result above as a pilot-stage hypothesis, not a firm conclusion "
            "(recommended 20-30 runs/type for stable power)."
        )

    plot_run_level_dots(summary)
    print(f"Saved per-counter dot plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
