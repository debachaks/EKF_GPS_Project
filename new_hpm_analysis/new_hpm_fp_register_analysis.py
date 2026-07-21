"""Run-level comparison of FP registers for the new-HPM-mapping runs
(seed_new_1..5), using the cleaned files from CLEAN_HPC_NEW_HPM/.

Registers are decoded as IEEE-754 doubles (bit-cast, not hex->int) as in
seed_new_fp_register_analysis.py. Since cleaning already drops any register
that's all-zero for a given run/mode, the columns present here are already
the "live" ones. Each register is reduced to one per-run median (same
run-level treatment as new_hpm_run_level_hpmcounter_analysis.py - this
covers both registers that are a single fixed value for the whole run and
registers that vary row-to-row, since the median collapses a constant run
to that constant), then the 5 normal-run medians are compared against the
5 anomaly-run medians per register.
"""

import glob
import os
import struct
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from hpmcounter_analysis import benjamini_hochberg, cliffs_delta  # noqa: E402

CLEAN_ROOT = os.path.join(SCRIPT_DIR, "CLEAN_HPC_NEW_HPM")
SUMMARY_OUT = os.path.join(SCRIPT_DIR, "new_hpm_fp_run_level_summary.csv")
STATS_OUT = os.path.join(SCRIPT_DIR, "new_hpm_fp_run_level_stats.csv")

CANDIDATE_DOUBLE_REGS = (
    [f"ft{i}" for i in range(12)]
    + [f"fs{i}" for i in range(12)]
    + [f"fa{i}" for i in range(8)]
)
CANDIDATE_STATUS_REGS = ["fflags", "frm", "fcsr"]
ANOMALY_TYPES = ["drift", "jump", "replay"]
ALL_TYPES = ["normal"] + ANOMALY_TYPES


def find_seed_dirs():
    return sorted(d for d in glob.glob(os.path.join(CLEAN_ROOT, "seed_new_[0-9]*")) if os.path.isdir(d))


def bits_to_double(hex_str):
    s = str(hex_str).strip()
    bits = int(s, 16) if s.lower().startswith("0x") else int(s)
    bits &= (1 << 64) - 1
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def present_fp_columns(columns):
    double_regs = [c for c in CANDIDATE_DOUBLE_REGS if c in columns]
    status_regs = [c for c in CANDIDATE_STATUS_REGS if c in columns]
    return double_regs, status_regs


def per_run_medians(seed_dir, run_type, double_regs, status_regs):
    path = os.path.join(seed_dir, f"ekf_{run_type}_hpc.csv")
    df = pd.read_csv(path)
    medians = {}
    for col in double_regs:
        medians[col] = df[col].map(bits_to_double).median()
    for col in status_regs:
        vals = df[col].map(lambda v: int(str(v), 16) if str(v).lower().startswith("0x") else int(v))
        medians[col] = vals.median()
    return medians


def build_run_summary(seed_dirs):
    first_df = pd.read_csv(os.path.join(seed_dirs[0], "ekf_normal_hpc.csv"))
    double_regs, status_regs = present_fp_columns(set(first_df.columns))
    fp_cols = double_regs + status_regs

    rows = []
    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        for run_type in ALL_TYPES:
            medians = per_run_medians(seed_dir, run_type, double_regs, status_regs)
            row = {"seed": seed_name, "run_type": run_type}
            row.update(medians)
            rows.append(row)
    return pd.DataFrame(rows), fp_cols


def run_tests(summary, fp_cols):
    normal = summary[summary["run_type"] == "normal"]

    rows = []
    for anomaly in ANOMALY_TYPES:
        anomaly_df = summary[summary["run_type"] == anomaly]
        for reg in fp_cols:
            x = anomaly_df[reg].to_numpy()
            y = normal[reg].to_numpy()
            u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
            delta = cliffs_delta(x, y)
            rows.append({
                "register": reg,
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

    summary, fp_cols = build_run_summary(seed_dirs)
    print(f"FP-related registers surviving cleaning: {fp_cols}")
    summary.to_csv(SUMMARY_OUT, index=False)
    print(f"Saved per-run medians to {SUMMARY_OUT}")

    results = run_tests(summary, fp_cols)
    results.to_csv(STATS_OUT, index=False)
    print(results.to_string(index=False))
    print(f"\nSaved run-level test results to {STATS_OUT}")

    n_seeds = len(seed_dirs)
    if n_seeds < 15:
        print(
            f"\nCaveat: only {n_seeds} run(s) per type. Treat any 'significant' "
            "result above as a pilot-stage hypothesis, not a firm conclusion."
        )


if __name__ == "__main__":
    main()
