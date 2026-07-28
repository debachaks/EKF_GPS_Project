"""Warm-up-window sensitivity check.

test_seed_pre_post_attack_analysis.py and a direct scan of the t5 scratch
register both traced hpmcounter5's jump-vs-normal separation to run order:
runs immediately preceded by a "normal" run show an elevated value,
regardless of their own mode, and jump happened to follow normal in 12/20
seeds (drift only 4/20, replay only 1/20) - explaining why jump looked
uniquely different in the whole-run averages.

If that's really a carryover artifact (leftover branch-predictor/pipeline
state from whatever ran immediately before), it should behave like a
warm-up transient: strongest right at the start of a run, fading as the
CPU executes enough of the *current* run's own instructions to retrain on
its own pattern. A genuine attack effect would not fade this way, since
the attack's effect on the EKF's computation doesn't depend on how long
ago the run started.

This script re-runs the two strongest pieces of evidence for the run-order
theory - the hpmcounter5 rate test and the t5-by-preceding-mode test -
after discarding the first WARMUP_ROWS rows of every run, and prints both
the WARMUP_ROWS=0 (original) and WARMUP_ROWS=25 results side by side.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED_OLD_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SEED_OLD_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from data_preprocessing import hex_to_int  # noqa: E402
from hpmcounter_analysis import cliffs_delta  # noqa: E402

CLEAN_ROOT = os.path.join(SEED_OLD_DIR, "CLEAN_HPC_TEST_SEED")
STATS_OUT = os.path.join(SCRIPT_DIR, "warmup_window_comparison.csv")

MODES = ["normal", "drift", "jump", "replay"]
WARMUP_ROWS_TO_TEST = (0, 25, 50, 100, 150)


def find_seed_names():
    dirs = sorted(d for d in glob.glob(os.path.join(CLEAN_ROOT, "test_seed_[0-9]*")) if os.path.isdir(d))
    return [os.path.basename(d) for d in dirs]


def load_trimmed(seed_name, mode, warmup_rows):
    path = os.path.join(CLEAN_ROOT, seed_name, f"ekf_{mode}_hpc.csv")
    df = pd.read_csv(path)
    if warmup_rows:
        df = df.iloc[warmup_rows:].reset_index(drop=True)
    return df


def hpmcounter5_rate_median(seed_name, mode, warmup_rows):
    df = load_trimmed(seed_name, mode, warmup_rows)
    mcycle_delta = df["mcycle"].map(hex_to_int).diff()
    counter_delta = df["hpmcounter5"].map(hex_to_int).diff()
    rate = (counter_delta / mcycle_delta).iloc[1:].replace([np.inf, -np.inf], np.nan)
    return rate.median()


def run_hpmcounter5_check(seed_names, warmup_rows):
    normal_vals = np.array([hpmcounter5_rate_median(s, "normal", warmup_rows) for s in seed_names])
    jump_vals = np.array([hpmcounter5_rate_median(s, "jump", warmup_rows) for s in seed_names])
    u, p = mannwhitneyu(jump_vals, normal_vals, alternative="two-sided")
    d = cliffs_delta(jump_vals, normal_vals)
    return {
        "check": "hpmcounter5_rate_jump_vs_normal",
        "warmup_rows": warmup_rows,
        "median_normal": np.median(normal_vals),
        "median_jump": np.median(jump_vals),
        "cliffs_delta": d,
        "p_value": p,
    }


def run_t5_prev_mode_check(seed_names, warmup_rows):
    rows = []
    for seed_name in seed_names:
        for mode in MODES:
            df = load_trimmed(seed_name, mode, warmup_rows)
            path = os.path.join(CLEAN_ROOT, seed_name, f"ekf_{mode}_hpc.csv")
            ts_first = pd.read_csv(path, usecols=["timestamp_ms"])["timestamp_ms"].map(hex_to_int).iloc[0]
            rows.append({
                "seed": seed_name,
                "mode": mode,
                "ts_first": ts_first,
                "t5_median": df["t5"].map(hex_to_int).median(),
            })
    result_df = pd.DataFrame(rows).sort_values("ts_first").reset_index(drop=True)
    result_df["prev_mode"] = result_df["mode"].shift(1)

    grouped = result_df.dropna(subset=["prev_mode"]).groupby("prev_mode")["t5_median"].median()
    preceded_by_normal = result_df[result_df["prev_mode"] == "normal"]["t5_median"].to_numpy()
    preceded_by_other = result_df[
        result_df["prev_mode"].notna() & (result_df["prev_mode"] != "normal")
    ]["t5_median"].to_numpy()
    u, p = mannwhitneyu(preceded_by_normal, preceded_by_other, alternative="two-sided")
    d = cliffs_delta(preceded_by_normal, preceded_by_other)
    stats_row = {
        "check": "t5_preceded_by_normal_vs_other",
        "warmup_rows": warmup_rows,
        "median_preceded_by_normal": np.median(preceded_by_normal),
        "median_preceded_by_other": np.median(preceded_by_other),
        "cliffs_delta": d,
        "p_value": p,
    }
    return stats_row, grouped


def main():
    seed_names = find_seed_names()
    if not seed_names:
        print(f"No cleaned test_seed_N folders found under {CLEAN_ROOT} - run test_seed_clean_hpc.py first")
        return
    print(f"Found {len(seed_names)} seeds\n")

    all_rows = []
    for warmup_rows in WARMUP_ROWS_TO_TEST:
        print(f"=== warmup_rows={warmup_rows} ({'original, no trimming' if warmup_rows == 0 else 'trimmed'}) ===")

        r1 = run_hpmcounter5_check(seed_names, warmup_rows)
        print(
            f"hpmcounter5 rate, jump vs normal: "
            f"median_normal={r1['median_normal']:.6f} median_jump={r1['median_jump']:.6f} "
            f"cliffs_delta={r1['cliffs_delta']:.3f} p={r1['p_value']:.6f}"
        )
        all_rows.append(r1)

        r2, grouped = run_t5_prev_mode_check(seed_names, warmup_rows)
        print(
            f"t5, preceded-by-normal vs preceded-by-other: "
            f"median_preceded_by_normal={r2['median_preceded_by_normal']:.1f} "
            f"median_preceded_by_other={r2['median_preceded_by_other']:.1f} "
            f"cliffs_delta={r2['cliffs_delta']:.3f} p={r2['p_value']:.6f}"
        )
        print("  t5 median by actual preceding mode:")
        print(grouped.to_string())
        all_rows.append(r2)
        print()

    results = pd.DataFrame(all_rows)
    results.to_csv(STATS_OUT, index=False)
    print(f"Saved comparison to {STATS_OUT}")


if __name__ == "__main__":
    main()
