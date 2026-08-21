"""Whole-run cumulative detector, aimed specifically at drift's failure
mode under the windowed D/V/z_stat detectors: a handful of drift seeds
(seed5, seed14) have z elevated across MOST of the run (z>1 for 200+ of
299 iterations) but never spike hard enough, or step sharply enough
within any single 10-sample window, to trip a local peak (z_stat) or
local slope (D) detector. A locally-constant elevated series has slope
~0 in every window and a peak that can stay under the single-window
threshold indefinitely -- exactly the shape a windowed detector is blind
to, and exactly the shape a cumulative statistic is built for.

Two whole-run statistics per (counter, mode, seed), both computed on the
raw z-score (not windowed):

    exceed_count = count(z[i] > 1)   over the whole run
        A z>1 threshold is a natural, unit-driven choice (z is already
        standardized) rather than a fit/tuned constant.

    CUSUM: S[i] = max(0, S[i-1] + (z[i] - k)),  k = 0.5
        Standard one-sided CUSUM with reference value k = half the
        target shift size (targeting a ~1-sigma sustained shift). Flags
        via S_max = max(S) over the run. This is the classic tool for
        detecting small, persistent shifts that never look extreme at
        any single point.

Thresholds h_count, h_cusum are calibrated the same way as the rest of
this pipeline: 95th percentile of the statistic's value across the
normal-mode baseline runs, per counter.

Reads zscore_baseline.py's zscore_timeseries.csv.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")
STATS_OUT_PATH = os.path.join(RESULTS_DIR, "cumulative_stats.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, "cumulative_thresholds.csv")
FLAGGED_OUT_PATH = os.path.join(RESULTS_DIR, "cumulative_attack_flags.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, "cumulative_detection_summary.csv")

Z_EXCEED_THRESHOLD = 1.0
CUSUM_K = 0.5
PERCENTILE = 95


def cusum_max(z):
    s = 0.0
    s_max = 0.0
    for zi in z:
        s = max(0.0, s + (zi - CUSUM_K))
        s_max = max(s_max, s)
    return s_max


def compute_stats(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        z = group.sort_values("iter")["z"].to_numpy()
        rows.append({
            "counter": counter, "mode": mode, "seed": seed,
            "exceed_count": int(np.sum(z > Z_EXCEED_THRESHOLD)),
            "cusum_max": cusum_max(z),
        })
    return pd.DataFrame(rows)


def build_thresholds(stats_df):
    normal = stats_df[stats_df["mode"] == "normal"]
    thresholds = (
        normal.groupby("counter")[["exceed_count", "cusum_max"]]
        .quantile(PERCENTILE / 100)
        .rename(columns={"exceed_count": "h_count", "cusum_max": "h_cusum"})
        .reset_index()
    )
    return thresholds


def seed_sort_key(seed):
    return int(seed.replace("seed", ""))


def main():
    ts = pd.read_csv(TIMESERIES_PATH)

    stats_df = compute_stats(ts)
    stats_df.to_csv(STATS_OUT_PATH, index=False)
    print(f"Saved {STATS_OUT_PATH} ({len(stats_df)} rows)")

    thresholds = build_thresholds(stats_df)
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print(f"\nThresholds (95th percentile of normal runs), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack_modes = sorted(ts.loc[ts["mode"] != "normal", "mode"].unique())
    attack = stats_df[stats_df["mode"].isin(attack_modes)].merge(thresholds, on="counter")
    attack["count_flag"] = attack["exceed_count"] > attack["h_count"]
    attack["cusum_flag"] = attack["cusum_max"] > attack["h_cusum"]
    attack["flag_or"] = attack["count_flag"] | attack["cusum_flag"]
    attack.to_csv(FLAGGED_OUT_PATH, index=False)
    print(f"\nSaved {FLAGGED_OUT_PATH} ({len(attack)} rows)")

    summary = attack.groupby(["counter", "mode"]).agg(
        n_runs=("seed", "count"),
        n_count_only=("count_flag", "sum"),
        n_cusum_only=("cusum_flag", "sum"),
        n_or=("flag_or", "sum"),
    ).reset_index()
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print("\nPer-run detection summary:")
    print(summary.to_string(index=False))
    print(f"Saved {SUMMARY_OUT_PATH}")

    print("\n=== drift, per-seed (flag_or across strong counters 3/4/5/8/10) ===")
    strong = ["hpmcounter3", "hpmcounter4", "hpmcounter5", "hpmcounter8", "hpmcounter10"]
    drift = attack[(attack["mode"] == "drift") & (attack["counter"].isin(strong))]
    pivot = drift.pivot(index="seed", columns="counter", values="flag_or")
    pivot = pivot.reindex(sorted(pivot.index, key=seed_sort_key))
    pivot = pivot[[c for c in strong if c in pivot.columns]]
    print(pivot.to_string())
    any_fired = pivot.any(axis=1)
    print(f"\nDrift seeds with at least one strong counter firing: {any_fired.sum()}/{len(pivot)}")
    print(f"Still silent: {list(pivot.index[~any_fired])}")


if __name__ == "__main__":
    main()
