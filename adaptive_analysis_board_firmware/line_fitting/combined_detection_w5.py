"""Same as combined_detection.py, but built on the WINDOW=5 versions of D
and V (trend_score_windowed_w5.py, variability_metric_w5.py) instead of
WINDOW=10. Outputs go to separate _w5-suffixed files.
"""

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from trend_score_windowed_w5 import WINDOW, windowed_d  # noqa: E402
from variability_metric_w5 import compute_variation  # noqa: E402

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")
WINDOWED_OUT_PATH = os.path.join(RESULTS_DIR, "combined_detection_windowed_w5.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, "combined_detection_thresholds_w5.csv")
FLAGGED_OUT_PATH = os.path.join(RESULTS_DIR, "combined_detection_flags_w5.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, "combined_detection_summary_w5.csv")

STEP = 1
PERCENTILE = 95


def windowed_z(z_values):
    n = len(z_values)
    results = []
    for start in range(0, n - WINDOW + 1, STEP):
        window = z_values[start:start + WINDOW]
        if np.any(np.isnan(window)):
            continue
        results.append((start, np.max(np.abs(window))))
    return results


def compute_all_windows(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z = group["z"].to_numpy(dtype=float)
        iters = group["iter"].to_numpy()

        z_by_start = dict(windowed_z(z))
        d_by_start = {start: d for start, _beta, d in windowed_d(z)}
        v_by_start = {start: v for start, _variance, v in compute_variation(z)}

        starts = sorted(set(z_by_start) & set(d_by_start) & set(v_by_start))
        for start in starts:
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + WINDOW - 1],
                "z_stat": z_by_start[start],
                "D": d_by_start[start],
                "V": v_by_start[start],
            })
    return pd.DataFrame(rows)


def build_thresholds(windowed_df):
    normal = windowed_df[windowed_df["mode"] == "normal"]
    per_run_max = normal.groupby(["counter", "seed"])[["z_stat", "D", "V"]].max().reset_index()
    thresholds = (
        per_run_max.groupby("counter")[["z_stat", "D", "V"]]
        .quantile(PERCENTILE / 100)
        .rename(columns={"z_stat": "h_z", "D": "h_D", "V": "h_V"})
        .reset_index()
    )
    return thresholds


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = pd.read_csv(TIMESERIES_PATH)

    windowed_df = compute_all_windows(ts)
    windowed_df.to_csv(WINDOWED_OUT_PATH, index=False)
    print(f"Saved {WINDOWED_OUT_PATH} ({len(windowed_df)} rows)")

    thresholds = build_thresholds(windowed_df)
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print("\nThresholds (W=5) (95th percentile of normal runs' max), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack_modes = sorted(ts.loc[ts["mode"] != "normal", "mode"].unique())
    if not attack_modes:
        print("\nNo attack-mode runs present yet -- skipping flagging step.")
        return

    attack = windowed_df[windowed_df["mode"].isin(attack_modes)].merge(thresholds, on="counter")
    attack["z_flag"] = attack["z_stat"] > attack["h_z"]
    attack["d_flag"] = attack["D"] > attack["h_D"]
    attack["v_flag"] = attack["V"] > attack["h_V"]
    attack["n_agree"] = attack[["z_flag", "d_flag", "v_flag"]].sum(axis=1)
    attack["flag_or"] = attack["n_agree"] >= 1
    attack["flag_majority"] = attack["n_agree"] >= 2
    attack["flag_and"] = attack["n_agree"] == 3
    attack.to_csv(FLAGGED_OUT_PATH, index=False)
    print(f"\nSaved {FLAGGED_OUT_PATH} ({len(attack)} rows)")

    run_detected = attack.groupby(["counter", "mode", "seed"])[
        ["z_flag", "d_flag", "v_flag", "flag_or", "flag_majority", "flag_and"]
    ].any().reset_index()

    summary = run_detected.groupby(["counter", "mode"]).agg(
        n_runs=("seed", "count"),
        n_z_only=("z_flag", "sum"),
        n_d_only=("d_flag", "sum"),
        n_v_only=("v_flag", "sum"),
        n_or=("flag_or", "sum"),
        n_majority_2of3=("flag_majority", "sum"),
        n_and_all3=("flag_and", "sum"),
    ).reset_index()
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
