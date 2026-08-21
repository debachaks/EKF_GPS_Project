"""Same as variability_metric.py, but with WINDOW=20 instead of 10 -- the V
companion to trend_score_windowed_w20.py's D, for the same window-size
comparison. Reads the same zscore_timeseries.csv; outputs go to separate
_w20-suffixed files so the original WINDOW=10 results aren't touched.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")
V_OUT_PATH = os.path.join(RESULTS_DIR, "variation_windowed_results_w20.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, "variation_threshold_by_counter_w20.csv")
FLAGGED_OUT_PATH = os.path.join(RESULTS_DIR, "variation_attack_flags_w20.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, "variation_detection_summary_w20.csv")

WINDOW = 20
STEP = 1
PERCENTILE = 95


def compute_variation(z_values):
    results = []
    n = len(z_values)
    for start in range(0, n - WINDOW + 1, STEP):
        window = z_values[start:start + WINDOW]
        if np.any(np.isnan(window)):
            continue
        variance = np.var(window, ddof=1)
        v = np.log1p(variance)
        results.append((start, variance, v))
    return results


def compute_all_windows(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z = group["z"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, variance, v in compute_variation(z):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + WINDOW - 1],
                "variance": variance, "V": v,
            })
    return pd.DataFrame(rows)


def build_thresholds(variation_df):
    normal = variation_df[variation_df["mode"] == "normal"]
    per_run_max = normal.groupby(["counter", "seed"])["V"].max().reset_index(name="M_j")
    thresholds = (
        per_run_max.groupby("counter")["M_j"]
        .quantile(PERCENTILE / 100)
        .reset_index(name="h_V")
    )
    return thresholds


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = pd.read_csv(TIMESERIES_PATH)

    variation_df = compute_all_windows(ts)
    variation_df.to_csv(V_OUT_PATH, index=False)
    print(f"Saved {V_OUT_PATH} ({len(variation_df)} rows)")

    thresholds = build_thresholds(variation_df)
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print("\nh_V (W=20) (95th percentile of normal runs' max V), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack_modes = sorted(ts.loc[ts["mode"] != "normal", "mode"].unique())
    if not attack_modes:
        print("\nNo attack-mode runs present yet -- skipping flagging step.")
        return

    attack = variation_df[variation_df["mode"].isin(attack_modes)].merge(thresholds, on="counter")
    attack["flagged"] = attack["V"] > attack["h_V"]
    attack.to_csv(FLAGGED_OUT_PATH, index=False)
    print(f"\nSaved {FLAGGED_OUT_PATH} ({len(attack)} rows)")

    run_detected = attack.groupby(["counter", "mode", "seed"])["flagged"].any().reset_index(name="detected")
    summary = run_detected.groupby(["counter", "mode"])["detected"].agg(n_detected="sum", n_runs="count").reset_index()
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print("\nPer-run detection summary (W=20) (a run counts as detected if ANY window exceeds h_V):")
    print(summary.to_string(index=False))
    print(f"Saved {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
