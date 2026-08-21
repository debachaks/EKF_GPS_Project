"""New D metric per the whiteboard design (item 3b): denominator removed
entirely -- D_w is now just the absolute deviation of the windowed slope
from its across-trial center, with no normal-mode spread/floor scaling.

    D_w(t) = |beta_w(t) - beta~_0,w,beta(t)|

    beta_w(t)         = OLS slope fit over a W-sample window of the RAW
                         COUNTER VALUE (not the already-baselined z
                         series) ending at t
    beta~_0,w,beta(t) = center of beta_w(t) across the 20 normal trials,
                         at that same window position -- tried two ways:
                         mean{beta_0,w} and median{beta_0,w}

Threshold H_D, calibrated TWO ways for comparison (same as before), for
EACH centering choice (mean, median):
    (i)  flat-pooled:   Q95 of ALL (20 seeds x ~positions) D_w observations
    (ii) per-trial-max: Q95 of {max over t of D_w, per normal trial}
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")

W = 10
STEP = 1
PERCENTILE = 95

BETA_OUT_PATH = os.path.join(RESULTS_DIR, "d_w_beta.csv")
BASELINE_OUT_PATH = os.path.join(RESULTS_DIR, "d_w_baseline.csv")
DW_OUT_PATH = os.path.join(RESULTS_DIR, "d_w_values.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, "d_w_thresholds.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, "d_w_detection_summary.csv")

_S = np.arange(W, dtype=float)
_S_BAR = _S.mean()
_S_DEV = _S - _S_BAR
_SS_S = np.sum(_S_DEV**2)


def windowed_beta(raw_values):
    """OLS slope over each W-sample window of the raw counter value."""
    n = len(raw_values)
    results = []
    for start in range(0, n - W + 1, STEP):
        w = raw_values[start:start + W]
        if np.any(np.isnan(w)):
            continue
        beta = np.sum(_S_DEV * w) / _SS_S
        results.append((start, beta))
    return results


def compute_all_beta(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        raw = group["raw_value"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, beta in windowed_beta(raw):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + W - 1], "beta": beta,
            })
    return pd.DataFrame(rows)


def main():
    ts = pd.read_csv(TIMESERIES_PATH)
    counters = sorted(ts["counter"].unique(), key=lambda c: int(c.replace("hpmcounter", "")))

    beta_df = compute_all_beta(ts)
    beta_df.to_csv(BETA_OUT_PATH, index=False)
    print(f"Saved {BETA_OUT_PATH} ({len(beta_df)} rows)")

    baseline_rows, dw_rows = [], []
    for counter in counters:
        c_data = beta_df[beta_df["counter"] == counter]
        normal_pivot = c_data[c_data["mode"] == "normal"].pivot(
            index="seed", columns="window_end_iter", values="beta"
        )
        center = {
            "mean": normal_pivot.mean(axis=0),
            "median": normal_pivot.median(axis=0),
        }

        for it in normal_pivot.columns:
            baseline_rows.append({
                "counter": counter, "window_end_iter": it,
                "mean_beta": center["mean"][it], "median_beta": center["median"][it],
            })

        for (mode, seed), grp in c_data.groupby(["mode", "seed"]):
            grp = grp.set_index("window_end_iter").reindex(normal_pivot.columns)
            for cname, cvals in center.items():
                dw = (grp["beta"] - cvals).abs()
                for it, val in dw.items():
                    dw_rows.append({
                        "counter": counter, "mode": mode, "seed": seed,
                        "window_end_iter": it, "center": cname, "D_w": val,
                    })

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(BASELINE_OUT_PATH, index=False)
    print(f"Saved {BASELINE_OUT_PATH}")

    dw_df = pd.DataFrame(dw_rows)
    dw_df.to_csv(DW_OUT_PATH, index=False)
    print(f"Saved {DW_OUT_PATH} ({len(dw_df)} rows)")

    normal_dw = dw_df[dw_df["mode"] == "normal"]

    flat_pooled = normal_dw.groupby(["counter", "center"])["D_w"].quantile(PERCENTILE / 100).reset_index(name="H_D_pooled")
    per_trial_max = normal_dw.groupby(["counter", "center", "seed"])["D_w"].max().reset_index(name="M_j")
    per_trial = per_trial_max.groupby(["counter", "center"])["M_j"].quantile(PERCENTILE / 100).reset_index(name="H_D_trialmax")

    thresholds = flat_pooled.merge(per_trial, on=["counter", "center"])
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print("\nThresholds (both methods x both centerings), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack = dw_df[dw_df["mode"].isin(["jump", "drift"])].merge(thresholds, on=["counter", "center"])
    attack["flag_pooled"] = attack["D_w"] > attack["H_D_pooled"]
    attack["flag_trialmax"] = attack["D_w"] > attack["H_D_trialmax"]

    run_detected = attack.groupby(["counter", "center", "mode", "seed"])[["flag_pooled", "flag_trialmax"]].any().reset_index()
    summary = run_detected.groupby(["counter", "center", "mode"]).agg(
        n_runs=("seed", "count"),
        n_detected_pooled=("flag_pooled", "sum"),
        n_detected_trialmax=("flag_trialmax", "sum"),
    ).reset_index()
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print(f"\nPer-run detection summary (W={W}):")
    print(summary.to_string(index=False))
    print(f"Saved {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
