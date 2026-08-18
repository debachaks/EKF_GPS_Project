"""New V metric based on windowed energy of the already-baselined z series.

Design:
    1) Use the existing z metric values from zscore_timeseries.csv.
    2) For each window of size W, compute the windowed energy:
           E_w(t) = mean(z_i^2) for i in the W-sample window ending at t
       and then compress it with log(1 + E_w(t)) to keep it on a scale
       comparable across counters.
    3) Baseline that energy statistic using the 20 normal trials at the
       same window position:
           mu_0(t) = mean(V_E_w(t) across 20 normal trials at position t)
           sigma_0(t) = std(V_E_w(t) across 20 normal trials at position t)
           floor(t) = |Q10%(V_E_w(t) across 20 normal trials) - mu_0(t)|
       Use the robust floor inside the same max(sigma, floor) denominator
       convention used in the other new metrics.
    4) Standardized score:
           V_w(t) = |V_E_w(t) - mu_0(t)| / max(sigma_0(t), floor(t))
    5) Thresholds: compare both a flat-pooled Q95 and a per-trial-max Q95.

This is intentionally distinct from the original variability_metric.py,
which measures variance of the z values in a window. The new metric is
built from the already-baselined z values themselves and treats the
window as a sustained-energy deviation from the normal baseline instead of
just a local spread measure.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")

W = 10
STEP = 1
PERCENTILE = 95
EPS = 1e-9

ENERGY_OUT_PATH = os.path.join(RESULTS_DIR, "v_zscore_energy.csv")
BASELINE_OUT_PATH = os.path.join(RESULTS_DIR, "v_zscore_baseline.csv")
V_OUT_PATH = os.path.join(RESULTS_DIR, "v_zscore_values.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, "v_zscore_thresholds.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, "v_zscore_detection_summary.csv")


def windowed_energy(z_values, window):
    """Return a list of (window_start, log1p(mean(z^2))) for every full window."""
    n = len(z_values)
    results = []
    for start in range(0, n - window + 1, STEP):
        w = z_values[start:start + window]
        if np.any(np.isnan(w)):
            continue
        e = np.mean(np.square(w))
        v_e = np.log1p(e)
        results.append((start, v_e))
    return results


def compute_all_energy(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z = group["z"].to_numpy(dtype=float)
        iters = group["iter"].to_numpy()
        for start, v_e in windowed_energy(z, W):
            rows.append({
                "counter": counter,
                "mode": mode,
                "seed": seed,
                "window_end_iter": iters[start + W - 1],
                "V_E": v_e,
            })
    return pd.DataFrame(rows)


def main():
    ts = pd.read_csv(TIMESERIES_PATH)
    counters = sorted(ts["counter"].unique(), key=lambda c: int(c.replace("hpmcounter", "")))

    energy_df = compute_all_energy(ts)
    energy_df.to_csv(ENERGY_OUT_PATH, index=False)
    print(f"Saved {ENERGY_OUT_PATH} ({len(energy_df)} rows)")

    baseline_rows, v_rows = [], []
    for counter in counters:
        c_data = energy_df[energy_df["counter"] == counter]
        normal_pivot = c_data[c_data["mode"] == "normal"].pivot(
            index="seed", columns="window_end_iter", values="V_E"
        )
        mu = normal_pivot.mean(axis=0)
        sigma = normal_pivot.std(axis=0)
        q10 = normal_pivot.quantile(0.10, axis=0)
        floor = (q10 - mu).abs()
        denom = np.maximum(sigma, floor)
        denom_safe = denom.where(denom > EPS, EPS)

        for it in mu.index:
            baseline_rows.append({
                "counter": counter,
                "window_end_iter": it,
                "mu_0": mu[it],
                "sigma_0": sigma[it],
                "floor": floor[it],
                "denom": denom_safe[it],
            })

        for (mode, seed), grp in c_data.groupby(["mode", "seed"]):
            grp = grp.set_index("window_end_iter").reindex(mu.index)
            v_metric = (grp["V_E"] - mu).abs() / denom_safe
            for it, val in v_metric.items():
                v_rows.append({
                    "counter": counter,
                    "mode": mode,
                    "seed": seed,
                    "window_end_iter": it,
                    "V_metric": val,
                })

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(BASELINE_OUT_PATH, index=False)
    print(f"Saved {BASELINE_OUT_PATH} ({len(baseline_df)} rows)")

    v_df = pd.DataFrame(v_rows)
    v_df.to_csv(V_OUT_PATH, index=False)
    print(f"Saved {V_OUT_PATH} ({len(v_df)} rows)")

    normal_v = v_df[v_df["mode"] == "normal"]

    flat_pooled = normal_v.groupby("counter")["V_metric"].quantile(PERCENTILE / 100).reset_index(name="h_V_pooled")

    per_trial_max = normal_v.groupby(["counter", "seed"])["V_metric"].max().reset_index(name="M_j")
    per_trial = per_trial_max.groupby("counter")["M_j"].quantile(PERCENTILE / 100).reset_index(name="h_V_trialmax")

    thresholds = flat_pooled.merge(per_trial, on="counter")
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print("\nThresholds (both methods), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack = v_df[v_df["mode"].isin(["jump", "drift"])].merge(thresholds, on="counter")
    attack["flag_pooled"] = attack["V_metric"] > attack["h_V_pooled"]
    attack["flag_trialmax"] = attack["V_metric"] > attack["h_V_trialmax"]

    run_detected = attack.groupby(["counter", "mode", "seed"])[["flag_pooled", "flag_trialmax"]].any().reset_index()
    summary = run_detected.groupby(["counter", "mode"]).agg(
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
