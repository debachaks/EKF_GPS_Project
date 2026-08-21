"""Window-size sweep for the CUSUM-with-robust-floor metric
(cusum_zfloor_metric.py), trying W = 5, 10, 15, 20, 25, 30 to see whether
any window size gets this design competitive with the window-diff G
metric's 13/20 clean drift result at L=10.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")
SWEEP_OUT_PATH = os.path.join(RESULTS_DIR, "cusum_zfloor_sweep.csv")

WINDOWS = [5, 10, 15, 20, 25, 30]
STEP = 1
K = 0.1
PERCENTILE = 95
EPS = 1e-9
STRONG = ["hpmcounter3", "hpmcounter4", "hpmcounter5", "hpmcounter8", "hpmcounter10"]


def windowed_mean(z_values, window):
    n = len(z_values)
    results = []
    for start in range(0, n - window + 1, STEP):
        w = z_values[start:start + window]
        if np.any(np.isnan(w)):
            continue
        results.append((start, w.mean()))
    return results


def cusum(z_series):
    c_plus, c_minus = 0.0, 0.0
    out = []
    for z in z_series:
        c_plus = max(0.0, c_plus + z - K)
        c_minus = max(0.0, c_minus - z - K)
        out.append(max(c_plus, c_minus))
    return out


def run_one_window(ts, window):
    rows = []
    for (counter, mode, seed), group in ts[ts["counter"].isin(STRONG)].groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        raw_values = group["raw_value"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, xbar in windowed_mean(raw_values, window):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + window - 1], "xbar": xbar,
            })
    xbar_df = pd.DataFrame(rows)

    zw_rows = []
    for counter in STRONG:
        c_data = xbar_df[xbar_df["counter"] == counter]
        normal_pivot = c_data[c_data["mode"] == "normal"].pivot(
            index="seed", columns="window_end_iter", values="xbar"
        )
        mu = normal_pivot.mean(axis=0)
        sigma = normal_pivot.std(axis=0)
        q10 = normal_pivot.quantile(0.10, axis=0)
        floor = (q10 - mu).abs()
        denom = np.maximum(sigma, floor)
        denom_safe = denom.where(denom > EPS, EPS)

        for (mode, seed), grp in c_data.groupby(["mode", "seed"]):
            grp = grp.set_index("window_end_iter").reindex(mu.index)
            zw = (grp["xbar"] - mu) / denom_safe
            for it, val in zw.items():
                zw_rows.append({
                    "counter": counter, "mode": mode, "seed": seed,
                    "window_end_iter": it, "z_w": val,
                })
    zw_df = pd.DataFrame(zw_rows)

    cvalue_rows = []
    for (counter, mode, seed), grp in zw_df.groupby(["counter", "mode", "seed"]):
        grp = grp.sort_values("window_end_iter")
        c_values = cusum(grp["z_w"].to_numpy())
        for it, cv in zip(grp["window_end_iter"], c_values):
            cvalue_rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": it, "C_value": cv,
            })
    cvalue_df = pd.DataFrame(cvalue_rows)

    normal_c = cvalue_df[cvalue_df["mode"] == "normal"]
    flat_pooled = normal_c.groupby("counter")["C_value"].quantile(PERCENTILE / 100).reset_index(name="H_z_pooled")
    per_trial_max = normal_c.groupby(["counter", "seed"])["C_value"].max().reset_index(name="M_j")
    per_trial = per_trial_max.groupby("counter")["M_j"].quantile(PERCENTILE / 100).reset_index(name="H_z_trialmax")
    thresholds = flat_pooled.merge(per_trial, on="counter")

    attack = cvalue_df[cvalue_df["mode"].isin(["jump", "drift"])].merge(thresholds, on="counter")
    attack["flag_pooled"] = attack["C_value"] > attack["H_z_pooled"]
    attack["flag_trialmax"] = attack["C_value"] > attack["H_z_trialmax"]

    run_detected = attack.groupby(["counter", "mode", "seed"])[["flag_pooled", "flag_trialmax"]].any().reset_index()
    summary = run_detected.groupby(["counter", "mode"])[["flag_pooled", "flag_trialmax"]].sum().reset_index()
    return summary


def main():
    ts = pd.read_csv(TIMESERIES_PATH)

    sweep_rows = []
    print(f"{'W':>3s}  {'drift_pooled_avg':>16s}  {'drift_trialmax_avg':>19s}  {'jump_pooled_avg':>15s}  {'jump_trialmax_avg':>18s}")
    for w in WINDOWS:
        summary = run_one_window(ts, w)
        for _, row in summary.iterrows():
            sweep_rows.append({
                "window": w, "counter": row["counter"], "mode": row["mode"],
                "detected_pooled": int(row["flag_pooled"]), "detected_trialmax": int(row["flag_trialmax"]),
            })

        drift = summary[summary["mode"] == "drift"]
        jump = summary[summary["mode"] == "jump"]
        dp = drift["flag_pooled"].mean()
        dt = drift["flag_trialmax"].mean()
        jp = jump["flag_pooled"].mean()
        jt = jump["flag_trialmax"].mean()
        print(f"{w:3d}  {dp:16.1f}  {dt:19.1f}  {jp:15.1f}  {jt:18.1f}")

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(SWEEP_OUT_PATH, index=False)
    print(f"\nSaved {SWEEP_OUT_PATH}")


if __name__ == "__main__":
    main()
