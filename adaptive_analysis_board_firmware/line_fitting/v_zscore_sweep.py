"""Window-size sweep for the new V metric based on windowed z-energy.

This mirrors the earlier sweep pattern but uses the logic from
v_zscore_metric.py: the windowed statistic is the average of z^2 within the
window, compressed via log(1 + mean(z^2)). It is then normalized against the
20 normal trials at the same window position using the robust
max(sigma, floor) denominator.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")
SWEEP_OUT_PATH = os.path.join(RESULTS_DIR, "v_zscore_sweep.csv")

WINDOWS = [5, 10, 15, 20, 25, 30]
STEP = 1
PERCENTILE = 95
EPS = 1e-9
STRONG = ["hpmcounter3", "hpmcounter4", "hpmcounter5", "hpmcounter8", "hpmcounter10"]


def windowed_energy(z_values, window):
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


def run_one_window(ts, window):
    rows = []
    for (counter, mode, seed), group in ts[ts["counter"].isin(STRONG)].groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z = group["z"].to_numpy(dtype=float)
        iters = group["iter"].to_numpy()
        for start, v_e in windowed_energy(z, window):
            rows.append({
                "counter": counter,
                "mode": mode,
                "seed": seed,
                "window_end_iter": iters[start + window - 1],
                "V_E": v_e,
            })
    energy_df = pd.DataFrame(rows)

    v_rows = []
    for counter in STRONG:
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
    v_df = pd.DataFrame(v_rows)

    normal_v = v_df[v_df["mode"] == "normal"]
    flat_pooled = normal_v.groupby("counter")["V_metric"].quantile(PERCENTILE / 100).reset_index(name="h_V_pooled")
    per_trial_max = normal_v.groupby(["counter", "seed"])["V_metric"].max().reset_index(name="M_j")
    per_trial = per_trial_max.groupby("counter")["M_j"].quantile(PERCENTILE / 100).reset_index(name="h_V_trialmax")
    thresholds = flat_pooled.merge(per_trial, on="counter")

    attack = v_df[v_df["mode"].isin(["jump", "drift"])].merge(thresholds, on="counter")
    attack["flag_pooled"] = attack["V_metric"] > attack["h_V_pooled"]
    attack["flag_trialmax"] = attack["V_metric"] > attack["h_V_trialmax"]

    run_detected = attack.groupby(["counter", "mode", "seed"])[["flag_pooled", "flag_trialmax"]].any().reset_index()
    summary = run_detected.groupby(["counter", "mode"]).agg(
        n_runs=("seed", "count"),
        n_detected_pooled=("flag_pooled", "sum"),
        n_detected_trialmax=("flag_trialmax", "sum"),
    ).reset_index()
    return summary


def main():
    ts = pd.read_csv(TIMESERIES_PATH)
    sweep_rows = []
    for w in WINDOWS:
        summary = run_one_window(ts, w)
        for _, row in summary.iterrows():
            sweep_rows.append({
                "window": w,
                "counter": row["counter"],
                "mode": row["mode"],
                "detected_pooled": int(row["n_detected_pooled"]),
                "detected_trialmax": int(row["n_detected_trialmax"]),
            })

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(SWEEP_OUT_PATH, index=False)
    print(f"Saved {SWEEP_OUT_PATH}")

    for method in ["detected_trialmax", "detected_pooled"]:
        print(f"\n=== {method} ===")
        header = "Counter".ljust(14)
        for w in WINDOWS:
            header += f"W={w} drift".ljust(14)
        for w in WINDOWS:
            header += f"W={w} jump".ljust(13)
        print(header)

        for c in STRONG:
            row = c.ljust(14)
            for w in WINDOWS:
                v = sweep_df[(sweep_df["counter"] == c) & (sweep_df["window"] == w) & (sweep_df["mode"] == "drift")][method].values[0]
                row += f"{v}/20".ljust(14)
            for w in WINDOWS:
                v = sweep_df[(sweep_df["counter"] == c) & (sweep_df["window"] == w) & (sweep_df["mode"] == "jump")][method].values[0]
                row += f"{v}/20".ljust(13)
            print(row)


if __name__ == "__main__":
    main()
