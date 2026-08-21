"""Window-size sweep for G_final (g_final_metric.py: single sliding
window mean, diffed across time, z-scored), trying W = 5, 10, 15, 20, 25,
30 -- same set used for the D_final sweep, for direct comparison across
metrics. Same sigma-fragility guard, per-trial-max-then-95th-percentile
threshold, and post-onset-only detection rule (iter >= 150) as the base
script.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")
SWEEP_OUT_PATH = os.path.join(RESULTS_DIR, "g_final_sweep.csv")

WINDOWS = [5, 10, 15, 20, 25, 30]
STEP = 1
PERCENTILE = 95
SIGMA_FLOOR = 1e-6
ONSET_ITER = 150
STRONG = ["hpmcounter3", "hpmcounter4", "hpmcounter5", "hpmcounter8", "hpmcounter10", "hpmcounter9"]


def windowed_mean(z_values, window):
    n = len(z_values)
    results = []
    for start in range(0, n - window + 1, STEP):
        w = z_values[start:start + window]
        if np.any(np.isnan(w)):
            continue
        results.append((start, w.mean()))
    return results


def run_one_window(ts, window):
    rows = []
    for (counter, mode, seed), group in ts[ts["counter"].isin(STRONG)].groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z_values = group["z"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, g in windowed_mean(z_values, window):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + window - 1], "G": g,
            })
    g_df = pd.DataFrame(rows)

    gnew_rows = []
    for (counter, mode, seed), grp in g_df.groupby(["counter", "mode", "seed"]):
        grp = grp.sort_values("window_end_iter")
        g_new = grp["G"].diff().abs()
        for it, val in zip(grp["window_end_iter"], g_new):
            if pd.isna(val):
                continue
            gnew_rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": it, "G_new_w": val,
            })
    gnew_df = pd.DataFrame(gnew_rows)

    gscore_rows = []
    for counter in STRONG:
        c_data = gnew_df[gnew_df["counter"] == counter]
        normal_pivot = c_data[c_data["mode"] == "normal"].pivot(
            index="seed", columns="window_end_iter", values="G_new_w"
        )
        mu = normal_pivot.mean(axis=0)
        sigma = normal_pivot.std(axis=0)
        sigma_fragile = sigma < SIGMA_FLOOR
        sigma_safe = sigma.where(sigma > SIGMA_FLOOR, SIGMA_FLOOR)

        for (mode, seed), grp in c_data.groupby(["mode", "seed"]):
            grp = grp.set_index("window_end_iter").reindex(mu.index)
            g_score = (grp["G_new_w"] - mu) / sigma_safe
            for it, val in g_score.items():
                gscore_rows.append({
                    "counter": counter, "mode": mode, "seed": seed,
                    "window_end_iter": it, "g": val,
                    "sigma_fragile": bool(sigma_fragile[it]),
                })
    gscore_df = pd.DataFrame(gscore_rows)

    normal_g = gscore_df[(gscore_df["mode"] == "normal") & (~gscore_df["sigma_fragile"])].copy()
    normal_g["abs_g"] = normal_g["g"].abs()
    per_run_max = normal_g.groupby(["counter", "seed"])["abs_g"].max().reset_index(name="M_j")
    thresholds = per_run_max.groupby("counter")["M_j"].quantile(PERCENTILE / 100).reset_index(name="H_g")

    attack = gscore_df[
        gscore_df["mode"].isin(["jump", "drift"]) & (~gscore_df["sigma_fragile"])
    ].merge(thresholds, on="counter")
    attack["flagged"] = attack["g"].abs() > attack["H_g"]
    attack["post_onset_flag"] = attack["flagged"] & (attack["window_end_iter"] >= ONSET_ITER)

    run_detected = attack.groupby(["counter", "mode", "seed"])["post_onset_flag"].any().reset_index(name="detected")
    summary = run_detected.groupby(["counter", "mode"])["detected"].sum().reset_index()
    return summary


def main():
    ts = pd.read_csv(TIMESERIES_PATH)

    sweep_rows = []
    for w in WINDOWS:
        print(f"Running W={w}...")
        summary = run_one_window(ts, w)
        for _, row in summary.iterrows():
            sweep_rows.append({
                "window": w, "counter": row["counter"], "mode": row["mode"],
                "detected": int(row["detected"]),
            })

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(SWEEP_OUT_PATH, index=False)
    print(f"\nSaved {SWEEP_OUT_PATH}")

    for mode in ["drift", "jump"]:
        print(f"\n=== {mode} ===")
        header = "Counter".ljust(14) + "".join(f"W={w}".ljust(8) for w in WINDOWS)
        print(header)
        for c in STRONG:
            row = c.ljust(14)
            for w in WINDOWS:
                match = sweep_df[(sweep_df.counter == c) & (sweep_df.window == w) & (sweep_df["mode"] == mode)]["detected"]
                v = match.values[0] if len(match) else "n/a"
                row += f"{v}/20".ljust(8)
            print(row)


if __name__ == "__main__":
    main()
