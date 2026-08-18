"""Window-size sweep for V_final (v_final_metric.py: window-diff +
z-score built on the old V = |log(S^2_{t,W}+1)|), trying W = 5, 10, 20,
30, 50 -- wider than the D_final sweep since the old V metric was known
to work better at larger windows. Same sigma-fragility guard,
per-trial-max-then-95th-percentile threshold, and post-onset-only
detection rule (iter >= 150) as the base script.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")
SWEEP_OUT_PATH = os.path.join(RESULTS_DIR, "v_final_sweep.csv")

WINDOWS = [5, 10, 20, 30, 50]
STEP = 1
PERCENTILE = 95
SIGMA_FLOOR = 1e-6
ONSET_ITER = 150
STRONG = ["hpmcounter3", "hpmcounter4", "hpmcounter5", "hpmcounter8", "hpmcounter10", "hpmcounter9"]


def windowed_V(z_values, window):
    n = len(z_values)
    results = []
    for start in range(0, n - window + 1, STEP):
        w = z_values[start:start + window]
        if np.any(np.isnan(w)):
            continue
        variance = np.var(w, ddof=1)
        v = abs(np.log1p(variance))
        results.append((start, v))
    return results


def run_one_window(ts, window):
    rows = []
    for (counter, mode, seed), group in ts[ts["counter"].isin(STRONG)].groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z_values = group["z"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, v in windowed_V(z_values, window):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + window - 1], "V": v,
            })
    v_df = pd.DataFrame(rows)

    vnew_rows = []
    for (counter, mode, seed), grp in v_df.groupby(["counter", "mode", "seed"]):
        grp = grp.sort_values("window_end_iter")
        v_new = grp["V"].diff().abs()
        for it, val in zip(grp["window_end_iter"], v_new):
            if pd.isna(val):
                continue
            vnew_rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": it, "V_new_w": val,
            })
    vnew_df = pd.DataFrame(vnew_rows)

    vscore_rows = []
    for counter in STRONG:
        c_data = vnew_df[vnew_df["counter"] == counter]
        normal_pivot = c_data[c_data["mode"] == "normal"].pivot(
            index="seed", columns="window_end_iter", values="V_new_w"
        )
        mu = normal_pivot.mean(axis=0)
        sigma = normal_pivot.std(axis=0)
        sigma_fragile = sigma < SIGMA_FLOOR
        sigma_safe = sigma.where(sigma > SIGMA_FLOOR, SIGMA_FLOOR)

        for (mode, seed), grp in c_data.groupby(["mode", "seed"]):
            grp = grp.set_index("window_end_iter").reindex(mu.index)
            v_score = (grp["V_new_w"] - mu) / sigma_safe
            for it, val in v_score.items():
                vscore_rows.append({
                    "counter": counter, "mode": mode, "seed": seed,
                    "window_end_iter": it, "v": val,
                    "sigma_fragile": bool(sigma_fragile[it]),
                })
    vscore_df = pd.DataFrame(vscore_rows)

    normal_v = vscore_df[(vscore_df["mode"] == "normal") & (~vscore_df["sigma_fragile"])].copy()
    normal_v["abs_v"] = normal_v["v"].abs()
    per_run_max = normal_v.groupby(["counter", "seed"])["abs_v"].max().reset_index(name="M_j")
    thresholds = per_run_max.groupby("counter")["M_j"].quantile(PERCENTILE / 100).reset_index(name="H_v")

    attack = vscore_df[
        vscore_df["mode"].isin(["jump", "drift"]) & (~vscore_df["sigma_fragile"])
    ].merge(thresholds, on="counter")
    attack["flagged"] = attack["v"].abs() > attack["H_v"]
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
