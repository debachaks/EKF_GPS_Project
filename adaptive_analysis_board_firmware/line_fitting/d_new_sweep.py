"""Window-size sweep for d(t) (d_new_metric.py), mean-centering only,
trying W = 5, 10, 15, 20, 25, 30 to see how detection changes. Same
per-trial-max-then-95th-percentile threshold and sigma-fragility guard as
d_new_metric.py.

A run only counts as "detected" if it has a flagged window at or after
the attack onset (iter >= 150, where attack_active flips per
seed_old/test_seed_pre_post_attack_analysis.py) -- pre-onset flags are
real noise in some counters (hpmcounter9 especially) and are excluded so
the reported rate isn't inflated by false fires that happen before the
attack is even injected.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")
SWEEP_OUT_PATH = os.path.join(RESULTS_DIR, "d_new_sweep.csv")

WINDOWS = [5, 10, 15, 20, 25, 30]
STEP = 1
PERCENTILE = 95
SIGMA_FLOOR = 1e-6
ONSET_ITER = 150  # attack_active flips at t=150/299 -- see seed_old/test_seed_pre_post_attack_analysis.py
STRONG = ["hpmcounter3", "hpmcounter4", "hpmcounter5", "hpmcounter8", "hpmcounter10", "hpmcounter9"]


def windowed_beta(raw_values, window):
    n = len(raw_values)
    s = np.arange(window, dtype=float)
    s_bar = s.mean()
    s_dev = s - s_bar
    ss_s = np.sum(s_dev**2)

    results = []
    for start in range(0, n - window + 1, STEP):
        w = raw_values[start:start + window]
        if np.any(np.isnan(w)):
            continue
        beta = np.sum(s_dev * w) / ss_s
        results.append((start, beta))
    return results


def run_one_window(ts, window):
    rows = []
    for (counter, mode, seed), group in ts[ts["counter"].isin(STRONG)].groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        raw = group["raw_value"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, beta in windowed_beta(raw, window):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + window - 1], "beta": beta,
            })
    beta_df = pd.DataFrame(rows)

    dnew_rows = []
    for counter in STRONG:
        c_data = beta_df[beta_df["counter"] == counter]
        normal_pivot = c_data[c_data["mode"] == "normal"].pivot(
            index="seed", columns="window_end_iter", values="beta"
        )
        mean_center = normal_pivot.mean(axis=0)

        for (mode, seed), grp in c_data.groupby(["mode", "seed"]):
            grp = grp.set_index("window_end_iter").reindex(normal_pivot.columns)
            d_w = (grp["beta"] - mean_center).abs()
            d_new = d_w.diff().abs()
            for it, val in d_new.items():
                if pd.isna(val):
                    continue
                dnew_rows.append({
                    "counter": counter, "mode": mode, "seed": seed,
                    "window_end_iter": it, "D_new_w": val,
                })
    dnew_df = pd.DataFrame(dnew_rows)

    dscore_rows = []
    for counter in STRONG:
        c_data = dnew_df[dnew_df["counter"] == counter]
        normal_pivot = c_data[c_data["mode"] == "normal"].pivot(
            index="seed", columns="window_end_iter", values="D_new_w"
        )
        mu = normal_pivot.mean(axis=0)
        sigma = normal_pivot.std(axis=0)
        sigma_fragile = sigma < SIGMA_FLOOR
        sigma_safe = sigma.where(sigma > SIGMA_FLOOR, SIGMA_FLOOR)

        for (mode, seed), grp in c_data.groupby(["mode", "seed"]):
            grp = grp.set_index("window_end_iter").reindex(mu.index)
            d_score = (grp["D_new_w"] - mu) / sigma_safe
            for it, val in d_score.items():
                dscore_rows.append({
                    "counter": counter, "mode": mode, "seed": seed,
                    "window_end_iter": it, "d": val,
                    "sigma_fragile": bool(sigma_fragile[it]),
                })
    dscore_df = pd.DataFrame(dscore_rows)

    normal_d = dscore_df[(dscore_df["mode"] == "normal") & (~dscore_df["sigma_fragile"])].copy()
    normal_d["abs_d"] = normal_d["d"].abs()
    per_run_max = normal_d.groupby(["counter", "seed"])["abs_d"].max().reset_index(name="M_j")
    thresholds = per_run_max.groupby("counter")["M_j"].quantile(PERCENTILE / 100).reset_index(name="H_d")

    attack = dscore_df[
        dscore_df["mode"].isin(["jump", "drift"]) & (~dscore_df["sigma_fragile"])
    ].merge(thresholds, on="counter")
    attack["flagged"] = attack["d"].abs() > attack["H_d"]
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
