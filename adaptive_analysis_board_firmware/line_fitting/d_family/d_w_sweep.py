"""Window-size sweep for D_w (d_w_metric.py), trying W = 5, 10, 15, 20, 25,
30 to see how detection changes, using the trial-max threshold (the
pooled threshold was already shown to be unusable for this metric).

Matches the no-denominator D_w redesign in d_w_metric.py:
D_w(t) = |beta_w(t) - center_w,beta(t)|, mean centering only.

A run only counts as "detected" if it has a flagged window at or after
the attack onset (iter >= 150, where attack_active flips per
seed_old/test_seed_pre_post_attack_analysis.py) -- pre-onset flags are
real noise in some counters (hpmcounter9 especially, per the same
contamination check run on D_final/V_final/G_final) and are excluded so
the reported rate isn't inflated by false fires that happen before the
attack is even injected.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")
SWEEP_OUT_PATH = os.path.join(RESULTS_DIR, "d_w_sweep.csv")

WINDOWS = [5, 10, 15, 20, 25, 30]
STEP = 1
PERCENTILE = 95
ONSET_ITER = 150
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

    dw_rows = []
    for counter in STRONG:
        c_data = beta_df[beta_df["counter"] == counter]
        normal_pivot = c_data[c_data["mode"] == "normal"].pivot(
            index="seed", columns="window_end_iter", values="beta"
        )
        mean_center = normal_pivot.mean(axis=0)

        for (mode, seed), grp in c_data.groupby(["mode", "seed"]):
            grp = grp.set_index("window_end_iter").reindex(normal_pivot.columns)
            dw = (grp["beta"] - mean_center).abs()
            for it, val in dw.items():
                dw_rows.append({
                    "counter": counter, "mode": mode, "seed": seed,
                    "window_end_iter": it, "D_w": val,
                })
    dw_df = pd.DataFrame(dw_rows)

    normal_dw = dw_df[dw_df["mode"] == "normal"]
    per_trial_max = normal_dw.groupby(["counter", "seed"])["D_w"].max().reset_index(name="M_j")
    thresholds = per_trial_max.groupby("counter")["M_j"].quantile(PERCENTILE / 100).reset_index(name="H_D")

    attack = dw_df[dw_df["mode"].isin(["jump", "drift"])].merge(thresholds, on="counter")
    attack["flagged"] = attack["D_w"] > attack["H_D"]
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
                "window": w, "counter": row["counter"],
                "mode": row["mode"], "detected": int(row["detected"]),
            })

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(SWEEP_OUT_PATH, index=False)
    print(f"\nSaved {SWEEP_OUT_PATH}")

    for mode in ["drift", "jump"]:
        print(f"\n=== {mode} (mean centering, post-onset-only) ===")
        header = "Counter".ljust(14) + "".join(f"W={w}".ljust(8) for w in WINDOWS)
        print(header)
        for c in STRONG:
            row = c.ljust(14)
            for w in WINDOWS:
                v = sweep_df[(sweep_df.counter == c) & (sweep_df.window == w) & (sweep_df["mode"] == mode)]["detected"].values[0]
                row += f"{v}/20".ljust(8)
            print(row)


if __name__ == "__main__":
    main()
