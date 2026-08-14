"""Runs the z-scored V detector (same math as v_zscore_w50.py) at several
window sizes in one pass, specifically to chase hpmcounter9's apparent
sweet spot below W_V=30 (it jumped from 8/20 to 15/20 drift going from
W_V=50 to W_V=30 -- does it keep improving narrower, or was 30 the peak?).

Also reports, for each mode, the UNION across counters if each counter
uses its OWN best window size rather than one shared window for all --
i.e. hpmcounter9 at its best W_V, strong counters at theirs.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")

WINDOWS = [10, 15, 20, 25, 30]
STEP = 1
PERCENTILE = 95


def compute_variation(z_values, window):
    results = []
    n = len(z_values)
    for start in range(0, n - window + 1, STEP):
        w = z_values[start:start + window]
        if np.any(np.isnan(w)):
            continue
        variance = np.var(w, ddof=1)
        results.append((start, np.log1p(variance)))
    return results


def compute_raw_v(ts, window):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z = group["z"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, v in compute_variation(z, window):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + window - 1], "V": v,
            })
    return pd.DataFrame(rows)


def run_one_window(ts, counters, window):
    v_df = compute_raw_v(ts, window)

    zv_rows = []
    for counter in counters:
        c_data = v_df[v_df["counter"] == counter]
        normal_pivot = c_data[c_data["mode"] == "normal"].pivot(
            index="seed", columns="window_end_iter", values="V"
        )
        mu = normal_pivot.mean(axis=0)
        sigma = normal_pivot.std(axis=0)
        sigma_safe = sigma.where(sigma > 1e-9, 1e-9)

        for (mode, seed), g in c_data.groupby(["mode", "seed"]):
            g = g.set_index("window_end_iter").reindex(mu.index)
            zv = (g["V"] - mu) / sigma_safe
            for it, val in zv.items():
                zv_rows.append({
                    "counter": counter, "mode": mode, "seed": seed,
                    "window_end_iter": it, "z_V": val,
                })
    zv_df = pd.DataFrame(zv_rows)

    normal_zv = zv_df[zv_df["mode"] == "normal"]
    per_run_max = normal_zv.groupby(["counter", "seed"])["z_V"].max().reset_index(name="M_j")
    thresholds = (
        per_run_max.groupby("counter")["M_j"]
        .quantile(PERCENTILE / 100)
        .reset_index(name="h_zV")
    )

    attack = zv_df[zv_df["mode"].isin(["jump", "drift"])].merge(thresholds, on="counter")
    attack["flagged"] = attack["z_V"] > attack["h_zV"]

    run_detected = attack.groupby(["counter", "mode", "seed"])["flagged"].any().reset_index(name="detected")
    return run_detected


def main():
    ts = pd.read_csv(TIMESERIES_PATH)
    counters = sorted(ts["counter"].unique(), key=lambda c: int(c.replace("hpmcounter", "")))
    strong = ["hpmcounter3", "hpmcounter4", "hpmcounter5", "hpmcounter8", "hpmcounter10"]

    all_results = {}
    print("=== hpmcounter9 detection count by window size ===")
    for w in WINDOWS:
        run_detected = run_one_window(ts, counters, w)
        all_results[w] = run_detected

        c9 = run_detected[run_detected["counter"] == "hpmcounter9"]
        n_drift = c9[(c9["mode"] == "drift") & c9["detected"]]["seed"].nunique()
        n_jump = c9[(c9["mode"] == "jump") & c9["detected"]]["seed"].nunique()
        print(f"W_V={w:3d}:  drift {n_drift}/20   jump {n_jump}/20")

    # also add the already-computed W=50/70 numbers for context (from prior runs)
    print("\n(for reference, already computed earlier: W_V=50 -> drift 8/20, jump 12/20;"
          " W_V=70 -> drift 7/20, jump 9/20)")

    best_w9 = max(WINDOWS, key=lambda w: all_results[w][
        (all_results[w]["counter"] == "hpmcounter9") & (all_results[w]["mode"] == "drift") & all_results[w]["detected"]
    ]["seed"].nunique())
    print(f"\nBest window for hpmcounter9 drift among {WINDOWS}: W_V={best_w9}")

    # union: hpmcounter9 at its best window, strong counters at W_V=50 (their established sweet spot)
    w9_seeds = set(all_results[best_w9][
        (all_results[best_w9]["counter"] == "hpmcounter9")
        & (all_results[best_w9]["mode"] == "drift")
        & all_results[best_w9]["detected"]
    ]["seed"])

    print(f"\nhpmcounter9 (W_V={best_w9}) drift-detected seeds: {sorted(w9_seeds, key=lambda s: int(s.replace('seed','')))}")


if __name__ == "__main__":
    main()
