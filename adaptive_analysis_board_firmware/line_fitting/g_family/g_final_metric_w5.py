"""G_final at W=5 -- identical construction to g_final_metric.py, just a
different window size, kept as a separate script/output set (rather than
overwriting the W=10 results) since other work (g_final_sweep.py,
heatmap comparisons) still depends on the W=10 version.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")

W = 5
STEP = 1
PERCENTILE = 95
SIGMA_FLOOR = 1e-6
ONSET_ITER = 150

G_OUT_PATH = os.path.join(RESULTS_DIR, "g_final_w5_g.csv")
GNEW_OUT_PATH = os.path.join(RESULTS_DIR, "g_final_w5_gneww.csv")
BASELINE_OUT_PATH = os.path.join(RESULTS_DIR, "g_final_w5_baseline.csv")
GSCORE_OUT_PATH = os.path.join(RESULTS_DIR, "g_final_w5_gscore.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, "g_final_w5_thresholds.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, "g_final_w5_detection_summary.csv")


def windowed_mean(z_values, window):
    n = len(z_values)
    results = []
    for start in range(0, n - window + 1, STEP):
        w = z_values[start:start + window]
        if np.any(np.isnan(w)):
            continue
        results.append((start, w.mean()))
    return results


def compute_all_G(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z_values = group["z"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, g in windowed_mean(z_values, W):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + W - 1], "G": g,
            })
    return pd.DataFrame(rows)


def main():
    ts = pd.read_csv(TIMESERIES_PATH)
    counters = sorted(ts["counter"].unique(), key=lambda c: int(c.replace("hpmcounter", "")))

    g_df = compute_all_G(ts)
    g_df.to_csv(G_OUT_PATH, index=False)
    print(f"Saved {G_OUT_PATH} ({len(g_df)} rows)")

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
    gnew_df.to_csv(GNEW_OUT_PATH, index=False)
    print(f"Saved {GNEW_OUT_PATH} ({len(gnew_df)} rows)")

    baseline_rows, gscore_rows = [], []
    sigma_warnings = []
    for counter in counters:
        c_data = gnew_df[gnew_df["counter"] == counter]
        normal_pivot = c_data[c_data["mode"] == "normal"].pivot(
            index="seed", columns="window_end_iter", values="G_new_w"
        )
        mu = normal_pivot.mean(axis=0)
        sigma = normal_pivot.std(axis=0)

        n_fragile = int((sigma < SIGMA_FLOOR).sum())
        if n_fragile > 0:
            sigma_warnings.append((counter, n_fragile, len(sigma)))
        sigma_safe = sigma.where(sigma > SIGMA_FLOOR, SIGMA_FLOOR)

        for it in mu.index:
            baseline_rows.append({
                "counter": counter, "window_end_iter": it,
                "mu_Gnew": mu[it], "sigma_Gnew": sigma_safe[it],
                "sigma_fragile": bool(sigma[it] < SIGMA_FLOOR),
            })

        for (mode, seed), grp in c_data.groupby(["mode", "seed"]):
            grp = grp.set_index("window_end_iter").reindex(mu.index)
            g_score = (grp["G_new_w"] - mu) / sigma_safe
            for it, val in g_score.items():
                gscore_rows.append({
                    "counter": counter, "mode": mode, "seed": seed,
                    "window_end_iter": it, "g": val,
                    "sigma_fragile": bool(sigma[it] < SIGMA_FLOOR),
                })

    print("\n=== sigma_Gnew fragility check (positions with sigma_Gnew < 1e-6) ===")
    if sigma_warnings:
        for counter, n_fragile, n_total in sigma_warnings:
            print(f"  {counter}: {n_fragile}/{n_total} fragile positions")
    else:
        print("  none -- sigma_Gnew is well-behaved everywhere")

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(BASELINE_OUT_PATH, index=False)
    print(f"\nSaved {BASELINE_OUT_PATH}")

    gscore_df = pd.DataFrame(gscore_rows)
    gscore_df.to_csv(GSCORE_OUT_PATH, index=False)
    print(f"Saved {GSCORE_OUT_PATH} ({len(gscore_df)} rows)")

    normal_g = gscore_df[(gscore_df["mode"] == "normal") & (~gscore_df["sigma_fragile"])].copy()
    normal_g["abs_g"] = normal_g["g"].abs()
    per_run_max = normal_g.groupby(["counter", "seed"])["abs_g"].max().reset_index(name="M_j")
    thresholds = (
        per_run_max.groupby("counter")["M_j"]
        .quantile(PERCENTILE / 100)
        .reset_index(name="H_g")
    )
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print("\nH_g (95th percentile of normal runs' max |g|, excluding sigma-fragile positions), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack = gscore_df[
        gscore_df["mode"].isin(["jump", "drift"]) & (~gscore_df["sigma_fragile"])
    ].merge(thresholds, on="counter")
    attack["flagged"] = attack["g"].abs() > attack["H_g"]
    attack["post_onset_flag"] = attack["flagged"] & (attack["window_end_iter"] >= ONSET_ITER)

    run_detected = attack.groupby(["counter", "mode", "seed"])["post_onset_flag"].any().reset_index(name="detected")
    summary = run_detected.groupby(["counter", "mode"])["detected"].agg(n_detected="sum", n_runs="count").reset_index()
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print(f"\nPer-run detection summary (G_final, W={W}, post-onset-only):")
    print(summary.to_string(index=False))
    print(f"Saved {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
