"""Same as v_zscore_w50.py, but with W_V=30 -- narrowing down whether the
V sweet spot sits below W_V=50 (W_V=70 was already shown to be worse for
drift on every counter). Outputs go to separate _w30-suffixed files.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")

V_RAW_OUT_PATH = os.path.join(RESULTS_DIR, "v_raw_w30.csv")
V_BASELINE_OUT_PATH = os.path.join(RESULTS_DIR, "v_baseline_mu_sigma_w30.csv")
V_ZSCORE_OUT_PATH = os.path.join(RESULTS_DIR, "v_zscore_w30.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, "v_zscore_threshold_w30.csv")
FLAGGED_OUT_PATH = os.path.join(RESULTS_DIR, "v_zscore_attack_flags_w30.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, "v_zscore_detection_summary_w30.csv")

W_V = 30
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


def compute_raw_v(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z = group["z"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, v in compute_variation(z, W_V):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + W_V - 1], "V": v,
            })
    return pd.DataFrame(rows)


def main():
    ts = pd.read_csv(TIMESERIES_PATH)
    counters = sorted(ts["counter"].unique(), key=lambda c: int(c.replace("hpmcounter", "")))

    v_df = compute_raw_v(ts)
    v_df.to_csv(V_RAW_OUT_PATH, index=False)
    print(f"Saved {V_RAW_OUT_PATH} ({len(v_df)} rows)")

    mu_rows, zv_rows = [], []
    for counter in counters:
        c_data = v_df[v_df["counter"] == counter]
        normal_pivot = c_data[c_data["mode"] == "normal"].pivot(
            index="seed", columns="window_end_iter", values="V"
        )
        mu = normal_pivot.mean(axis=0)
        sigma = normal_pivot.std(axis=0)
        sigma_safe = sigma.where(sigma > 1e-9, 1e-9)

        for it in mu.index:
            mu_rows.append({
                "counter": counter, "window_end_iter": it,
                "mu_V": mu[it], "sigma_V": sigma_safe[it],
            })

        for (mode, seed), g in c_data.groupby(["mode", "seed"]):
            g = g.set_index("window_end_iter").reindex(mu.index)
            zv = (g["V"] - mu) / sigma_safe
            for it, val in zv.items():
                zv_rows.append({
                    "counter": counter, "mode": mode, "seed": seed,
                    "window_end_iter": it, "z_V": val,
                })

    mu_df = pd.DataFrame(mu_rows)
    mu_df.to_csv(V_BASELINE_OUT_PATH, index=False)
    print(f"Saved {V_BASELINE_OUT_PATH}")

    zv_df = pd.DataFrame(zv_rows)
    zv_df.to_csv(V_ZSCORE_OUT_PATH, index=False)
    print(f"Saved {V_ZSCORE_OUT_PATH} ({len(zv_df)} rows)")

    normal_zv = zv_df[zv_df["mode"] == "normal"]
    per_run_max = normal_zv.groupby(["counter", "seed"])["z_V"].max().reset_index(name="M_j")
    thresholds = (
        per_run_max.groupby("counter")["M_j"]
        .quantile(PERCENTILE / 100)
        .reset_index(name="h_zV")
    )
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print("\nh_zV (95th percentile of normal runs' max z_V), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack = zv_df[zv_df["mode"].isin(["jump", "drift"])].merge(thresholds, on="counter")
    attack["flagged"] = attack["z_V"] > attack["h_zV"]
    attack.to_csv(FLAGGED_OUT_PATH, index=False)
    print(f"\nSaved {FLAGGED_OUT_PATH} ({len(attack)} rows)")

    run_detected = attack.groupby(["counter", "mode", "seed"])["flagged"].any().reset_index(name="detected")
    summary = run_detected.groupby(["counter", "mode"])["detected"].agg(n_detected="sum", n_runs="count").reset_index()
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print("\nPer-run detection summary (z_V, W_V=30) (a run counts as detected if ANY window exceeds h_zV):")
    print(summary.to_string(index=False))
    print(f"Saved {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
