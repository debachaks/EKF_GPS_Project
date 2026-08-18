"""V_final: same window-diff + z-score construction as d_final_metric.py,
but built on top of the OLD V = |log(S^2_{t,W} + 1)| (variability_metric.py)
instead of D = |beta/SE(beta)|.

    S^2_{t,W} = (1/(W-1)) * sum_{i=t-W+1}^{t} (z_i - zbar_{t,W})^2
                (residual/sample variance of z within the window, using
                the window's own mean)
    V(t)      = |log(S^2_{t,W} + 1)|

    V_new_w(t) = |V(t) - V(t-1)|

    v(t) = (V_new_w(t) - mu_Vnew(t)) / sigma_Vnew(t)

        mu_Vnew(t)    = mean of V_new_w(t) across the 20 normal trials,
                        at that window position
        sigma_Vnew(t) = std of V_new_w(t) across the 20 normal trials,
                        at that window position

Same sigma-fragility guard as d_final_metric.py (positions with
sigma_Vnew(t) < 1e-6 excluded from thresholding/flagging rather than
clamped) and same per-trial-max-then-95th-percentile threshold.

Detection: a run counts as detected only if a flag occurs at or after the
attack onset (iter >= 150, per seed_old/test_seed_pre_post_attack_analysis.py)
-- pre-onset flags are excluded so the rate isn't inflated by false fires
that happen before the attack is even injected.
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
SIGMA_FLOOR = 1e-6
ONSET_ITER = 150

V_OUT_PATH = os.path.join(RESULTS_DIR, "v_final_v.csv")
VNEW_OUT_PATH = os.path.join(RESULTS_DIR, "v_final_vneww.csv")
BASELINE_OUT_PATH = os.path.join(RESULTS_DIR, "v_final_baseline.csv")
VSCORE_OUT_PATH = os.path.join(RESULTS_DIR, "v_final_vscore.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, "v_final_thresholds.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, "v_final_detection_summary.csv")


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


def compute_all_V(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z_values = group["z"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, v in windowed_V(z_values, W):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + W - 1], "V": v,
            })
    return pd.DataFrame(rows)


def main():
    ts = pd.read_csv(TIMESERIES_PATH)
    counters = sorted(ts["counter"].unique(), key=lambda c: int(c.replace("hpmcounter", "")))

    v_df = compute_all_V(ts)
    v_df.to_csv(V_OUT_PATH, index=False)
    print(f"Saved {V_OUT_PATH} ({len(v_df)} rows)")

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
    vnew_df.to_csv(VNEW_OUT_PATH, index=False)
    print(f"Saved {VNEW_OUT_PATH} ({len(vnew_df)} rows)")

    baseline_rows, vscore_rows = [], []
    sigma_warnings = []
    for counter in counters:
        c_data = vnew_df[vnew_df["counter"] == counter]
        normal_pivot = c_data[c_data["mode"] == "normal"].pivot(
            index="seed", columns="window_end_iter", values="V_new_w"
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
                "mu_Vnew": mu[it], "sigma_Vnew": sigma_safe[it],
                "sigma_fragile": bool(sigma[it] < SIGMA_FLOOR),
            })

        for (mode, seed), grp in c_data.groupby(["mode", "seed"]):
            grp = grp.set_index("window_end_iter").reindex(mu.index)
            v_score = (grp["V_new_w"] - mu) / sigma_safe
            for it, val in v_score.items():
                vscore_rows.append({
                    "counter": counter, "mode": mode, "seed": seed,
                    "window_end_iter": it, "v": val,
                    "sigma_fragile": bool(sigma[it] < SIGMA_FLOOR),
                })

    print("\n=== sigma_Vnew fragility check (positions with sigma_Vnew < 1e-6) ===")
    if sigma_warnings:
        for counter, n_fragile, n_total in sigma_warnings:
            print(f"  {counter}: {n_fragile}/{n_total} fragile positions")
    else:
        print("  none -- sigma_Vnew is well-behaved everywhere")

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(BASELINE_OUT_PATH, index=False)
    print(f"\nSaved {BASELINE_OUT_PATH}")

    vscore_df = pd.DataFrame(vscore_rows)
    vscore_df.to_csv(VSCORE_OUT_PATH, index=False)
    print(f"Saved {VSCORE_OUT_PATH} ({len(vscore_df)} rows)")

    normal_v = vscore_df[(vscore_df["mode"] == "normal") & (~vscore_df["sigma_fragile"])].copy()
    normal_v["abs_v"] = normal_v["v"].abs()
    per_run_max = normal_v.groupby(["counter", "seed"])["abs_v"].max().reset_index(name="M_j")
    thresholds = (
        per_run_max.groupby("counter")["M_j"]
        .quantile(PERCENTILE / 100)
        .reset_index(name="H_v")
    )
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print("\nH_v (95th percentile of normal runs' max |v|, excluding sigma-fragile positions), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack = vscore_df[
        vscore_df["mode"].isin(["jump", "drift"]) & (~vscore_df["sigma_fragile"])
    ].merge(thresholds, on="counter")
    attack["flagged"] = attack["v"].abs() > attack["H_v"]
    attack["post_onset_flag"] = attack["flagged"] & (attack["window_end_iter"] >= ONSET_ITER)

    run_detected = attack.groupby(["counter", "mode", "seed"])["post_onset_flag"].any().reset_index(name="detected")
    summary = run_detected.groupby(["counter", "mode"])["detected"].agg(n_detected="sum", n_runs="count").reset_index()
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print(f"\nPer-run detection summary (V_final, W={W}, post-onset-only):")
    print(summary.to_string(index=False))
    print(f"Saved {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
