"""New metric per the whiteboard design: a windowed-mean z-score with a
robust quantile-based denominator floor (fixing the sigma-collapse
failure mode found earlier in the V and window-diff g metrics), fed into
a two-sided CUSUM.

Step 1 -- windowed mean of the RAW COUNTER VALUE (not the already-
baselined z series), window size W, ending at position t, for every run:
    xbar(t) = mean(raw_value_i)  for i in the W-sample window ending at t
This baselines the windowed statistic directly against the 20 normal
trials in one pass, rather than baselining per-iteration first (z) and
then baselining the windowed mean of THAT a second time.

Step 2 -- normal-mode baseline, per counter, per t (from the 20 normal
trials' xbar(t) values at that SAME position):
    mu_0(t)    = mean of the 20 normal trials' xbar(t)
    sigma_0(t) = std  of the 20 normal trials' xbar(t)
    floor(t)   = |Q10%(the 20 normal trials' xbar(t)) - mu_0(t)|
        -- a robust, quantile-based floor: how far below the mean the
        10th-percentile normal trial sits, used as a lower bound on the
        denominator so a coincidentally tiny sigma_0(t) at one position
        can't blow up the z-score into a meaningless huge value.

Step 3 -- z_w(t) for every run (normal/jump/drift):
    z_w(t) = (xbar(t) - mu_0(t)) / max(sigma_0(t), floor(t))

Step 4 -- two-sided CUSUM, small slack k=0.1:
    C_plus(t)  = max(0, C_plus(t-1)  + z_w(t) - k)
    C_minus(t) = max(0, C_minus(t-1) - z_w(t) - k)
    C_value(t) = max(C_plus(t), C_minus(t))

Step 5 -- threshold H_z, calibrated TWO ways for comparison:
    (i)  flat-pooled:      Q95 of ALL (20 seeds x ~290 positions) C_value
    (ii) per-trial-max:    Q95 of {max over t of C_value, per normal trial}
                           (same convention as h_D/Qgs elsewhere in this
                           pipeline)

Detection: C_value(t) > H_z for any t => flagged. Reported separately for
both threshold variants.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")

W = 10
STEP = 1
K = 0.1
PERCENTILE = 95
EPS = 1e-9

XBAR_OUT_PATH = os.path.join(RESULTS_DIR, "cusum_zfloor_xbar.csv")
BASELINE_OUT_PATH = os.path.join(RESULTS_DIR, "cusum_zfloor_baseline.csv")
ZW_OUT_PATH = os.path.join(RESULTS_DIR, "cusum_zfloor_zw.csv")
CVALUE_OUT_PATH = os.path.join(RESULTS_DIR, "cusum_zfloor_cvalue.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, "cusum_zfloor_thresholds.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, "cusum_zfloor_detection_summary.csv")


def windowed_mean(z_values, window):
    n = len(z_values)
    results = []
    for start in range(0, n - window + 1, STEP):
        w = z_values[start:start + window]
        if np.any(np.isnan(w)):
            continue
        results.append((start, w.mean()))
    return results


def compute_all_xbar(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        raw = group["raw_value"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, xbar in windowed_mean(raw, W):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + W - 1], "xbar": xbar,
            })
    return pd.DataFrame(rows)


def cusum(z_series):
    c_plus, c_minus = 0.0, 0.0
    out = []
    for z in z_series:
        c_plus = max(0.0, c_plus + z - K)
        c_minus = max(0.0, c_minus - z - K)
        out.append(max(c_plus, c_minus))
    return out


def main():
    ts = pd.read_csv(TIMESERIES_PATH)
    counters = sorted(ts["counter"].unique(), key=lambda c: int(c.replace("hpmcounter", "")))

    xbar_df = compute_all_xbar(ts)
    xbar_df.to_csv(XBAR_OUT_PATH, index=False)
    print(f"Saved {XBAR_OUT_PATH} ({len(xbar_df)} rows)")

    baseline_rows, zw_rows = [], []
    for counter in counters:
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

        for it in mu.index:
            baseline_rows.append({
                "counter": counter, "window_end_iter": it,
                "mu_0": mu[it], "sigma_0": sigma[it], "floor": floor[it], "denom": denom_safe[it],
            })

        for (mode, seed), grp in c_data.groupby(["mode", "seed"]):
            grp = grp.set_index("window_end_iter").reindex(mu.index)
            zw = (grp["xbar"] - mu) / denom_safe
            for it, val in zw.items():
                zw_rows.append({
                    "counter": counter, "mode": mode, "seed": seed,
                    "window_end_iter": it, "z_w": val,
                })

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(BASELINE_OUT_PATH, index=False)
    print(f"Saved {BASELINE_OUT_PATH}")

    zw_df = pd.DataFrame(zw_rows)
    zw_df.to_csv(ZW_OUT_PATH, index=False)
    print(f"Saved {ZW_OUT_PATH} ({len(zw_df)} rows)")

    # CUSUM per (counter, mode, seed)
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
    cvalue_df.to_csv(CVALUE_OUT_PATH, index=False)
    print(f"Saved {CVALUE_OUT_PATH} ({len(cvalue_df)} rows)")

    normal_c = cvalue_df[cvalue_df["mode"] == "normal"]

    # (i) flat-pooled: Q95 of ALL normal C_value observations
    flat_pooled = normal_c.groupby("counter")["C_value"].quantile(PERCENTILE / 100).reset_index(name="H_z_pooled")

    # (ii) per-trial-max: Q95 of {max C_value per normal trial}
    per_trial_max = normal_c.groupby(["counter", "seed"])["C_value"].max().reset_index(name="M_j")
    per_trial = per_trial_max.groupby("counter")["M_j"].quantile(PERCENTILE / 100).reset_index(name="H_z_trialmax")

    thresholds = flat_pooled.merge(per_trial, on="counter")
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print("\nThresholds (both methods), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack = cvalue_df[cvalue_df["mode"].isin(["jump", "drift"])].merge(thresholds, on="counter")
    attack["flag_pooled"] = attack["C_value"] > attack["H_z_pooled"]
    attack["flag_trialmax"] = attack["C_value"] > attack["H_z_trialmax"]

    run_detected = attack.groupby(["counter", "mode", "seed"])[["flag_pooled", "flag_trialmax"]].any().reset_index()
    summary = run_detected.groupby(["counter", "mode"]).agg(
        n_runs=("seed", "count"),
        n_detected_pooled=("flag_pooled", "sum"),
        n_detected_trialmax=("flag_trialmax", "sum"),
    ).reset_index()
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print(f"\nPer-run detection summary (W={W}, k={K}):")
    print(summary.to_string(index=False))
    print(f"Saved {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
