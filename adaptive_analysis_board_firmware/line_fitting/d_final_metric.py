"""D_final: same window-diff + z-score construction as d_new_metric.py,
but built on top of the ORIGINAL D = |beta / SE(beta)| (trend_score_windowed.py)
instead of the no-denominator D_w from d_w_metric.py (item 3b).

    D(t)       = |beta_w(t) / SE(beta_w(t))|   (windowed OLS slope
                 t-statistic on the already-baselined z series -- already
                 self-normalized, assumes normal-mode slope ~ 0, so no
                 separate cross-trial "center" is needed here)
    D_new_w(t) = |D(t) - D(t-1)|

    d(t) = (D_new_w(t) - mu_Dnew(t)) / sigma_Dnew(t)

        mu_Dnew(t)    = mean of D_new_w(t) across the 20 normal trials,
                        at that window position
        sigma_Dnew(t) = std of D_new_w(t) across the 20 normal trials,
                        at that window position

Same sigma-fragility guard as d_new_metric.py (positions with
sigma_Dnew(t) < 1e-6 excluded from thresholding/flagging rather than
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
SE_EPS = 1e-20   # avoids exact 0/0 on a perfectly flat window (matches trend_score_windowed.py)
SIGMA_FLOOR = 1e-6
ONSET_ITER = 150

D_OUT_PATH = os.path.join(RESULTS_DIR, "d_final_d.csv")
DNEW_OUT_PATH = os.path.join(RESULTS_DIR, "d_final_dneww.csv")
BASELINE_OUT_PATH = os.path.join(RESULTS_DIR, "d_final_baseline.csv")
DSCORE_OUT_PATH = os.path.join(RESULTS_DIR, "d_final_dscore.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, "d_final_thresholds.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, "d_final_detection_summary.csv")

_S = np.arange(W, dtype=float)
_S_BAR = _S.mean()
_S_DEV = _S - _S_BAR
_SS_S = np.sum(_S_DEV**2)


def windowed_d(z_values):
    n = len(z_values)
    results = []
    for start in range(0, n - W + 1, STEP):
        window = z_values[start:start + W]
        if np.any(np.isnan(window)):
            continue
        beta = np.sum(_S_DEV * window) / _SS_S
        alpha = window.mean() - beta * _S_BAR
        fitted = alpha + beta * _S
        resid = window - fitted
        sigma_eps2 = np.sum(resid**2) / (W - 2)
        se_beta = np.sqrt(sigma_eps2 / _SS_S + SE_EPS)
        d = abs(beta / se_beta)
        results.append((start, d))
    return results


def compute_all_D(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z_values = group["z"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, d in windowed_d(z_values):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + W - 1], "D": d,
            })
    return pd.DataFrame(rows)


def main():
    ts = pd.read_csv(TIMESERIES_PATH)
    counters = sorted(ts["counter"].unique(), key=lambda c: int(c.replace("hpmcounter", "")))

    d_df = compute_all_D(ts)
    d_df.to_csv(D_OUT_PATH, index=False)
    print(f"Saved {D_OUT_PATH} ({len(d_df)} rows)")

    dnew_rows = []
    for (counter, mode, seed), grp in d_df.groupby(["counter", "mode", "seed"]):
        grp = grp.sort_values("window_end_iter")
        d_new = grp["D"].diff().abs()
        for it, val in zip(grp["window_end_iter"], d_new):
            if pd.isna(val):
                continue
            dnew_rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": it, "D_new_w": val,
            })
    dnew_df = pd.DataFrame(dnew_rows)
    dnew_df.to_csv(DNEW_OUT_PATH, index=False)
    print(f"Saved {DNEW_OUT_PATH} ({len(dnew_df)} rows)")

    baseline_rows, dscore_rows = [], []
    sigma_warnings = []
    for counter in counters:
        c_data = dnew_df[dnew_df["counter"] == counter]
        normal_pivot = c_data[c_data["mode"] == "normal"].pivot(
            index="seed", columns="window_end_iter", values="D_new_w"
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
                "mu_Dnew": mu[it], "sigma_Dnew": sigma_safe[it],
                "sigma_fragile": bool(sigma[it] < SIGMA_FLOOR),
            })

        for (mode, seed), grp in c_data.groupby(["mode", "seed"]):
            grp = grp.set_index("window_end_iter").reindex(mu.index)
            d_score = (grp["D_new_w"] - mu) / sigma_safe
            for it, val in d_score.items():
                dscore_rows.append({
                    "counter": counter, "mode": mode, "seed": seed,
                    "window_end_iter": it, "d": val,
                    "sigma_fragile": bool(sigma[it] < SIGMA_FLOOR),
                })

    print("\n=== sigma_Dnew fragility check (positions with sigma_Dnew < 1e-6) ===")
    if sigma_warnings:
        for counter, n_fragile, n_total in sigma_warnings:
            print(f"  {counter}: {n_fragile}/{n_total} fragile positions")
    else:
        print("  none -- sigma_Dnew is well-behaved everywhere")

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(BASELINE_OUT_PATH, index=False)
    print(f"\nSaved {BASELINE_OUT_PATH}")

    dscore_df = pd.DataFrame(dscore_rows)
    dscore_df.to_csv(DSCORE_OUT_PATH, index=False)
    print(f"Saved {DSCORE_OUT_PATH} ({len(dscore_df)} rows)")

    normal_d = dscore_df[(dscore_df["mode"] == "normal") & (~dscore_df["sigma_fragile"])].copy()
    normal_d["abs_d"] = normal_d["d"].abs()
    per_run_max = normal_d.groupby(["counter", "seed"])["abs_d"].max().reset_index(name="M_j")
    thresholds = (
        per_run_max.groupby("counter")["M_j"]
        .quantile(PERCENTILE / 100)
        .reset_index(name="H_d")
    )
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print("\nH_d (95th percentile of normal runs' max |d|, excluding sigma-fragile positions), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack = dscore_df[
        dscore_df["mode"].isin(["jump", "drift"]) & (~dscore_df["sigma_fragile"])
    ].merge(thresholds, on="counter")
    attack["flagged"] = attack["d"].abs() > attack["H_d"]
    attack["post_onset_flag"] = attack["flagged"] & (attack["window_end_iter"] >= ONSET_ITER)

    run_detected = attack.groupby(["counter", "mode", "seed"])["post_onset_flag"].any().reset_index(name="detected")
    summary = run_detected.groupby(["counter", "mode"])["detected"].agg(n_detected="sum", n_runs="count").reset_index()
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print(f"\nPer-run detection summary (D_final, W={W}, post-onset-only):")
    print(summary.to_string(index=False))
    print(f"Saved {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
