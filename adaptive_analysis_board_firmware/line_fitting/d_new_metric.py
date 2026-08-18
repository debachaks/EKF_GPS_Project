"""New D metric per the whiteboard design (item 3c): a window-diff on top
of the no-denominator D_w from d_w_metric.py (item 3b), then z-scored
against its own normal-mode baseline -- same overall shape as the G/g
window-diff metric (window_diff_metric.py), applied to D_w instead of z.

    D_new_w(t) = |D_w(t) - D_w(t-1)|

        D_w(t)   = |beta_w(t) - center_w,beta(t)|   (the 3b metric, no
                    denominator -- taking a DIFFERENCE here instead of a
                    ratio is the whole point: it avoids dividing by
                    something super small, the failure mode that broke
                    the original sigma/floor D_w)
        center(t) = mean or median of beta_w(t) across the 20 normal
                    trials at that window position -- tried both ways,
                    same as 3b

    d(t) = (D_new_w(t) - mu_Dnew(t)) / sigma_Dnew(t)

        mu_Dnew(t)    = mean of D_new_w(t) across the 20 normal trials,
                        at that window position
        sigma_Dnew(t) = std of D_new_w(t) across the 20 normal trials,
                        at that window position

Sigma fragility guard (same lesson as window_diff_metric.py / the
hpmcounter9 D_w "broken" investigation): positions where sigma_Dnew(t) is
near zero are flagged and excluded from thresholding, instead of being
clamped to a tiny epsilon that would blow up d(t).

Threshold: per-trial-max-then-95th-percentile of |d(t)|, excluding
sigma-fragile positions -- same convention as G/g and D/V.
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

BETA_OUT_PATH = os.path.join(RESULTS_DIR, "d_new_beta.csv")
DNEW_OUT_PATH = os.path.join(RESULTS_DIR, "d_new_dneww.csv")
BASELINE_OUT_PATH = os.path.join(RESULTS_DIR, "d_new_baseline.csv")
DSCORE_OUT_PATH = os.path.join(RESULTS_DIR, "d_new_dscore.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, "d_new_thresholds.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, "d_new_detection_summary.csv")

_S = np.arange(W, dtype=float)
_S_BAR = _S.mean()
_S_DEV = _S - _S_BAR
_SS_S = np.sum(_S_DEV**2)


def windowed_beta(raw_values):
    n = len(raw_values)
    results = []
    for start in range(0, n - W + 1, STEP):
        w = raw_values[start:start + W]
        if np.any(np.isnan(w)):
            continue
        beta = np.sum(_S_DEV * w) / _SS_S
        results.append((start, beta))
    return results


def compute_all_beta(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        raw = group["raw_value"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, beta in windowed_beta(raw):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + W - 1], "beta": beta,
            })
    return pd.DataFrame(rows)


def main():
    ts = pd.read_csv(TIMESERIES_PATH)
    counters = sorted(ts["counter"].unique(), key=lambda c: int(c.replace("hpmcounter", "")))

    beta_df = compute_all_beta(ts)
    beta_df.to_csv(BETA_OUT_PATH, index=False)
    print(f"Saved {BETA_OUT_PATH} ({len(beta_df)} rows)")

    dnew_rows = []
    for counter in counters:
        c_data = beta_df[beta_df["counter"] == counter]
        normal_pivot = c_data[c_data["mode"] == "normal"].pivot(
            index="seed", columns="window_end_iter", values="beta"
        )
        centers = {
            "mean": normal_pivot.mean(axis=0),
            "median": normal_pivot.median(axis=0),
        }

        for cname, cvals in centers.items():
            for (mode, seed), grp in c_data.groupby(["mode", "seed"]):
                grp = grp.set_index("window_end_iter").reindex(normal_pivot.columns)
                d_w = (grp["beta"] - cvals).abs()
                d_new = d_w.diff().abs()
                for it, val in d_new.items():
                    if pd.isna(val):
                        continue
                    dnew_rows.append({
                        "counter": counter, "center": cname, "mode": mode, "seed": seed,
                        "window_end_iter": it, "D_new_w": val,
                    })

    dnew_df = pd.DataFrame(dnew_rows)
    dnew_df.to_csv(DNEW_OUT_PATH, index=False)
    print(f"Saved {DNEW_OUT_PATH} ({len(dnew_df)} rows)")

    baseline_rows, dscore_rows = [], []
    sigma_warnings = []
    for counter in counters:
        for cname in ["mean", "median"]:
            c_data = dnew_df[(dnew_df["counter"] == counter) & (dnew_df["center"] == cname)]
            normal_pivot = c_data[c_data["mode"] == "normal"].pivot(
                index="seed", columns="window_end_iter", values="D_new_w"
            )
            mu = normal_pivot.mean(axis=0)
            sigma = normal_pivot.std(axis=0)

            n_fragile = int((sigma < SIGMA_FLOOR).sum())
            if n_fragile > 0:
                sigma_warnings.append((counter, cname, n_fragile, len(sigma)))
            sigma_safe = sigma.where(sigma > SIGMA_FLOOR, SIGMA_FLOOR)

            for it in mu.index:
                baseline_rows.append({
                    "counter": counter, "center": cname, "window_end_iter": it,
                    "mu_Dnew": mu[it], "sigma_Dnew": sigma_safe[it],
                    "sigma_fragile": bool(sigma[it] < SIGMA_FLOOR),
                })

            for (mode, seed), grp in c_data.groupby(["mode", "seed"]):
                grp = grp.set_index("window_end_iter").reindex(mu.index)
                d_score = (grp["D_new_w"] - mu) / sigma_safe
                for it, val in d_score.items():
                    dscore_rows.append({
                        "counter": counter, "center": cname, "mode": mode, "seed": seed,
                        "window_end_iter": it, "d": val,
                        "sigma_fragile": bool(sigma[it] < SIGMA_FLOOR),
                    })

    print("\n=== sigma_Dnew fragility check (positions with sigma_Dnew < 1e-6) ===")
    if sigma_warnings:
        for counter, cname, n_fragile, n_total in sigma_warnings:
            print(f"  {counter} ({cname}): {n_fragile}/{n_total} fragile positions")
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
    per_run_max = normal_d.groupby(["counter", "center", "seed"])["abs_d"].max().reset_index(name="M_j")
    thresholds = (
        per_run_max.groupby(["counter", "center"])["M_j"]
        .quantile(PERCENTILE / 100)
        .reset_index(name="H_d")
    )
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print("\nH_d (95th percentile of normal runs' max |d|, excluding sigma-fragile positions), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack = dscore_df[
        dscore_df["mode"].isin(["jump", "drift"]) & (~dscore_df["sigma_fragile"])
    ].merge(thresholds, on=["counter", "center"])
    attack["flagged"] = attack["d"].abs() > attack["H_d"]

    run_detected = attack.groupby(["counter", "center", "mode", "seed"])["flagged"].any().reset_index(name="detected")
    summary = run_detected.groupby(["counter", "center", "mode"])["detected"].agg(n_detected="sum", n_runs="count").reset_index()
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print(f"\nPer-run detection summary (d metric, W={W}):")
    print(summary.to_string(index=False))
    print(f"Saved {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
