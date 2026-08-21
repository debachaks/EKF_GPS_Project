"""Same as trend_score_windowed.py, but with WINDOW=20 instead of 10 --
a comparison run to see whether a wider fitting window changes which
seeds/counters trip the trend score D, since a bigger window means more
degrees of freedom for the OLS fit (smoother/less noisy slope estimate,
but also more averaging-out of short bursts). Reads the same
zscore_timeseries.csv (raw z-score doesn't depend on window size at all,
that's purely a per-iteration baseline calculation) -- only the windowing
step here changes. Outputs go to separate _w20-suffixed files so the
original WINDOW=10 results aren't touched.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")
D_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_windowed_results_w20.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_threshold_by_counter_w20.csv")
FLAGGED_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_attack_flags_w20.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_detection_summary_w20.csv")

WINDOW = 20
STEP = 1
PERCENTILE = 95
EPS = 1e-20   # see trend_score_windowed.py -- avoids 0/0 NaN on flat windows

_S = np.arange(WINDOW, dtype=float)
_S_BAR = _S.mean()
_S_DEV = _S - _S_BAR
_SS_S = np.sum(_S_DEV**2)


def windowed_d(z_values):
    n = len(z_values)
    results = []
    for start in range(0, n - WINDOW + 1, STEP):
        window = z_values[start:start + WINDOW]
        if np.any(np.isnan(window)):
            continue
        beta = np.sum(_S_DEV * window) / _SS_S
        alpha = window.mean() - beta * _S_BAR
        fitted = alpha + beta * _S
        resid = window - fitted
        sigma_eps2 = np.sum(resid**2) / (WINDOW - 2)
        se_beta = np.sqrt(sigma_eps2 / _SS_S + EPS)
        d = abs(beta / se_beta)
        results.append((start, beta, d))
    return results


def compute_all_windows(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z_values = group["z"].to_numpy()
        iters = group["iter"].to_numpy()
        for start, beta, d in windowed_d(z_values):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + WINDOW - 1],
                "beta_local": beta, "D": d,
            })
    return pd.DataFrame(rows)


def build_thresholds(d_df):
    normal = d_df[d_df["mode"] == "normal"]
    per_run_max = normal.groupby(["counter", "seed"])["D"].max().reset_index(name="M_j")
    thresholds = (
        per_run_max.groupby("counter")["M_j"]
        .quantile(PERCENTILE / 100)
        .reset_index(name="h_D")
    )
    return thresholds


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = pd.read_csv(TIMESERIES_PATH)

    d_df = compute_all_windows(ts)
    d_df.to_csv(D_OUT_PATH, index=False)
    print(f"Saved {D_OUT_PATH} ({len(d_df)} rows)")

    thresholds = build_thresholds(d_df)
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print("\nh_D (W=20) (95th percentile of normal runs' max D), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    attack_modes = sorted(ts.loc[ts["mode"] != "normal", "mode"].unique())
    if not attack_modes:
        print("\nNo attack-mode runs present yet -- skipping flagging step.")
        return

    attack = d_df[d_df["mode"].isin(attack_modes)].merge(thresholds, on="counter")
    attack["flagged"] = attack["D"] > attack["h_D"]
    attack.to_csv(FLAGGED_OUT_PATH, index=False)
    print(f"\nSaved {FLAGGED_OUT_PATH} ({len(attack)} rows)")

    run_detected = attack.groupby(["counter", "mode", "seed"])["flagged"].any().reset_index(name="detected")
    summary = run_detected.groupby(["counter", "mode"])["detected"].agg(n_detected="sum", n_runs="count").reset_index()
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print("\nPer-run detection summary (W=20) (a run counts as detected if ANY window exceeds h_D):")
    print(summary.to_string(index=False))
    print(f"Saved {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
