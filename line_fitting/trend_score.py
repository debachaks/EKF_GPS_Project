"""Trend score D = |beta_hat / SE(beta_hat)| for the windowed-z line fit.

    SE(beta_hat)   = sqrt( sigma_eps^2 / sum((s_i - s_bar)^2) )
    sigma_eps^2    = sum(e_i^2) / (W - 2)          (W-2: alpha_hat and
                                                     beta_hat each use up
                                                     one degree of freedom)
    e_i            = z_i - z_hat_i                 (residual, from the fit
                                                     to the windowed/rolling z)
    D              = |beta_hat / SE(beta_hat)|

This is the standard t-statistic for a simple linear regression slope -
D asks "is beta_hat large relative to how uncertain the fit itself is",
not just "is beta_hat large in absolute terms".

Reads line_fitting_timeseries.csv (already has residual, sample_index per
row from line_fitting_analysis.py's fit to the windowed z) and
line_fitting_summary.csv (already has beta per run). Computed for every
(counter, mode, seed) EXCEPT normal - normal's own reference/baseline
handling is separate (per instructions, to be added next).
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "line_fitting_timeseries.csv")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "line_fitting_summary.csv")
OUT_PATH = os.path.join(RESULTS_DIR, "trend_score_results.csv")


def compute_d_for_group(group):
    valid = group.dropna(subset=["residual"])
    w = len(valid)
    if w <= 2:
        return pd.Series({"W": w, "sigma_eps2": np.nan, "SE_beta": np.nan, "D": np.nan})

    s = valid["sample_index"].to_numpy(dtype=float)
    e = valid["residual"].to_numpy(dtype=float)

    sigma_eps2 = np.sum(e**2) / (w - 2)
    s_bar = s.mean()
    ss_s = np.sum((s - s_bar) ** 2)
    se_beta = np.sqrt(sigma_eps2 / ss_s)

    return pd.Series({"W": w, "sigma_eps2": sigma_eps2, "SE_beta": se_beta})


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = pd.read_csv(TIMESERIES_PATH)
    summary = pd.read_csv(SUMMARY_PATH)

    ts = ts[ts["mode"] != "normal"]
    summary = summary[summary["mode"] != "normal"]

    stats = ts.groupby(["counter", "mode", "seed"]).apply(compute_d_for_group).reset_index()

    result = summary.merge(stats, on=["counter", "mode", "seed"])
    result["D"] = (result["beta"] / result["SE_beta"]).abs()

    result = result[["counter", "mode", "seed", "alpha", "beta", "W", "sigma_eps2", "SE_beta", "D"]]
    result = result.sort_values(["counter", "mode", "D"], ascending=[True, True, False])
    result.to_csv(OUT_PATH, index=False)

    print(result.to_string(index=False))
    print(f"\nSaved {OUT_PATH}")

    print("\nMean D by counter and mode (across 20 seeds):")
    print(result.groupby(["counter", "mode"])["D"].agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
