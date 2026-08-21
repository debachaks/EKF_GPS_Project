"""Window-size sensitivity check for D(t,W) and V(t,W).

Reuses the raw per-sample z(t) already saved in line_fitting_timeseries.csv
(the "z" column, computed once in line_fitting_analysis.py and independent
of window size) rather than re-running the whole raw-data pipeline. For
each candidate window size W in WINDOW_SIZES, this:

    1. Recomputes rolling_z_wW = W-sample rolling mean of z(t), per
       (counter, mode, seed) - same idea as rolling_z_w10 in
       line_fitting_analysis.py, just with a different W.
    2. Recomputes D(t,W) (line-fit slope t-stat) and V(t,W) = log(var+1)
       over sliding windows of that same size W, step 1, fit/measured on
       rolling_z_wW - same formulas as trend_score_windowed.py and
       variability_metric.py, just parametrized by W instead of hardcoded
       to 10.
    3. Builds h_D / h_V per counter from the 20 normal trials (95th
       percentile of each trial's own max D or V).
    4. Flags every attack-mode (jump/drift/replay) window exceeding its
       counter's threshold and counts flags.

This is exploratory (window sizes are not committed as new canonical
outputs) - printed to stdout only, not saved to CSV.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "line_fitting_timeseries.csv")

WINDOW_SIZES = [5, 10, 20]
STEP = 1
PERCENTILE = 95
ATTACK_MODES = ["drift", "jump", "replay"]


def rolling_z(z, window):
    return pd.Series(z).rolling(window).mean().to_numpy()


def sliding_d_and_v(rz, window):
    """Returns list of (window_start, D, V) for every full, NaN-free
    window of the given size."""
    s = np.arange(window, dtype=float)
    s_bar = s.mean()
    s_dev = s - s_bar
    ss_s = np.sum(s_dev**2)

    n = len(rz)
    results = []
    for start in range(0, n - window + 1, STEP):
        w = rz[start:start + window]
        if np.any(np.isnan(w)):
            continue

        beta = np.sum(s_dev * w) / ss_s
        alpha = w.mean() - beta * s_bar
        fitted = alpha + beta * s
        resid = w - fitted
        sigma_eps2 = np.sum(resid**2) / (window - 2)
        se_beta = np.sqrt(sigma_eps2 / ss_s)
        d = abs(beta / se_beta) if se_beta > 0 else np.nan

        variance = np.var(w, ddof=1)
        v = np.log1p(variance)

        results.append((start, d, v))
    return results


def compute_for_window(ts, window):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("sample_index")
        z = group["z"].to_numpy(dtype=float)
        rz = rolling_z(z, window)
        for start, d, v in sliding_d_and_v(rz, window):
            rows.append({"counter": counter, "mode": mode, "seed": seed, "D": d, "V": v})
    return pd.DataFrame(rows)


def threshold_and_flag(df, metric):
    normal = df[df["mode"] == "normal"]
    per_trial_max = normal.groupby(["counter", "seed"])[metric].max().reset_index(name="M_j")
    thresholds = per_trial_max.groupby("counter")["M_j"].quantile(PERCENTILE / 100).reset_index(name="h")

    attack = df[df["mode"].isin(ATTACK_MODES)].merge(thresholds, on="counter")
    attack["flagged"] = attack[metric] > attack["h"]
    return attack


def main():
    ts = pd.read_csv(TIMESERIES_PATH)

    for window in WINDOW_SIZES:
        print(f"\n{'='*60}\nWindow size W = {window}\n{'='*60}")
        df = compute_for_window(ts, window)

        for metric in ["D", "V"]:
            attack = threshold_and_flag(df, metric)
            total = len(attack)
            flagged = int(attack["flagged"].sum())
            print(f"\n{metric}: {flagged} / {total} windows flagged ({100*flagged/total:.3f}%)")
            by_counter = attack.groupby(["counter", "mode"])["flagged"].agg(n_flagged="sum", n_windows="count")
            nonzero = by_counter[by_counter["n_flagged"] > 0]
            if len(nonzero):
                print(nonzero.to_string())
            else:
                print("  (no windows flagged for any counter/mode)")


if __name__ == "__main__":
    main()
