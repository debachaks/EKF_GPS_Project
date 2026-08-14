"""Widens the window for z_stat ONLY (W_Z=20), leaving D and V exactly as
already computed at WINDOW=10 (results/combined_detection_windowed.csv).
Tests the hypothesis that z_stat = max(|z_i|) in a window is essentially
window-size-invariant for per-run detection outcome: for ANY window size
W (as long as W <= run length), the max over ALL window positions of
z_stat equals the run's global max(|z|) -- because sliding the window
across the whole run guarantees some window position contains the point
where the global peak occurs, and a window's max can never exceed the
run's global max. So M_j (used to build h_z, and to decide per-run
detection) should come out numerically identical regardless of window
size; the only thing a wider window can change is WHEN (which
window_end_iter) the statistic first crosses threshold, since a wider
window "reaches back" further and can incorporate the peak sooner in
absolute end-iteration terms.

Reports, for jump only, per counter:
    - whether the per-run z-only detection SET changes between W=10 and
      W_Z=20 (expected: no)
    - whether h_z changes (expected: no)
    - the earliest window_end_iter each detected run crosses threshold,
      W=10 vs W_Z=20 (expected: W_Z=20 fires at the same iteration or
      earlier, never later)
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")

W_Z = 20
STEP = 1
PERCENTILE = 95
COUNTERS = ["hpmcounter10", "hpmcounter8", "hpmcounter3", "hpmcounter4", "hpmcounter5"]


def windowed_z(z_values, window):
    n = len(z_values)
    results = []
    for start in range(0, n - window + 1, STEP):
        w = z_values[start:start + window]
        if np.any(np.isnan(w)):
            continue
        results.append((start, np.max(np.abs(w))))
    return results


def compute_all(ts, window):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z = group["z"].to_numpy(dtype=float)
        iters = group["iter"].to_numpy()
        for start, val in windowed_z(z, window):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[start + window - 1],
                "z_stat": val,
            })
    return pd.DataFrame(rows)


def main():
    ts = pd.read_csv(TIMESERIES_PATH)
    ts = ts[ts["counter"].isin(COUNTERS)]

    old = pd.read_csv(os.path.join(RESULTS_DIR, "combined_detection_windowed.csv"))
    old_thresh = pd.read_csv(os.path.join(RESULTS_DIR, "combined_detection_thresholds.csv")).set_index("counter")

    new = compute_all(ts, W_Z)
    normal = new[new["mode"] == "normal"]
    per_run_max = normal.groupby(["counter", "seed"])["z_stat"].max().reset_index(name="M_j")
    new_thresh = per_run_max.groupby("counter")["M_j"].quantile(PERCENTILE / 100)

    print(f"=== h_z: WINDOW=10 (existing) vs W_Z={W_Z} (new) ===")
    for c in COUNTERS:
        print(f"{c:14s}  h_z(W=10)={old_thresh.loc[c,'h_z']:.4f}   h_z(W_Z={W_Z})={new_thresh.loc[c]:.4f}")

    print(f"\n=== jump: per-run z-only detection set, W=10 vs W_Z={W_Z} ===")
    for c in COUNTERS:
        h10 = old_thresh.loc[c, "h_z"]
        h20 = new_thresh.loc[c]

        old_sub = old[(old["counter"] == c) & (old["mode"] == "jump")]
        fired_10 = old_sub[old_sub["z_stat"] > h10].groupby("seed")["window_end_iter"].min()

        new_sub = new[(new["counter"] == c) & (new["mode"] == "jump")]
        fired_20 = new_sub[new_sub["z_stat"] > h20].groupby("seed")["window_end_iter"].min()

        set10, set20 = set(fired_10.index), set(fired_20.index)
        same_set = set10 == set20
        print(f"\n{c}: same detection set = {same_set}  (W=10: {len(set10)}/20, W_Z={W_Z}: {len(set20)}/20)")
        if set10 != set20:
            print(f"  only W=10: {sorted(set10-set20)}   only W_Z={W_Z}: {sorted(set20-set10)}")

        common = sorted(set10 & set20, key=lambda s: int(s.replace("seed", "")))
        if common:
            print("  first-detection iteration (W=10 -> W_Z=20), for seeds caught by both:")
            for s in common:
                i10, i20 = fired_10[s], fired_20[s]
                arrow = "earlier" if i20 < i10 else ("later" if i20 > i10 else "same")
                print(f"    {s}: {i10} -> {i20}  ({arrow})")


if __name__ == "__main__":
    main()
