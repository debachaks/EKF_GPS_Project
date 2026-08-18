"""New metric G / g, per the whiteboard design: a two-adjacent-window
contrast, standardized against its own per-position normal-mode baseline
-- conceptually a simpler alternative to D (which fits a full OLS line)
that just asks "did the mean level step up between this window and the
one right before it."

    G_L(k) = mean(z_i for i in [k-L+1, k])          <- current window, length L
             - mean(z_i for i in [k-2L+1, k-L])     <- immediately preceding,
                                                          NON-overlapping window, length L

For every run (all 20 normal + jump/drift), G_L(k) is computed at every
valid k (k >= 2L-1, so both windows fit inside the run).

From the 20 normal runs, at EACH position k:
    mu_G0(k)    = mean of G_L(k) across the 20 normal trials
    sigma_G0(k) = std  of G_L(k) across the 20 normal trials

    g(k) = (G_L(k) - mu_G0(k)) / sigma_G0(k)

Threshold Qgs per counter = 95th percentile of {max over k of |g(k)|, per
normal trial} -- same per-trial-max-then-percentile calibration used for
D/V (not the flat-pooled method used for the current z metric).

Detection rule: |g(k)| > Qgs for any k => flagged.

Sanity-checks sigma_G0 for near-zero collapse before trusting results --
see the earlier hpmcounter9/V lesson (division by a near-zero baseline
std can fabricate huge, meaningless g values).

Detection: a run counts as detected only if a flag occurs at or after the
attack onset (iter >= 150, per seed_old/test_seed_pre_post_attack_analysis.py)
-- pre-onset flags are excluded so the rate isn't inflated by false fires
that happen before the attack is even injected (same correction applied
to D_final/V_final after finding real pre-onset contamination there).
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")

L = 10
STEP = 1
PERCENTILE = 95
SIGMA_FLOOR = 1e-6   # below this, sigma_G0 is treated as too fragile to trust
ONSET_ITER = 150

G_OUT_PATH = os.path.join(RESULTS_DIR, f"window_diff_G_L{L}.csv")
BASELINE_OUT_PATH = os.path.join(RESULTS_DIR, f"window_diff_baseline_L{L}.csv")
G_ZSCORE_OUT_PATH = os.path.join(RESULTS_DIR, f"window_diff_g_L{L}.csv")
THRESHOLD_OUT_PATH = os.path.join(RESULTS_DIR, f"window_diff_threshold_L{L}.csv")
FLAGGED_OUT_PATH = os.path.join(RESULTS_DIR, f"window_diff_attack_flags_L{L}.csv")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, f"window_diff_detection_summary_L{L}.csv")


def windowed_G(z_values, L):
    """Returns (k, G) for every k where both the current and preceding
    L-sample windows fit inside the run (k is the 0-indexed array
    position of the last sample in the current window)."""
    n = len(z_values)
    results = []
    for k in range(2 * L - 1, n, STEP):
        current = z_values[k - L + 1: k + 1]
        prev = z_values[k - 2 * L + 1: k - L + 1]
        if np.any(np.isnan(current)) or np.any(np.isnan(prev)):
            continue
        results.append((k, current.mean() - prev.mean()))
    return results


def compute_all_G(ts):
    rows = []
    for (counter, mode, seed), group in ts.groupby(["counter", "mode", "seed"]):
        group = group.sort_values("iter")
        z = group["z"].to_numpy()
        iters = group["iter"].to_numpy()
        for k, G in windowed_G(z, L):
            rows.append({
                "counter": counter, "mode": mode, "seed": seed,
                "window_end_iter": iters[k], "G": G,
            })
    return pd.DataFrame(rows)


def main():
    ts = pd.read_csv(TIMESERIES_PATH)
    counters = sorted(ts["counter"].unique(), key=lambda c: int(c.replace("hpmcounter", "")))

    g_df = compute_all_G(ts)
    g_df.to_csv(G_OUT_PATH, index=False)
    print(f"Saved {G_OUT_PATH} ({len(g_df)} rows)")

    baseline_rows, gscore_rows = [], []
    sigma_warnings = []
    for counter in counters:
        c_data = g_df[g_df["counter"] == counter]
        normal_pivot = c_data[c_data["mode"] == "normal"].pivot(
            index="seed", columns="window_end_iter", values="G"
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
                "mu_G0": mu[it], "sigma_G0": sigma_safe[it],
                "sigma_fragile": bool(sigma[it] < SIGMA_FLOOR),
            })

        for (mode, seed), grp in c_data.groupby(["mode", "seed"]):
            grp = grp.set_index("window_end_iter").reindex(mu.index)
            g = (grp["G"] - mu) / sigma_safe
            for it, val in g.items():
                gscore_rows.append({
                    "counter": counter, "mode": mode, "seed": seed,
                    "window_end_iter": it, "g": val,
                    "sigma_fragile": bool(sigma[it] < SIGMA_FLOOR),
                })

    print("\n=== sigma_G0 fragility check (positions with sigma_G0 < 1e-6) ===")
    if sigma_warnings:
        for counter, n_fragile, n_total in sigma_warnings:
            print(f"  {counter}: {n_fragile}/{n_total} fragile positions")
    else:
        print("  none -- sigma_G0 is well-behaved everywhere")

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(BASELINE_OUT_PATH, index=False)
    print(f"\nSaved {BASELINE_OUT_PATH}")

    gscore_df = pd.DataFrame(gscore_rows)
    gscore_df.to_csv(G_ZSCORE_OUT_PATH, index=False)
    print(f"Saved {G_ZSCORE_OUT_PATH} ({len(gscore_df)} rows)")

    # threshold Qgs: 95th percentile of {max |g| per normal trial}, per counter
    # -- computed excluding any window position flagged as sigma-fragile
    normal_g = gscore_df[(gscore_df["mode"] == "normal") & (~gscore_df["sigma_fragile"])].copy()
    normal_g["abs_g"] = normal_g["g"].abs()
    per_run_max = normal_g.groupby(["counter", "seed"])["abs_g"].max().reset_index(name="M_j")
    thresholds = (
        per_run_max.groupby("counter")["M_j"]
        .quantile(PERCENTILE / 100)
        .reset_index(name="Qgs")
    )
    thresholds.to_csv(THRESHOLD_OUT_PATH, index=False)
    print("\nQgs (95th percentile of normal runs' max |g|, excluding sigma-fragile positions), per counter:")
    print(thresholds.to_string(index=False))
    print(f"Saved {THRESHOLD_OUT_PATH}")

    # flag attack runs, also excluding sigma-fragile positions
    attack = gscore_df[
        gscore_df["mode"].isin(["jump", "drift"]) & (~gscore_df["sigma_fragile"])
    ].merge(thresholds, on="counter")
    attack["flagged"] = attack["g"].abs() > attack["Qgs"]
    attack["post_onset_flag"] = attack["flagged"] & (attack["window_end_iter"] >= ONSET_ITER)
    attack.to_csv(FLAGGED_OUT_PATH, index=False)
    print(f"\nSaved {FLAGGED_OUT_PATH} ({len(attack)} rows)")

    run_detected = attack.groupby(["counter", "mode", "seed"])["post_onset_flag"].any().reset_index(name="detected")
    summary = run_detected.groupby(["counter", "mode"])["detected"].agg(n_detected="sum", n_runs="count").reset_index()
    summary.to_csv(SUMMARY_OUT_PATH, index=False)
    print(f"\nPer-run detection summary (window-diff g metric, L={L}, post-onset-only):")
    print(summary.to_string(index=False))
    print(f"Saved {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
