"""Builds the normal-mode baseline mu(iter)/sigma(iter) per counter from the
newMapping dataset (adaptive_analysis_board_firmware/plots/seed*_newMapping/),
then standardizes every run (normal + any attack modes present) against it.

Unlike the original line_fitting/line_fitting_analysis.py (which had to
interpolate onto a shared elapsed-time grid, since raw register-dump samples
land at slightly different times per run), these captures are already
exactly one row per EKF iteration, 0..299, identical across every run -- so
iter itself IS the shared grid, no interpolation needed.

Baseline is built from the RAW counter value (not rate), same choice as the
original line_fitting pipeline -- the windowed OLS slope in
trend_score_windowed.py captures rate-of-change information from the level
series itself, no need to difference first.

Discovers seeds/modes dynamically: any seed*_newMapping folder becomes a
seed, and the mode is read off each CSV's filename prefix (normal_hpc*.csv
-> "normal", jump_hpc*.csv -> "jump", etc), so newly-added attack-mode seeds
are picked up automatically on a rerun -- nothing here is hardcoded to
seed2/seed3.

Outputs:
    results/baseline_mu_sigma.csv   - mu, sigma per (counter, iter)
    results/zscore_timeseries.csv   - long format: counter, mode, seed,
                                       iter, raw_value, z
"""

import glob
import os
import re

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
PLOTS_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), "plots")
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")

COUNTERS = [f"hpmcounter{i}" for i in range(3, 11)]
MODE_RE = re.compile(r"^([a-zA-Z]+)_hpc")


def hex_to_int(val):
    s = str(val).strip()
    if s.lower().startswith("0x"):
        return int(s, 16)
    return int(float(s))


def discover_runs():
    """Returns {(mode, seed): csv_path} for every seed*_newMapping/*_hpc*.csv."""
    runs = {}
    for seed_dir in sorted(glob.glob(os.path.join(PLOTS_ROOT, "seed*_newMapping"))):
        seed = os.path.basename(seed_dir).replace("_newMapping", "")
        for csv_path in sorted(glob.glob(os.path.join(seed_dir, "*_hpc*.csv"))):
            m = MODE_RE.match(os.path.basename(csv_path))
            if not m:
                continue
            mode = m.group(1).lower()
            runs[(mode, seed)] = csv_path
    return runs


def load_trace(csv_path):
    df = pd.read_csv(csv_path)
    df["iter"] = df["iter"].map(hex_to_int)
    for c in COUNTERS:
        df[c] = df[c].map(hex_to_int)
    return df.set_index("iter")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    runs = discover_runs()
    if not runs:
        print(f"No seed*_newMapping runs found under {PLOTS_ROOT}")
        return

    normal_runs = {seed: path for (mode, seed), path in runs.items() if mode == "normal"}
    attack_modes = sorted({mode for mode, _ in runs if mode != "normal"})
    print(f"Found {len(runs)} runs across {len(set(s for _, s in runs))} seeds")
    print(f"Normal baseline built from {len(normal_runs)} seeds: {sorted(normal_runs)}")
    print(f"Attack modes present: {attack_modes or '(none yet)'}")

    traces = {key: load_trace(path) for key, path in runs.items()}
    n_iters = min(len(df) for df in traces.values())

    mu_rows, z_rows = [], []
    for counter in COUNTERS:
        normal_matrix = np.array([
            traces[("normal", seed)][counter].to_numpy()[:n_iters]
            for seed in normal_runs
        ])
        mu = normal_matrix.mean(axis=0)
        sigma = normal_matrix.std(axis=0)
        sigma_safe = np.where(sigma < 1e-9, 1e-9, sigma)

        for it in range(n_iters):
            mu_rows.append({"counter": counter, "iter": it, "mu": mu[it], "sigma": sigma_safe[it]})

        for (mode, seed), df in traces.items():
            x = df[counter].to_numpy()[:n_iters]
            z = (x - mu) / sigma_safe
            for it in range(n_iters):
                z_rows.append({
                    "counter": counter, "mode": mode, "seed": seed, "iter": it,
                    "raw_value": x[it], "z": z[it],
                })

    mu_df = pd.DataFrame(mu_rows)
    mu_df.to_csv(os.path.join(RESULTS_DIR, "baseline_mu_sigma.csv"), index=False)
    print(f"Saved {os.path.join(RESULTS_DIR, 'baseline_mu_sigma.csv')}")

    z_df = pd.DataFrame(z_rows)
    z_df.to_csv(os.path.join(RESULTS_DIR, "zscore_timeseries.csv"), index=False)
    print(f"Saved {os.path.join(RESULTS_DIR, 'zscore_timeseries.csv')} ({len(z_df)} rows)")


if __name__ == "__main__":
    main()
