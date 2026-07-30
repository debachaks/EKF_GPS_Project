"""Z-score line-fitting analysis.

For each counter, builds the normal baseline mu(t)/sigma(t) - the mean and
std of the raw counter value across all 20 normal runs, at each point on a
shared elapsed-time grid (same technique as
seed_old/test_seed_mode_comparison_mean_std.py's normal-mode curve, reused
here rather than the pairwise-DIFFERENCE script, since this needs the
actual baseline level/spread, not differences between normal run pairs).

For every individual run (20 seeds x 4 modes = 80 runs), that run's own
raw trace is interpolated onto the SAME shared grid to get x(t), then:

    z(t)          = (x(t) - mu(t)) / sigma(t)
    rolling_z(t)  = 10-sample rolling mean of z(t)              (first 9 points are NaN)
    rolling_z(t) ~ alpha + beta * i    (OLS line fit on the WINDOWED series; i = sample index 0..N-1)
    residual(t)   = rolling_z(t) - (alpha + beta*i)

The line is fit to the windowed/smoothed z, not the raw point-wise z, so
alpha/beta/residual describe the trend of the smoothed series - less
sensitive to sample-to-sample noise than fitting the raw z directly. The
raw z(t) is still kept in the output for reference alongside rolling_z.

Outputs:
    line_fitting_summary.csv    - one row per (counter, mode, seed) with
                                   alpha, beta, residual mean/std.
    line_fitting_timeseries.csv - long-format, one row per
                                   (counter, mode, seed, sample_index) with
                                   z, fitted, residual, rolling_z_w10.
    beta_by_counter_mode.csv    - mean/std of beta, grouped by counter and
                                   mode, across the 20 seeds - the number
                                   to look at first for "does the z-score
                                   trend differently under attack".
    plots/<counter>_z_by_mode.png - mean z(t) trace per mode, all 4 modes
                                     on one figure, per counter.
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
while not os.path.isdir(os.path.join(PROJECT_ROOT, "original_pipeline")):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from data_preprocessing import hex_to_int  # noqa: E402

CLEAN_ROOT = os.path.join(PROJECT_ROOT, "seed_old", "CLEAN_HPC_TEST_SEED")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
GRID_POINTS = 300
WINDOW = 10

MODES = ["normal", "drift", "jump", "replay"]
MODE_COLOR = {
    "normal": "#2a78d6",
    "drift": "#1baf7a",
    "jump": "#eda100",
    "replay": "#c0392b",
}
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"


def find_seed_names():
    dirs = sorted(d for d in glob.glob(os.path.join(CLEAN_ROOT, "test_seed_[0-9]*")) if os.path.isdir(d))
    return [os.path.basename(d) for d in dirs]


def find_counters(seed_names):
    common = None
    for seed_name in seed_names:
        for mode in MODES:
            path = os.path.join(CLEAN_ROOT, seed_name, f"ekf_{mode}_hpc.csv")
            cols = frozenset(c for c in pd.read_csv(path, nrows=1).columns if c.startswith("hpmcounter"))
            common = cols if common is None else (common & cols)
    return sorted(common, key=lambda c: int(c.replace("hpmcounter", "")))


def raw_trace(seed_name, mode, counter):
    path = os.path.join(CLEAN_ROOT, seed_name, f"ekf_{mode}_hpc.csv")
    df = pd.read_csv(path)
    elapsed = df["timestamp_ms"].map(hex_to_int) - df["timestamp_ms"].map(hex_to_int).iloc[0]
    values = df[counter].map(hex_to_int)
    return elapsed.to_numpy(), values.to_numpy()


def build_global_grid(seed_names, counters):
    """One shared elapsed-time grid, short enough to be valid (no
    extrapolation) for every run of every mode and every counter."""
    max_elapsed = min(
        raw_trace(seed_name, mode, counter)[0].max()
        for seed_name in seed_names
        for mode in MODES
        for counter in counters
    )
    return np.linspace(0, max_elapsed, GRID_POINTS)


def normal_baseline(seed_names, counter, grid):
    traces = np.array([np.interp(grid, *raw_trace(s, "normal", counter)) for s in seed_names])
    mu = traces.mean(axis=0)
    sigma = traces.std(axis=0)
    sigma_safe = np.where(sigma < 1e-9, 1e-9, sigma)
    return mu, sigma_safe


def fit_line(z):
    """Fits z ~ alpha + beta*s (s = original sample index), using only the
    finite entries of z - the first WINDOW-1 rolling_z values are NaN, so
    when this is called on the windowed series those points are skipped
    for fitting, and fitted/residual come back NaN there too rather than
    silently treated as index 0."""
    z = np.asarray(z, dtype=float)
    s = np.arange(len(z))
    valid = np.isfinite(z)
    beta, alpha = np.polyfit(s[valid], z[valid], 1)
    fitted = np.full_like(z, np.nan)
    fitted[valid] = alpha + beta * s[valid]
    residual = z - fitted
    return alpha, beta, fitted, residual


def rolling_z(z, window=WINDOW):
    return pd.Series(z).rolling(window).mean().to_numpy()


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_MUTED)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1)
    ax.axhline(0, color=AXIS_COLOR, linewidth=1)
    ax.set_axisbelow(True)


def main():
    seed_names = find_seed_names()
    if not seed_names:
        print(f"No cleaned test_seed_N folders found under {CLEAN_ROOT}")
        return
    print(f"Found {len(seed_names)} seeds")

    counters = find_counters(seed_names)
    print(f"Counters: {counters}")

    grid = build_global_grid(seed_names, counters)
    print(f"Shared grid: {GRID_POINTS} points, 0 to {grid[-1]:.0f} ms\n")

    summary_rows = []
    ts_rows = []

    for counter in counters:
        print(f"Processing {counter}...")
        mu_t, sigma_t = normal_baseline(seed_names, counter, grid)

        for mode in MODES:
            for seed_name in seed_names:
                elapsed, values = raw_trace(seed_name, mode, counter)
                x_t = np.interp(grid, elapsed, values)
                z_t = (x_t - mu_t) / sigma_t
                rz = rolling_z(z_t)
                alpha, beta, fitted, residual = fit_line(rz)

                summary_rows.append({
                    "counter": counter, "mode": mode, "seed": seed_name,
                    "alpha": alpha, "beta": beta,
                    "residual_mean": np.nanmean(residual), "residual_std": np.nanstd(residual),
                })
                for i in range(len(grid)):
                    ts_rows.append({
                        "counter": counter, "mode": mode, "seed": seed_name,
                        "sample_index": i, "elapsed_ms": grid[i],
                        "z": z_t[i], "fitted": fitted[i], "residual": residual[i],
                        "rolling_z_w10": rz[i],
                    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(SCRIPT_DIR, "line_fitting_summary.csv"), index=False)

    ts_df = pd.DataFrame(ts_rows)
    ts_df.to_csv(os.path.join(SCRIPT_DIR, "line_fitting_timeseries.csv"), index=False)

    beta_summary = summary_df.groupby(["counter", "mode"])["beta"].agg(["mean", "std"]).reset_index()
    beta_summary.to_csv(os.path.join(SCRIPT_DIR, "beta_by_counter_mode.csv"), index=False)
    print("\nbeta (z-score trend slope), mean +/- std across 20 seeds, by counter and mode:")
    print(beta_summary.to_string(index=False))

    os.makedirs(PLOT_DIR, exist_ok=True)
    for counter in counters:
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
        for mode in MODES:
            sub = ts_df[(ts_df.counter == counter) & (ts_df["mode"] == mode)]
            mean_z = sub.groupby("sample_index")["z"].mean().to_numpy()
            ax.plot(grid, mean_z, color=MODE_COLOR[mode], linewidth=2, label=f"{mode} (mean z, n={len(seed_names)})")
        ax.set_title(f"{counter}: mean z-score over time, by mode", color=TEXT_PRIMARY, fontsize=12)
        ax.set_xlabel("elapsed time (ms since run start)", color=TEXT_MUTED)
        ax.set_ylabel("z = (x - mu_normal) / sigma_normal", color=TEXT_MUTED)
        ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
        style_axes(ax)
        fig.tight_layout()
        out_path = os.path.join(PLOT_DIR, f"{counter}_z_by_mode.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

    print(f"\nSaved {os.path.join(SCRIPT_DIR, 'line_fitting_summary.csv')}")
    print(f"Saved {os.path.join(SCRIPT_DIR, 'line_fitting_timeseries.csv')}")
    print(f"Saved {os.path.join(SCRIPT_DIR, 'beta_by_counter_mode.csv')}")


if __name__ == "__main__":
    main()
