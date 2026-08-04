"""Per-seed, per-counter raw value traces: for every (counter, seed) pair,
one plot showing that seed's drift/jump/replay raw traces together with
the NORMAL baseline - mean +/- 1 std across all 20 normal seeds (not just
that seed's own normal run) - as a reference band.

Reuses the same raw-trace decoding and shared elapsed-time grid approach
as line_fitting_analysis.py (hex_to_int decode, np.interp onto a common
grid so every trace lines up on the same time axis).
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
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots", "seed_counter_traces")
GRID_POINTS = 300
ATTACK_MODES = ["drift", "jump", "replay"]
MODE_COLOR = {"drift": "#1baf7a", "jump": "#eda100", "replay": "#c0392b"}
NORMAL_COLOR = "#2a78d6"

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
        for mode in ["normal"] + ATTACK_MODES:
            path = os.path.join(CLEAN_ROOT, seed_name, f"ekf_{mode}_hpc.csv")
            cols = frozenset(c for c in pd.read_csv(path, nrows=1).columns if c.startswith("hpmcounter"))
            common = cols if common is None else (common & cols)
    return sorted(common, key=lambda c: int(c.replace("hpmcounter", "")))


def raw_trace(seed_name, mode, counter):
    path = os.path.join(CLEAN_ROOT, seed_name, f"ekf_{mode}_hpc.csv")
    df = pd.read_csv(path)
    ts = df["timestamp_ms"].map(hex_to_int)
    elapsed = ts - ts.iloc[0]
    values = df[counter].map(hex_to_int)
    return elapsed.to_numpy(), values.to_numpy()


def build_global_grid(seed_names, counters):
    max_elapsed = min(
        raw_trace(seed_name, mode, counter)[0].max()
        for seed_name in seed_names
        for mode in ["normal"] + ATTACK_MODES
        for counter in counters
    )
    return np.linspace(0, max_elapsed, GRID_POINTS)


def normal_baseline(seed_names, counter, grid):
    traces = np.array([np.interp(grid, *raw_trace(s, "normal", counter)) for s in seed_names])
    return traces.mean(axis=0), traces.std(axis=0)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_MUTED)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)


def main():
    seed_names = find_seed_names()
    counters = find_counters(seed_names)
    grid = build_global_grid(seed_names, counters)
    print(f"{len(seed_names)} seeds, {len(counters)} counters, grid up to {grid[-1]:.0f} ms")

    for counter in counters:
        mu_t, sigma_t = normal_baseline(seed_names, counter, grid)
        counter_dir = os.path.join(PLOT_DIR, counter)
        os.makedirs(counter_dir, exist_ok=True)

        for seed_name in seed_names:
            fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)

            ax.plot(grid, mu_t, color=NORMAL_COLOR, linewidth=2, label="normal mean (n=20 seeds)")
            ax.fill_between(grid, mu_t - sigma_t, mu_t + sigma_t, color=NORMAL_COLOR, alpha=0.15,
                             linewidth=0, label="normal mean +/- 1 std")

            for mode in ATTACK_MODES:
                elapsed, values = raw_trace(seed_name, mode, counter)
                x_t = np.interp(grid, elapsed, values)
                ax.plot(grid, x_t, color=MODE_COLOR[mode], linewidth=1.5, label=mode)

            ax.set_title(f"{seed_name}: {counter} raw value over time", color=TEXT_PRIMARY, fontsize=12)
            ax.set_xlabel("elapsed time (ms since run start)", color=TEXT_MUTED)
            ax.set_ylabel(counter, color=TEXT_MUTED)
            ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
            style_axes(ax)
            fig.tight_layout()

            out_path = os.path.join(counter_dir, f"{seed_name}.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

        print(f"Saved {len(seed_names)} plots for {counter} -> {counter_dir}")


if __name__ == "__main__":
    main()
