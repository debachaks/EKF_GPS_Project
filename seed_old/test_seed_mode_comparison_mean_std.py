"""Per-counter mode comparison: mean +/- 1 std across all 20 runs, for
each of normal/jump/drift/replay, all four on the same plot.

Unlike test_seed_normal_pairwise_diff_distribution.py and
test_seed_normal_attack_diff_distribution.py (which plot DIFFERENCES
between modes), this plots each mode's own raw value trajectory directly -
one mean line + std band per mode, four modes per figure, one figure per
counter. Uses raw counter value (not rate); see module docstrings
elsewhere in this folder for why raw values grow in spread over the
course of a run (accumulation), which will be visible here as widening
bands for all four modes together.

Output: test_seed_mode_comparison_plots/<counter>_mode_comparison.png
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from data_preprocessing import hex_to_int  # noqa: E402

CLEAN_ROOT = os.path.join(SRC_DIR, "CLEAN_HPC_TEST_SEED")
PLOT_DIR = os.path.join(SRC_DIR, "test_seed_mode_comparison_plots")
GRID_POINTS = 300

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


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_MUTED)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)


def mean_std_for_mode(seed_names, mode, counter, grid):
    traces = {s: raw_trace(s, mode, counter) for s in seed_names}
    interp = np.array([np.interp(grid, e, v) for s, (e, v) in traces.items()])
    return interp.mean(axis=0), interp.std(axis=0)


def plot_counter(counter, seed_names, out_dir):
    all_traces = {(s, m): raw_trace(s, m, counter) for s in seed_names for m in MODES}
    grid_end = min(elapsed.max() for elapsed, _ in all_traces.values())
    grid = np.linspace(0, grid_end, GRID_POINTS)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    for mode in MODES:
        mean_vals, std_vals = mean_std_for_mode(seed_names, mode, counter, grid)
        color = MODE_COLOR[mode]
        ax.plot(grid, mean_vals, color=color, linewidth=2, label=f"{mode} (mean, n={len(seed_names)})")
        ax.plot(grid, mean_vals + std_vals, color=color, linewidth=1, linestyle="--")
        ax.plot(grid, mean_vals - std_vals, color=color, linewidth=1, linestyle="--")

    ax.set_title(f"{counter}: mean +/- 1 std by mode (n={len(seed_names)} runs each)",
                 color=TEXT_PRIMARY, fontsize=12)
    ax.set_xlabel("elapsed time (ms since run start)", color=TEXT_MUTED)
    ax.set_ylabel("counter value", color=TEXT_MUTED)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
    style_axes(ax)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{counter}_mode_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    seed_names = find_seed_names()
    if not seed_names:
        print(f"No cleaned test_seed_N folders found under {CLEAN_ROOT} - run test_seed_clean_hpc.py first")
        return
    print(f"Found {len(seed_names)} seeds: {seed_names}")

    counters = find_counters(seed_names)
    print(f"hpmcounters present after cleaning: {counters}")

    os.makedirs(PLOT_DIR, exist_ok=True)
    for counter in counters:
        plot_counter(counter, seed_names, PLOT_DIR)


if __name__ == "__main__":
    main()
