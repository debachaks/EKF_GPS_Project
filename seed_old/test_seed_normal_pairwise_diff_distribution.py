"""Normal-vs-normal pairwise difference distribution, per counter.

Establishes what "no attack at all" noise looks like: for every counter,
takes all unique pairs of the 20 normal runs (190 pairs), interpolates each
run's raw value onto a shared elapsed-time grid (runs have slightly
different row counts, as in test_seed_raw_trace_per_seed.py), and computes
the signed difference (run_i - run_j) at every point on that grid for every
pair. At each time point, plots the mean difference across all 190 pairs
(solid line) and the mean +/- one standard deviation (dotted lines).

If normal-vs-normal noise hovers around zero with a roughly constant
spread, that's a clean baseline. If the mean drifts away from zero over
time, that itself is evidence of a systematic (non-random) effect between
normal runs - e.g. the run-order artifact found elsewhere in this
analysis - not just noise.

Output: test_seed_normal_pairwise_diff_plots/<counter>_pairwise_diff.png
"""

import glob
import itertools
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
PLOT_DIR = os.path.join(SRC_DIR, "test_seed_normal_pairwise_diff_plots")
GRID_POINTS = 300

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"
MEAN_COLOR = "#2a78d6"
BAND_COLOR = "#2a78d6"


def find_seed_names():
    dirs = sorted(d for d in glob.glob(os.path.join(CLEAN_ROOT, "test_seed_[0-9]*")) if os.path.isdir(d))
    return [os.path.basename(d) for d in dirs]


def find_counters(seed_names):
    common = None
    for seed_name in seed_names:
        path = os.path.join(CLEAN_ROOT, seed_name, "ekf_normal_hpc.csv")
        cols = frozenset(c for c in pd.read_csv(path, nrows=1).columns if c.startswith("hpmcounter"))
        common = cols if common is None else (common & cols)
    return sorted(common, key=lambda c: int(c.replace("hpmcounter", "")))


def normal_trace(seed_name, counter):
    path = os.path.join(CLEAN_ROOT, seed_name, "ekf_normal_hpc.csv")
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
    ax.axhline(0, color=AXIS_COLOR, linewidth=1)
    ax.set_axisbelow(True)


def build_pairwise_diffs(seed_names, counter):
    traces = {s: normal_trace(s, counter) for s in seed_names}
    grid_end = min(elapsed.max() for elapsed, _ in traces.values())
    grid = np.linspace(0, grid_end, GRID_POINTS)

    interp = {s: np.interp(grid, elapsed, values) for s, (elapsed, values) in traces.items()}

    diffs = np.array([interp[a] - interp[b] for a, b in itertools.combinations(seed_names, 2)])
    return grid, diffs


def plot_counter(counter, grid, diffs, out_dir):
    mean_diff = diffs.mean(axis=0)
    std_diff = diffs.std(axis=0)
    n_pairs = diffs.shape[0]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.plot(grid, mean_diff, color=MEAN_COLOR, linewidth=2, label=f"mean of {n_pairs} normal-vs-normal pairs")
    ax.plot(grid, mean_diff + std_diff, color=BAND_COLOR, linewidth=1.2, linestyle="--", label="mean +/- 1 std")
    ax.plot(grid, mean_diff - std_diff, color=BAND_COLOR, linewidth=1.2, linestyle="--")

    ax.set_title(f"{counter}: normal-vs-normal pairwise difference (n={n_pairs} pairs)",
                 color=TEXT_PRIMARY, fontsize=12)
    ax.set_xlabel("elapsed time (ms since run start)", color=TEXT_MUTED)
    ax.set_ylabel("run_i - run_j (counter value)", color=TEXT_MUTED)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    style_axes(ax)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{counter}_pairwise_diff.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    seed_names = find_seed_names()
    if not seed_names:
        print(f"No cleaned test_seed_N folders found under {CLEAN_ROOT} - run test_seed_clean_hpc.py first")
        return
    print(f"Found {len(seed_names)} normal runs: {seed_names}")

    counters = find_counters(seed_names)
    print(f"hpmcounters present after cleaning: {counters}")

    os.makedirs(PLOT_DIR, exist_ok=True)
    for counter in counters + ["sp", "fp"]:
        grid, diffs = build_pairwise_diffs(seed_names, counter)
        plot_counter(counter, grid, diffs, PLOT_DIR)


if __name__ == "__main__":
    main()
