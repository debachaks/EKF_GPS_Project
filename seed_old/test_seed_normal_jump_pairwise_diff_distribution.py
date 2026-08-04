"""Normal-vs-jump companion to test_seed_normal_pairwise_diff_distribution.py.

Same raw-value pairwise difference technique (interpolate each run's raw
counter value onto a shared elapsed-time grid, difference at every grid
point), but instead of the 190 unordered normal-vs-normal pairs
(itertools.combinations over one set of 20 runs), this takes ALL 400
ordered pairs of (normal seed X, jump seed Y) via itertools.product over
the two separate 20-seed sets: normal_seed1 vs jump_seed1, normal_seed1 vs
jump_seed2, ..., normal_seed20 vs jump_seed20.

For each counter: diff_i = jump_Y(t) - normal_X(t) (raw value, not rate) for
all 400 (X, Y) pairs, then plot mean(t) +/- std(t) across those 400 diff
curves -- answers "does jump systematically shift the raw counter value
relative to normal, regardless of which specific normal/jump seed you
happen to pair it with."

Output: test_seed_normal_jump_pairwise_diff_plots/<counter>_pairwise_diff.png
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
PLOT_DIR = os.path.join(SRC_DIR, "test_seed_normal_jump_pairwise_diff_plots")
GRID_POINTS = 300

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"
MEAN_COLOR = "#eda100"
BAND_COLOR = "#eda100"


def find_seed_names():
    dirs = sorted(d for d in glob.glob(os.path.join(CLEAN_ROOT, "test_seed_[0-9]*")) if os.path.isdir(d))
    return [os.path.basename(d) for d in dirs]


def find_counters(seed_names):
    common = None
    for seed_name in seed_names:
        for mode in ("normal", "jump"):
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
    ax.axhline(0, color=AXIS_COLOR, linewidth=1)
    ax.set_axisbelow(True)


def build_pairwise_diffs(seed_names, counter):
    normal_traces = {s: raw_trace(s, "normal", counter) for s in seed_names}
    jump_traces = {s: raw_trace(s, "jump", counter) for s in seed_names}

    grid_end = min(
        elapsed.max()
        for traces in (normal_traces, jump_traces)
        for elapsed, _ in traces.values()
    )
    grid = np.linspace(0, grid_end, GRID_POINTS)

    normal_interp = {s: np.interp(grid, elapsed, values) for s, (elapsed, values) in normal_traces.items()}
    jump_interp = {s: np.interp(grid, elapsed, values) for s, (elapsed, values) in jump_traces.items()}

    diffs = np.array([
        jump_interp[jump_seed] - normal_interp[normal_seed]
        for normal_seed, jump_seed in itertools.product(seed_names, seed_names)
    ])
    return grid, diffs


def plot_counter(counter, grid, diffs, out_dir):
    mean_diff = diffs.mean(axis=0)
    std_diff = diffs.std(axis=0)
    n_pairs = diffs.shape[0]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.plot(grid, mean_diff, color=MEAN_COLOR, linewidth=2, label=f"mean of {n_pairs} normal-vs-jump pairs")
    ax.plot(grid, mean_diff + std_diff, color=BAND_COLOR, linewidth=1.2, linestyle="--", label="mean +/- 1 std")
    ax.plot(grid, mean_diff - std_diff, color=BAND_COLOR, linewidth=1.2, linestyle="--")

    ax.set_title(f"{counter}: normal-vs-jump pairwise difference (n={n_pairs} pairs)",
                 color=TEXT_PRIMARY, fontsize=12)
    ax.set_xlabel("elapsed time (ms since run start)", color=TEXT_MUTED)
    ax.set_ylabel("jump - normal (counter value)", color=TEXT_MUTED)
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
    print(f"Found {len(seed_names)} seeds -> {len(seed_names) ** 2} normal-vs-jump pairs")

    counters = find_counters(seed_names)
    print(f"hpmcounters present after cleaning: {counters}")

    os.makedirs(PLOT_DIR, exist_ok=True)
    for counter in counters + ["sp", "fp"]:
        grid, diffs = build_pairwise_diffs(seed_names, counter)
        plot_counter(counter, grid, diffs, PLOT_DIR)


if __name__ == "__main__":
    main()
