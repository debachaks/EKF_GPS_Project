"""Raw-value variant of test_seed_normal_attack_diff_distribution_rate.py.

Same normal-vs-attack, run-wise (same-seed-paired) difference distribution,
but using the raw accumulated counter value instead of the rate metric -
companion to test_seed_normal_pairwise_diff_distribution.py (the raw-value
normal-vs-normal baseline).

For each counter and each attack type: for each of the 20 seeds, normal's
own raw trace minus that same seed's jump/drift/replay raw trace,
interpolated onto a shared elapsed-time grid. Across the 20 seeds this
gives 20 difference curves per attack type; the plot shows their mean
(solid) and mean +/- 1 std (dotted), for jump, drift, and replay together
on one figure per counter.

Output: test_seed_normal_attack_diff_plots/<counter>_normal_vs_attack_diff.png
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
PLOT_DIR = os.path.join(SRC_DIR, "test_seed_normal_attack_diff_plots")
GRID_POINTS = 300

ANOMALY_TYPES = ["jump", "drift", "replay"]
MODE_COLOR = {
    "jump": "#eda100",
    "drift": "#2a78d6",
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
        for mode in ["normal"] + ANOMALY_TYPES:
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


def build_mean_std(seed_names, mode, counter):
    normal_traces = {s: raw_trace(s, "normal", counter) for s in seed_names}
    attack_traces = {s: raw_trace(s, mode, counter) for s in seed_names}

    grid_end = min(
        min(normal_traces[s][0].max(), attack_traces[s][0].max()) for s in seed_names
    )
    grid = np.linspace(0, grid_end, GRID_POINTS)

    diffs = []
    for s in seed_names:
        n_elapsed, n_values = normal_traces[s]
        a_elapsed, a_values = attack_traces[s]
        n_interp = np.interp(grid, n_elapsed, n_values)
        a_interp = np.interp(grid, a_elapsed, a_values)
        diffs.append(n_interp - a_interp)
    diffs = np.array(diffs)

    return grid, diffs.mean(axis=0), diffs.std(axis=0), len(seed_names)


def plot_counter(counter, seed_names, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)

    for mode in ANOMALY_TYPES:
        grid, mean_diff, std_diff, n = build_mean_std(seed_names, mode, counter)
        color = MODE_COLOR[mode]
        ax.plot(grid, mean_diff, color=color, linewidth=2, label=f"normal - {mode} (mean, n={n} seeds)")
        ax.plot(grid, mean_diff + std_diff, color=color, linewidth=1, linestyle="--")
        ax.plot(grid, mean_diff - std_diff, color=color, linewidth=1, linestyle="--")

    ax.set_title(f"{counter}: normal minus attack, run-wise (mean +/- 1 std)",
                 color=TEXT_PRIMARY, fontsize=12)
    ax.set_xlabel("elapsed time (ms since run start)", color=TEXT_MUTED)
    ax.set_ylabel("normal_value - attack_value (counter value)", color=TEXT_MUTED)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
    style_axes(ax)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{counter}_normal_vs_attack_diff.png")
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
