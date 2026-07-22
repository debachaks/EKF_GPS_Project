"""Raw time-series trace plots for a single seed: one PNG per (seed, counter),
overlaying the actual raw hpmcounter value against elapsed time for normal vs
drift vs jump vs replay.

Elapsed time (timestamp_ms - timestamp_ms.iloc[0]), not row index, is used for
the x-axis: normal/drift/jump/replay are independent runs with slightly
different row counts (e.g. 311-315 rows in test_seed_1), so raw index would
misalign samples that don't correspond to the same point in the run. Elapsed
time keeps each run's own start at 0 and lets runs of slightly different
length/sampling just end at different x, rather than forcing a false
alignment.

Output: test_seed_kde_plots/<counter>_per_seed/<seed>_<counter>_raw_trace.png
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from data_preprocessing import hex_to_int  # noqa: E402

CLEAN_ROOT = os.path.join(SRC_DIR, "CLEAN_HPC_TEST_SEED")
PLOT_ROOT = os.path.join(SRC_DIR, "test_seed_kde_plots")

MODES = ["normal", "drift", "jump", "replay"]
MODE_COLOR = {
    "normal": "#2a78d6",
    "drift": "#1baf7a",
    "jump": "#eda100",
    "replay": "#008300",
}
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"


def find_seed_dirs():
    return sorted(d for d in glob.glob(os.path.join(CLEAN_ROOT, "test_seed_[0-9]*")) if os.path.isdir(d))


def find_counters(seed_dirs):
    common = None
    for seed_dir in seed_dirs:
        for mode in MODES:
            path = os.path.join(seed_dir, f"ekf_{mode}_hpc.csv")
            cols = frozenset(c for c in pd.read_csv(path, nrows=1).columns if c.startswith("hpmcounter"))
            common = cols if common is None else (common & cols)
    return sorted(common, key=lambda c: int(c.replace("hpmcounter", "")))


def seed_trace(seed_dir, mode, counter):
    path = os.path.join(seed_dir, f"ekf_{mode}_hpc.csv")
    df = pd.read_csv(path)
    elapsed_ms = df["timestamp_ms"].map(hex_to_int) - df["timestamp_ms"].map(hex_to_int).iloc[0]
    values = df[counter].map(hex_to_int)
    return elapsed_ms.to_numpy(), values.to_numpy()


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_MUTED)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)


def plot_seed_counter(seed_name, counter, traces, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    for mode in MODES:
        elapsed_ms, values = traces[mode]
        ax.plot(elapsed_ms, values, color=MODE_COLOR[mode], linewidth=1.5, label=mode)

    ax.set_title(f"{counter}: raw value over elapsed time - {seed_name} only",
                 color=TEXT_PRIMARY, fontsize=12)
    ax.set_xlabel("elapsed time (ms since run start)", color=TEXT_MUTED)
    ax.set_ylabel("counter value", color=TEXT_MUTED)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    style_axes(ax)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{seed_name}_{counter}_raw_trace.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    seed_dirs = find_seed_dirs()
    if not seed_dirs:
        print(f"No cleaned test_seed_N folders found under {CLEAN_ROOT} - run test_seed_clean_hpc.py first")
        return

    counters = find_counters(seed_dirs)
    print(f"hpmcounters present after cleaning: {counters}")

    for counter in counters:
        out_dir = os.path.join(PLOT_ROOT, f"{counter}_per_seed")
        os.makedirs(out_dir, exist_ok=True)
        for seed_dir in seed_dirs:
            seed_name = os.path.basename(seed_dir)
            traces = {mode: seed_trace(seed_dir, mode, counter) for mode in MODES}
            plot_seed_counter(seed_name, counter, traces, out_dir)


if __name__ == "__main__":
    main()
