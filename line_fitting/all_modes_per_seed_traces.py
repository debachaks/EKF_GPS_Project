"""Per-seed, per-counter raw value traces: for every (counter, seed) pair,
one plot showing all 4 modes (normal, drift, jump, replay) as raw elapsed-time
vs counter-value lines, using that seed's own runs only (no cross-seed
aggregation).

20 seeds x 8 counters = 160 plots.

Reuses the same raw-trace decoding as line_fitting_analysis.py /
seed_counter_traces.py (hex_to_int decode, elapsed time relative to each
run's own start).
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
while not os.path.isdir(os.path.join(PROJECT_ROOT, "original_pipeline")):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from data_preprocessing import hex_to_int  # noqa: E402

CLEAN_ROOT = os.path.join(PROJECT_ROOT, "seed_old", "CLEAN_HPC_TEST_SEED")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots", "all_modes_per_seed")

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
    ts = df["timestamp_ms"].map(hex_to_int)
    elapsed = ts - ts.iloc[0]
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


def main():
    seed_names = find_seed_names()
    counters = find_counters(seed_names)
    print(f"{len(seed_names)} seeds, {len(counters)} counters -> {len(seed_names) * len(counters)} plots")

    for counter in counters:
        counter_dir = os.path.join(PLOT_DIR, counter)
        os.makedirs(counter_dir, exist_ok=True)

        for seed_name in seed_names:
            fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)

            for mode in MODES:
                elapsed, values = raw_trace(seed_name, mode, counter)
                ax.plot(elapsed, values, color=MODE_COLOR[mode], linewidth=1.5, label=mode)

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
