"""KDE plots of hpmcounter3-10 for test_seed/: normal vs each attack type,
one PNG per counter. Single run (not pooled/averaged across multiple
seeds - test_seed/ is one run per mode).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from data_preprocessing import hex_to_int  # noqa: E402

CLEAN_DIR = os.path.join(SRC_DIR, "CLEAN_HPC_TEST_SEED")
PLOT_DIR = os.path.join(SRC_DIR, "test_seed_kde_plots")

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


def find_counters():
    df = pd.read_csv(os.path.join(CLEAN_DIR, "ekf_normal_hpc.csv"), nrows=1)
    return sorted(
        (c for c in df.columns if c.startswith("hpmcounter")),
        key=lambda c: int(c.replace("hpmcounter", "")),
    )


def mode_values(mode, counter):
    path = os.path.join(CLEAN_DIR, f"ekf_{mode}_hpc.csv")
    df = pd.read_csv(path)
    return df[counter].map(hex_to_int).to_numpy()


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_MUTED)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)


def plot_counter(counter, series):
    all_vals = np.concatenate(list(series.values()))
    lo, hi = all_vals.min(), all_vals.max()
    pad = (hi - lo) * 0.05 if hi > lo else 1
    xs = np.linspace(lo - pad, hi + pad, 400)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    for mode in MODES:
        vals = series[mode]
        if np.ptp(vals) == 0:
            ax.axvline(vals[0], color=MODE_COLOR[mode], linewidth=2, label=mode)
            continue
        kde = gaussian_kde(vals)
        ys = kde(xs)
        ax.plot(xs, ys, color=MODE_COLOR[mode], linewidth=2, label=mode)
        ax.fill_between(xs, ys, color=MODE_COLOR[mode], alpha=0.10)

    ax.set_title(f"{counter}: normal vs attack type (test_seed)", color=TEXT_PRIMARY, fontsize=12)
    ax.set_xlabel("counter value", color=TEXT_MUTED)
    ax.set_ylabel("density", color=TEXT_MUTED)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    style_axes(ax)

    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, f"{counter}_kde.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    counters = find_counters()
    print(f"hpmcounters present after cleaning: {counters}")
    for counter in counters:
        series = {mode: mode_values(mode, counter) for mode in MODES}
        plot_counter(counter, series)


if __name__ == "__main__":
    main()
