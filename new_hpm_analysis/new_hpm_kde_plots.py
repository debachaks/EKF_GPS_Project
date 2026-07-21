"""KDE plots of hpmcounter3-10 (new-HPM mapping) for seed_new_1..5: normal
vs each attack type, one subplot per counter.

Rows are pooled across all 5 runs per mode (5 runs x ~230-300 rows each)
purely for a smooth density estimate - this is a visualization of shape and
overlap, not a significance test (see new_hpm_run_level_hpmcounter_analysis.py
for the proper run-level statistical comparison).
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data_preprocessing import hex_to_int  # noqa: E402

CLEAN_ROOT = os.path.join(SCRIPT_DIR, "CLEAN_HPC_NEW_HPM")
PLOT_DIR = os.path.join(SCRIPT_DIR, "new_hpm_kde_plots")

COUNTERS = [f"hpmcounter{i}" for i in range(3, 11)]
MODES = ["normal", "drift", "jump", "replay"]

MODE_COLOR = {
    "normal": "#2a78d6",   # blue
    "drift": "#1baf7a",    # aqua
    "jump": "#eda100",     # yellow
    "replay": "#008300",   # green
}
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"


def find_seed_dirs():
    return sorted(d for d in glob.glob(os.path.join(CLEAN_ROOT, "seed_new_[0-9]*")) if os.path.isdir(d))


def pooled_values(seed_dirs, mode, counter):
    vals = []
    for seed_dir in seed_dirs:
        path = os.path.join(seed_dir, f"ekf_{mode}_hpc.csv")
        df = pd.read_csv(path)
        vals.append(df[counter].map(hex_to_int).to_numpy())
    return np.concatenate(vals)


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

    ax.set_title(
        f"{counter}: normal vs attack type (seed_new_1..5, pooled rows)",
        color=TEXT_PRIMARY, fontsize=12,
    )
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
    seed_dirs = find_seed_dirs()
    if not seed_dirs:
        print(f"No cleaned seed_new_N folders found under {CLEAN_ROOT} - run new_hpm_clean_hpc.py first")
        return
    os.makedirs(PLOT_DIR, exist_ok=True)

    for counter in COUNTERS:
        series = {mode: pooled_values(seed_dirs, mode, counter) for mode in MODES}
        plot_counter(counter, series)


if __name__ == "__main__":
    main()
