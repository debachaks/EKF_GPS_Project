"""Per-seed KDE plots for every remaining hpmcounter (new-HPM mapping,
post-cleaning): one PNG per (counter, seed_new_N), using only that seed's
rows - normal vs drift vs jump vs replay. Generalizes
new_hpm_kde_plots_hpmcounter5_per_seed.py to all counters that survived
new_hpm_clean_hpc.py's filters, not just hpmcounter5.

Output: new_hpm_kde_plots/<counter>_per_seed/<seed>_<counter>_kde.png
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
PLOT_ROOT = os.path.join(SCRIPT_DIR, "new_hpm_kde_plots")

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


def find_counters(seed_dirs):
    common = None
    for seed_dir in seed_dirs:
        for mode in MODES:
            path = os.path.join(seed_dir, f"ekf_{mode}_hpc.csv")
            cols = frozenset(c for c in pd.read_csv(path, nrows=1).columns if c.startswith("hpmcounter"))
            common = cols if common is None else (common & cols)
    return sorted(common, key=lambda c: int(c.replace("hpmcounter", "")))


def seed_values(seed_dir, mode, counter):
    path = os.path.join(seed_dir, f"ekf_{mode}_hpc.csv")
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


def plot_seed_counter(seed_name, counter, series, out_dir):
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

    ax.set_title(f"{counter}: normal vs attack type - {seed_name} only",
                 color=TEXT_PRIMARY, fontsize=12)
    ax.set_xlabel("counter value", color=TEXT_MUTED)
    ax.set_ylabel("density", color=TEXT_MUTED)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    style_axes(ax)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{seed_name}_{counter}_kde.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    seed_dirs = find_seed_dirs()
    if not seed_dirs:
        print(f"No cleaned seed_new_N folders found under {CLEAN_ROOT} - run new_hpm_clean_hpc.py first")
        return

    counters = find_counters(seed_dirs)
    print(f"hpmcounters present after cleaning: {counters}")

    for counter in counters:
        out_dir = os.path.join(PLOT_ROOT, f"{counter}_per_seed")
        os.makedirs(out_dir, exist_ok=True)
        for seed_dir in seed_dirs:
            seed_name = os.path.basename(seed_dir)
            series = {mode: seed_values(seed_dir, mode, counter) for mode in MODES}
            plot_seed_counter(seed_name, counter, series, out_dir)


if __name__ == "__main__":
    main()
