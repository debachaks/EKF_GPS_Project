"""KDE plots of the per-counter z-score distribution for one seed's three
runs (normal/jump/drift), overlaid. Reads zscore_baseline.py's already-
computed zscore_timeseries.csv -- no baseline/z recomputation here.

Each run contributes 299 z-score samples (one per iteration); the KDE
shows how that sample's distribution differs across modes for a single
seed, which is a different view than the windowed D/V/z_stat detectors --
those look at *where in the run* a deviation happens, this looks at the
*overall shape* of the z-score distribution regardless of iteration.

Change SEED below to look at a different seed (must have normal, jump,
AND drift runs present -- currently seed2, seed3, seed10..seed20).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")
PLOT_DIR = os.path.join(LINE_FITTING_DIR, "plots_kde")

SEED = "seed2"

COUNTERS = [f"hpmcounter{i}" for i in range(3, 11)]
MODES = ["normal", "jump", "drift"]
MODE_COLOR = {"normal": "#2a78d6", "jump": "#eda100", "drift": "#1baf7a"}

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"


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
    os.makedirs(PLOT_DIR, exist_ok=True)
    ts = pd.read_csv(TIMESERIES_PATH)

    seed_df = ts[ts["seed"] == SEED]
    if seed_df.empty:
        print(f"No runs found for seed={SEED!r} in {TIMESERIES_PATH}")
        return
    missing = [m for m in MODES if m not in seed_df["mode"].unique()]
    if missing:
        print(f"seed={SEED!r} is missing mode(s) {missing} -- plotting whatever is present")

    for counter in COUNTERS:
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
        any_plotted = False

        for mode in MODES:
            z = seed_df[(seed_df["counter"] == counter) & (seed_df["mode"] == mode)]["z"].to_numpy()
            z = z[np.isfinite(z)]
            if len(z) < 2 or np.std(z) < 1e-9:
                continue
            kde = gaussian_kde(z)
            x_grid = np.linspace(z.min() - 1, z.max() + 1, 400)
            ax.plot(x_grid, kde(x_grid), color=MODE_COLOR[mode], linewidth=2, label=mode)
            ax.fill_between(x_grid, kde(x_grid), color=MODE_COLOR[mode], alpha=0.15)
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        ax.set_title(f"{SEED}: {counter} z-score distribution by mode", color=TEXT_PRIMARY, fontsize=12)
        ax.set_xlabel("z = (raw_value - mu_normal(iter)) / sigma_normal(iter)", color=TEXT_MUTED)
        ax.set_ylabel("density", color=TEXT_MUTED)
        ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
        style_axes(ax)
        fig.tight_layout()

        out_path = os.path.join(PLOT_DIR, f"{SEED}_{counter}_kde.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
