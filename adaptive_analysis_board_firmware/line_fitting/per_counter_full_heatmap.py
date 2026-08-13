"""One heatmap PER counter (TARGET_COUNTERS): 3 rows (z_stat, D, V) x 3
columns (normal, jump, drift), y-axis = seed (all 20), x-axis = EKF
iteration, color = metric value / that counter's own threshold (same
convention as heatmap_detection.py / per_counter_seed_heatmap.py).

Combines the two earlier heatmaps' ideas: heatmap_detection.py's 3-metric
rows, and per_counter_seed_heatmap.py's per-seed reproducibility view
(instead of averaging/picking one test seed, every seed gets its own row)
- now for one counter at a time, with normal mode included as a reference
column to check nothing lights up there across all 20 seeds.
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
WINDOWED_PATH = os.path.join(RESULTS_DIR, "combined_detection_windowed.csv")
THRESHOLD_PATH = os.path.join(RESULTS_DIR, "combined_detection_thresholds.csv")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots_heatmap")

TARGET_COUNTERS = ["hpmcounter8", "hpmcounter10", "hpmcounter3", "hpmcounter4"]
MODES = ["normal", "jump", "drift"]
METRICS = [("z_stat", "h_z", "z"), ("D", "h_D", "D"), ("V", "h_V", "V")]
ATTACK_START = 150

VMAX = 3.0
CMAP = "RdYlGn_r"


def seed_sort_key(s):
    m = re.match(r"seed(\d+)", s)
    return int(m.group(1)) if m else s


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    windowed = pd.read_csv(WINDOWED_PATH)
    thresholds = pd.read_csv(THRESHOLD_PATH).set_index("counter")
    seeds = sorted(windowed["seed"].unique(), key=seed_sort_key)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=VMAX)

    for counter in TARGET_COUNTERS:
        fig, axes = plt.subplots(3, 3, figsize=(16, 11), squeeze=False)

        for row, (value_col, thresh_col, label) in enumerate(METRICS):
            h = thresholds.loc[counter, thresh_col]
            im = None

            for col, mode in enumerate(MODES):
                d = windowed[(windowed["counter"] == counter) & (windowed["mode"] == mode)]
                pivot = d.pivot(index="seed", columns="window_end_iter", values=value_col).reindex(seeds)
                with np.errstate(divide="ignore", invalid="ignore"):
                    data = pivot.to_numpy() / h

                ax = axes[row, col]
                extent = [pivot.columns.min(), pivot.columns.max(), len(seeds), 0]
                im = ax.imshow(data, aspect="auto", cmap=CMAP, norm=norm, extent=extent)
                ax.axvline(ATTACK_START, color="black", linewidth=1, linestyle="--", alpha=0.6)

                if col == 0:
                    ax.set_yticks(np.arange(len(seeds)) + 0.5)
                    ax.set_yticklabels(seeds, fontsize=6)
                    ax.set_ylabel(f"{label} (h={h:.3g})\nseed", fontsize=9)
                else:
                    ax.set_yticks([])

                ax.set_xlabel("EKF iteration", fontsize=8)
                if row == 0:
                    ax.set_title(mode, fontsize=12)

            # one colorbar per metric (row) -- numerically identical across
            # rows (all share the same norm), but visually tied to its own
            # row instead of one bar for the whole grid
            fig.colorbar(im, ax=axes[row, :], shrink=0.85, pad=0.015, label=f"{label} / h_{label}")

        fig.suptitle(f"{counter}: z / D / V across all 20 seeds", fontsize=14)

        out_path = os.path.join(PLOT_DIR, f"per_counter_full_{counter}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
