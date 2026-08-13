"""Per-counter reproducibility heatmap: for each of TARGET_COUNTERS, one row
with seed (y-axis, all 20) x EKF iteration (x-axis), color = D / h_D (that
counter's own trend-score value normalized by its own threshold, same
convention as heatmap_detection.py). Columns are jump and drift.

This is the seed-level check the counter-level heatmap_detection.py can't
answer: trend_score_detection_summary.csv says e.g. hpmcounter10 detects
jump in 19/20 trials, but that's a single aggregate number -- this plot
shows WHICH seeds and WHEN, so a high detection rate that's actually
concentrated in a few seeds (not reproducible) looks visually different
from one that's consistent across all 20 rows.

D only (not z/V) -- D is the metric that actually showed strong detection
in trend_score_detection_summary.csv; z and V were both weak there.
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
MODES = ["jump", "drift"]
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

    fig, axes = plt.subplots(
        len(TARGET_COUNTERS), len(MODES), figsize=(14, 3.2 * len(TARGET_COUNTERS)),
        squeeze=False,
    )
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=VMAX)

    im = None
    for row, counter in enumerate(TARGET_COUNTERS):
        h_D = thresholds.loc[counter, "h_D"]

        for col, mode in enumerate(MODES):
            d = windowed[(windowed["counter"] == counter) & (windowed["mode"] == mode)]
            pivot = d.pivot(index="seed", columns="window_end_iter", values="D").reindex(seeds)
            data = pivot.to_numpy() / h_D

            ax = axes[row, col]
            extent = [pivot.columns.min(), pivot.columns.max(), len(seeds), 0]
            im = ax.imshow(data, aspect="auto", cmap=CMAP, norm=norm, extent=extent)
            ax.axvline(ATTACK_START, color="black", linewidth=1, linestyle="--", alpha=0.6)

            if col == 0:
                ax.set_yticks(np.arange(len(seeds)) + 0.5)
                ax.set_yticklabels(seeds, fontsize=6)
                ax.set_ylabel(f"{counter}\n(h_D={h_D:.1f})\nseed", fontsize=9)
            else:
                ax.set_yticks([])

            ax.set_xlabel("EKF iteration", fontsize=8)
            if row == 0:
                ax.set_title(mode, fontsize=11)

    fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, label="D / h_D  (1.0 = threshold)")
    fig.suptitle("Per-counter reproducibility: D/threshold, all 20 seeds", fontsize=13)

    out_path = os.path.join(PLOT_DIR, "per_counter_seed_heatmap.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
