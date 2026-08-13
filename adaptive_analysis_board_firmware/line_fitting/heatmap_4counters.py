"""Single figure for hpmcounter3/4/8/10: 3 rows (z_stat, D, V) x 3 columns
(average of all 20 normal seeds, one jump trial, one drift trial).

Unlike heatmap_detection.py / per_counter_full_heatmap.py, this does NOT
divide values by threshold to share one color scale across counters.
Instead each of the 4 counters gets its OWN color scale (own colorbar)
showing its RAW metric value -- so within one row, there are 4 independent
color mappings, one per counter, each scaled to that counter's own units
(shared across the 3 mode columns, so normal/jump/drift ARE comparable to
each other for a given counter -- just not comparable across different
counters).

Each counter's color scale runs from 0 to that counter's REAL detection
threshold (h_z/h_D/h_V, calibrated from the 20 normal-mode baseline runs)
-- NOT to the max/percentile of whatever data happens to be in this
figure. This matters: a value that's only ~15% of the way to its own
threshold renders as pale green here, not red, even if it's the largest
value visible in one particular trial -- red specifically means "at or
above the real, calibrated attack threshold," not just "biggest number on
screen."

Layout per row: 3 wide heatmap columns (normal/jump/drift), each containing
all 4 counters stacked as 4 separate 1-row images (own norm per counter),
followed by 4 narrow colorbar axes, one per counter.
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
WINDOWED_PATH = os.path.join(RESULTS_DIR, "combined_detection_windowed.csv")
THRESHOLD_PATH = os.path.join(RESULTS_DIR, "combined_detection_thresholds.csv")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots_heatmap")

TARGET_COUNTERS = ["hpmcounter3", "hpmcounter4", "hpmcounter8", "hpmcounter10"]
MODES = ["normal", "jump", "drift"]
METRICS = [("z_stat", "h_z", "z"), ("D", "h_D", "D"), ("V", "h_V", "V")]
ATTACK_START = 150

CMAP = "RdYlGn_r"


def get_series(windowed, counter, value_col, mode, seed=None):
    d = windowed[(windowed["counter"] == counter) & (windowed["mode"] == mode)]
    if seed is not None:
        d = d[d["seed"] == seed]
        s = d.set_index("window_end_iter")[value_col].sort_index()
    else:
        s = d.groupby("window_end_iter")[value_col].mean().sort_index()
    return s


def seed_sort_key(s):
    m = re.match(r"seed(\d+)", s)
    return int(m.group(1)) if m else s


def make_figure(windowed, thresholds, test_seed):
    n = len(TARGET_COUNTERS)
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(
        3, 3 + n,
        width_ratios=[10, 10, 10] + [0.35] * n,
        wspace=0.15,
    )

    for row, (value_col, thresh_col, label) in enumerate(METRICS):
        # one series per counter per mode, all sharing the SAME iteration axis
        series = {
            (counter, mode): get_series(
                windowed, counter, value_col, mode,
                seed=(test_seed if mode != "normal" else None),
            )
            for counter in TARGET_COUNTERS for mode in MODES
        }
        iters = sorted(set().union(*[s.index for s in series.values()]))

        # per-counter color scale, in RAW units (not normalized), but
        # anchored to that counter's REAL detection threshold (h_z/h_D/h_V,
        # built from the 20 normal-mode baseline runs) as vmax, not to
        # whatever this trial's own data happens to reach. That's the
        # difference from the earlier percentile-based version: a raw value
        # that's genuinely small relative to the real threshold (e.g. V's
        # attack-window bump, ~15% of h_V) now correctly renders pale
        # green instead of misleadingly red just because it was the
        # largest value visible in this particular trial. Anything AT or
        # ABOVE the real threshold clips to the top (red) color.
        counter_norm = {
            counter: Normalize(vmin=0.0, vmax=thresholds.loc[counter, thresh_col])
            for counter in TARGET_COUNTERS
        }

        mode_axes = [fig.add_subplot(gs[row, c]) for c in range(3)]

        for col, mode in enumerate(MODES):
            ax = mode_axes[col]
            x0, x1 = min(iters), max(iters)

            for i, counter in enumerate(TARGET_COUNTERS):
                s = series[(counter, mode)].reindex(iters)
                data = s.to_numpy().reshape(1, -1)
                extent = [x0, x1, n - i, n - i - 1]   # stack counters top to bottom
                ax.imshow(data, aspect="auto", cmap=CMAP, norm=counter_norm[counter], extent=extent)

            ax.set_ylim(0, n)
            ax.axvline(ATTACK_START, color="black", linewidth=1, linestyle="--", alpha=0.6)
            ax.set_xlabel("EKF iteration", fontsize=8)

            if col == 0:
                ax.set_yticks(np.arange(n) + 0.5)
                ax.set_yticklabels(TARGET_COUNTERS[::-1], fontsize=8)
                ax.set_ylabel(f"{label}\ncounter", fontsize=10)
            else:
                ax.set_yticks([])

            if row == 0:
                title = "average normal (20 seeds)" if mode == "normal" else f"{test_seed} {mode}"
                ax.set_title(title, fontsize=11)

        # 4 colorbars for this row, one per counter, own scale
        for i, counter in enumerate(TARGET_COUNTERS):
            cax = fig.add_subplot(gs[row, 3 + i])
            sm = plt.cm.ScalarMappable(norm=counter_norm[counter], cmap=CMAP)
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label(counter, fontsize=8)
            cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        f"hpmcounter3/4/8/10: z / D / V, raw values ({test_seed})\n"
        "(color scale per counter: 0 -> that counter's real threshold h_z/h_D/h_V)",
        fontsize=14,
    )

    out_path = os.path.join(PLOT_DIR, f"heatmap_4counters_{test_seed}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    windowed = pd.read_csv(WINDOWED_PATH)
    thresholds = pd.read_csv(THRESHOLD_PATH).set_index("counter")

    seeds = sorted(windowed["seed"].unique(), key=seed_sort_key)
    for test_seed in seeds:
        make_figure(windowed, thresholds, test_seed)


if __name__ == "__main__":
    main()
