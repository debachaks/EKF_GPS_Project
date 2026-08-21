"""Same as heatmap_4counters_midthreshold.py, but built on the WINDOW=20
D/V computation (combined_detection_w20.py's outputs) instead of WINDOW=10,
for a side-by-side comparison of how window size changes the picture.

vmax multipliers (K) were re-derived from the W=20 data, NOT reused from
the W=10 version, since the observed value/threshold range shifted:

    z_stat: observed max ratio ~2.14 (same as W=10) -> K_Z = 2.5 (unchanged)
    D:      observed max ratio ~8.28 (W=10 was ~4.71) -> K_D = 9.0
            (h_D itself dropped a lot at W=20 -- more degrees of freedom
            in the OLS fit shrinks the normal-baseline ceiling faster than
            it shrinks the strongest attack windows, so the RATIO grew)
    V:      observed max ratio ~1.23 (W=10 was ~1.47) -> K_V = 1.5

z_stat's ratio is essentially unchanged because z_stat = max(|z|) over a
window converges to the same value (the run's global peak |z|) for any
window size once the window is smaller than the run length -- it isn't
actually sensitive to W the way D and V are.
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
WINDOWED_PATH = os.path.join(RESULTS_DIR, "combined_detection_windowed_w20.csv")
THRESHOLD_PATH = os.path.join(RESULTS_DIR, "combined_detection_thresholds_w20.csv")
PLOT_DIR = os.path.join(LINE_FITTING_DIR, "plots_heatmap")

TARGET_COUNTERS = ["hpmcounter3", "hpmcounter4", "hpmcounter8", "hpmcounter10"]
MODES = ["normal", "jump", "drift"]
METRICS = [("z_stat", "h_z", "z", 2.5), ("D", "h_D", "D", 9.0), ("V", "h_V", "V", 1.5)]
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

    for row, (value_col, thresh_col, label, k) in enumerate(METRICS):
        series = {
            (counter, mode): get_series(
                windowed, counter, value_col, mode,
                seed=(test_seed if mode != "normal" else None),
            )
            for counter in TARGET_COUNTERS for mode in MODES
        }
        iters = sorted(set().union(*[s.index for s in series.values()]))

        counter_norm = {
            counter: TwoSlopeNorm(
                vmin=0.0,
                vcenter=thresholds.loc[counter, thresh_col],
                vmax=thresholds.loc[counter, thresh_col] * k,
            )
            for counter in TARGET_COUNTERS
        }

        mode_axes = [fig.add_subplot(gs[row, c]) for c in range(3)]

        for col, mode in enumerate(MODES):
            ax = mode_axes[col]
            x0, x1 = min(iters), max(iters)

            for i, counter in enumerate(TARGET_COUNTERS):
                s = series[(counter, mode)].reindex(iters)
                data = s.to_numpy().reshape(1, -1)
                extent = [x0, x1, n - i, n - i - 1]
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

        for i, counter in enumerate(TARGET_COUNTERS):
            cax = fig.add_subplot(gs[row, 3 + i])
            sm = plt.cm.ScalarMappable(norm=counter_norm[counter], cmap=CMAP)
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label(counter, fontsize=8)
            cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        f"hpmcounter3/4/8/10: z / D / V, raw values, WINDOW=20 ({test_seed})\n"
        "(color scale per counter: threshold h_z/h_D/h_V sits at the MIDDLE "
        "(yellow); vmax = 2.5x/9x/1.5x threshold per metric, from observed W=20 data range)",
        fontsize=14,
    )

    out_path = os.path.join(PLOT_DIR, f"heatmap_4counters_midthreshold_w20_{test_seed}.png")
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
