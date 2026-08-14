"""Same figure as heatmap_4counters.py (hpmcounter3/4/8/10, z_stat/D/V rows,
normal/jump/drift columns, raw units, own color scale per counter) but with
the threshold placed at the MIDDLE of the color scale instead of the top:
0 -> green, threshold -> yellow/orange (exact color midpoint), further
above threshold -> red. heatmap_4counters.py itself is left untouched;
this is an alternate rendering for comparison.

Uses matplotlib's TwoSlopeNorm(vmin=0, vcenter=threshold, vmax=threshold*K)
per counter, in RAW units (not value/threshold like heatmap_detection.py) --
vcenter always maps to the middle of the colormap regardless of the
numeric gap to vmin/vmax, so forcing threshold to be the midpoint doesn't
require vmax = 2*threshold; the real design choice is just how far past
threshold the scale should extend before clipping to full red.

K is chosen PER METRIC (not one shared constant like the existing scripts'
VMAX=3.0), from the actual max(value/threshold) observed across all
normal/jump/drift runs in this dataset, with a little headroom -- rather
than an arbitrary shared cap:

    z_stat: observed max ratio ~2.14  -> K_Z = 2.5
    D:      observed max ratio ~4.71  -> K_D = 5.0   (a shared K=3.0 would
                                                        clip real D values)

V row uses the z-scored, W_V=50 variant (v_zscore_w50.py: z_V standardized
against its own normal-mode baseline) instead of variability_metric.py's
plain V -- found to be a much stronger detector (drift 60-65% vs 0-15%,
jump 100% vs near-0% on hpmcounter3/4/5/8/10). z_stat and D stay at W=10,
unchanged; only V's data source/window size changed. Deliberate tradeoff:
W=50 blends a sixth of the whole 300-iteration run into every window, so
this row shows overall run "shape" rather than precise attack-onset
timing the way the W=10 z_stat/D rows do -- the first valid window doesn't
end until iteration 49. z_V's ratio range is much wider than plain V's
(max ~15.6x threshold, 99th pct ~8.9x, vs plain V's ~1.47x), so K_V=10 (a
little headroom above the 99th percentile) is used instead of the raw
max, so the rare extreme outlier doesn't wash out the rest of the scale:

    z_V:    99th pct ratio ~8.9  -> K_V = 10
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
V_WINDOWED_PATH = os.path.join(RESULTS_DIR, "v_zscore_w50.csv")
V_THRESHOLD_PATH = os.path.join(RESULTS_DIR, "v_zscore_threshold_w50.csv")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots_heatmap")

TARGET_COUNTERS = ["hpmcounter3", "hpmcounter4", "hpmcounter8", "hpmcounter10"]
MODES = ["normal", "jump", "drift"]
# (value_col, thresh_col, label, k, "windowed"|"v_windowed" data source)
METRICS = [
    ("z_stat", "h_z", "z", 2.5, "windowed"),
    ("D", "h_D", "D", 5.0, "windowed"),
    ("z_V", "h_zV", "V (W=50)", 10.0, "v_windowed"),
]
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


def make_figure(data_sources, threshold_sources, test_seed):
    n = len(TARGET_COUNTERS)
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(
        3, 3 + n,
        width_ratios=[10, 10, 10] + [0.35] * n,
        wspace=0.15,
    )

    for row, (value_col, thresh_col, label, k, source) in enumerate(METRICS):
        windowed = data_sources[source]
        thresholds = threshold_sources[source]

        series = {
            (counter, mode): get_series(
                windowed, counter, value_col, mode,
                seed=(test_seed if mode != "normal" else None),
            )
            for counter in TARGET_COUNTERS for mode in MODES
        }
        iters = sorted(set().union(*[s.index for s in series.values()]))

        # threshold sits at the color midpoint (vcenter); vmax = k * threshold,
        # k chosen per metric from the real observed data range (see docstring)
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
        f"hpmcounter3/4/8/10: z (W=10) / D (W=10) / V (z-scored, W=50), raw values ({test_seed})\n"
        "(color scale per counter: threshold sits at the MIDDLE (yellow); "
        "vmax = 2.5x/5x/10x threshold per metric, from observed data range)",
        fontsize=14,
    )

    out_path = os.path.join(PLOT_DIR, f"heatmap_4counters_midthreshold_{test_seed}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    windowed = pd.read_csv(WINDOWED_PATH)
    thresholds = pd.read_csv(THRESHOLD_PATH).set_index("counter")
    v_windowed = pd.read_csv(V_WINDOWED_PATH)
    v_thresholds = pd.read_csv(V_THRESHOLD_PATH).set_index("counter")

    data_sources = {"windowed": windowed, "v_windowed": v_windowed}
    threshold_sources = {"windowed": thresholds, "v_windowed": v_thresholds}

    seeds = sorted(windowed["seed"].unique(), key=seed_sort_key)
    for test_seed in seeds:
        make_figure(data_sources, threshold_sources, test_seed)


if __name__ == "__main__":
    main()
