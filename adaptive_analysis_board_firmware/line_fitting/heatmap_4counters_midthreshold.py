"""hpmcounter3/4/8/10 heatmap, D_final/G_final/V_final rows, normal/jump/
drift columns, one PNG per seed (plus the normal column averaged across
all 20 seeds) -- same visual format as the original z_stat/D/V(z-scored)
version this replaces, but sourced from the window-diff-then-z-score
"final" metric family (d_final_metric.py, g_final_metric.py,
v_final_metric.py) instead.

Threshold placed at the MIDDLE of the color scale (matplotlib's
TwoSlopeNorm(vmin=0, vcenter=threshold, vmax=threshold*K)): 0 -> green,
threshold -> yellow/orange, further above threshold -> red. Color value
is |score| (all three metrics are signed: score = (X_new - mu) / sigma),
since detection itself thresholds on the absolute value.

Positions flagged sigma_fragile (near-zero baseline variance -- see each
metric's own script for why those are excluded from thresholding/
detection) are masked to NaN in get_series, then linearly interpolated
across in make_figure so the row reads as a continuous strip instead of
a blank gap. This is a visual smoothing only -- there is no real computed
value at a fragile position, and detection/thresholding still excludes
these positions entirely (see each metric's own script); the interpolated
color is not evidence of a genuine signal there.

K is chosen PER METRIC from the 99th percentile of |score|/threshold
observed across the target counters (not the raw max -- unlike the
original z_stat/D, all three of these metrics have a handful of extreme
outlier ratios in the hundreds, presumably from positions where sigma
sits just above the 1e-6 fragility floor but is still small; the 99th
percentile is a far more representative "how big does this normally get"
number), with headroom so the rare high tail doesn't wash out the rest
of the scale.

Window sizes are per-metric, not shared: D_final stays at W=10
(d_final_metric.py), G_final and V_final use the W=5 variants
(g_final_metric_w5.py, v_final_metric_w5.py) instead of their W=10
default -- separate scripts/output files so the W=10 results (still used
by g_final_sweep.py/v_final_sweep.py and other comparisons) aren't lost.

    D_final (W=10): 99th pct ratio ~2.4  -> K_D = 3.0
    G_final (W=5):  99th pct ratio ~1.9  -> K_G = 2.5
    V_final (W=5):  99th pct ratio ~1.1  -> K_V = 2.0
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots_heatmap")

TARGET_COUNTERS = ["hpmcounter3", "hpmcounter4", "hpmcounter8", "hpmcounter10"]
MODES = ["normal", "jump", "drift"]
# (score_col, thresh_col, label, k, score_path, thresh_path)
METRICS = [
    ("d", "H_d", "D_final (W=10)", 3.0, "d_final_dscore.csv", "d_final_thresholds.csv"),
    ("g", "H_g", "G_final (W=5)", 2.5, "g_final_w5_gscore.csv", "g_final_w5_thresholds.csv"),
    ("v", "H_v", "V_final (W=5)", 2.0, "v_final_w5_vscore.csv", "v_final_w5_thresholds.csv"),
]
ATTACK_START = 150

CMAP = "RdYlGn_r"


def get_series(scored, counter, value_col, mode, seed=None):
    d = scored[(scored["counter"] == counter) & (scored["mode"] == mode)]
    d = d.copy()
    d[value_col] = d[value_col].abs().where(~d["sigma_fragile"])
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

    for row, (value_col, thresh_col, label, k, _, _) in enumerate(METRICS):
        scored = data_sources[label]
        thresholds = threshold_sources[label]

        series = {
            (counter, mode): get_series(
                scored, counter, value_col, mode,
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
                # sigma-fragile positions are NaN here (masked in get_series);
                # interpolate across them so the row reads continuously instead
                # of leaving a blank gap -- this is purely a visual smoothing
                # of the color, not a real computed value at that position.
                s = s.interpolate(method="linear", limit_direction="both")
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
        f"hpmcounter3/4/8/10: D_final (W=10) / G_final (W=5) / V_final (W=5), |score| ({test_seed})\n"
        "(color scale per counter: threshold sits at the MIDDLE (yellow); "
        "vmax = 3x/2.5x/2x threshold per metric, from observed 99th-percentile ratio; "
        "sigma-fragile positions interpolated, not real values)",
        fontsize=14,
    )

    out_path = os.path.join(PLOT_DIR, f"heatmap_4counters_midthreshold_{test_seed}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    data_sources, threshold_sources = {}, {}
    for _, thresh_col, label, _, score_path, thresh_path in METRICS:
        scored = pd.read_csv(os.path.join(RESULTS_DIR, score_path))
        thresholds = pd.read_csv(os.path.join(RESULTS_DIR, thresh_path)).set_index("counter")
        data_sources[label] = scored
        threshold_sources[label] = thresholds

    seeds = sorted(data_sources["D_final (W=10)"]["seed"].unique(), key=seed_sort_key)
    for test_seed in seeds:
        make_figure(data_sources, threshold_sources, test_seed)


if __name__ == "__main__":
    main()
