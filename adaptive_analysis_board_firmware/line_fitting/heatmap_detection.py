"""Heatmap version of the z/D/V detection story: for each metric (row),
one heatmap per view (column) - counters (y) x EKF iteration (x), color =
metric value normalized by that counter's OWN threshold (value/threshold),
so all 8 counters (whose raw z/D/V scales differ hugely) sit on one shared
color scale: green well below threshold, yellow at 1.0x = the threshold
itself, red above it.

Columns, left to right:
    1. average-normal - mean over all 20 normal-mode seeds
    1.5. threshold strip - a single narrow column showing where each
         counter's own h_z/h_D/h_V threshold sits (always exactly 1.0x
         itself, so this strip is always the boundary color) - a visual
         anchor for "what counts as the threshold" next to the average-
         normal view it was calibrated from.
    2. TEST_SEED, jump mode - one single trial (not averaged).
    3. TEST_SEED, drift mode - one single trial, same seed as column 2.

Rows: z_stat (windowed max|z|), D (trend score), V (variability) - the
same three metrics/thresholds combined_detection.py already computes and
calibrates, read directly from its output (results/combined_detection_
windowed.csv, results/combined_detection_thresholds.csv) rather than
recomputed here.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TIMESERIES_PATH = os.path.join(RESULTS_DIR, "zscore_timeseries.csv")
WINDOWED_PATH = os.path.join(RESULTS_DIR, "combined_detection_windowed.csv")
THRESHOLD_PATH = os.path.join(RESULTS_DIR, "combined_detection_thresholds.csv")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots_heatmap")

TEST_SEED = "seed1"          # change this to look at a different seed
ATTACK_START = 150

ALL_COUNTERS = [f"hpmcounter{i}" for i in range(3, 11)]
METRICS = [("z_stat", "h_z", "z"), ("D", "h_D", "D"), ("V", "h_V", "V")]

VMAX = 3.0   # color scale caps at 3x the counter's own threshold
CMAP = "RdYlGn_r"   # green (low) -> yellow (1.0x threshold) -> red (high)
THRESHOLD_EPS = 1e-6   # thresholds at/below this are numerically degenerate


def find_constant_counters(timeseries_path, counters):
    """Counters whose raw normal-mode reading never changes (nunique<=1) --
    e.g. hpmcounter7 in this dataset, the same "flat counter" issue as hpm5
    in the original counter-selection study. Their z/D/V thresholds come
    out as 0/NaN (zero baseline variance), so they're dropped entirely
    rather than divided by a zero/NaN denominator."""
    ts = pd.read_csv(timeseries_path)
    normal = ts[(ts["mode"] == "normal") & (ts["counter"].isin(counters))]
    nunique = normal.groupby("counter")["raw_value"].nunique()
    return [c for c in counters if nunique.get(c, 0) <= 1]


def pivot_matrix(df, value_col, mode, counters, seed=None):
    d = df[df["mode"] == mode]
    if seed is not None:
        d = d[d["seed"] == seed]
        pivot = d.pivot(index="counter", columns="window_end_iter", values=value_col)
    else:
        avg = d.groupby(["counter", "window_end_iter"])[value_col].mean().reset_index()
        pivot = avg.pivot(index="counter", columns="window_end_iter", values=value_col)
    return pivot.reindex(counters)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    windowed = pd.read_csv(WINDOWED_PATH)
    thresholds_full = pd.read_csv(THRESHOLD_PATH).set_index("counter")

    constant = find_constant_counters(TIMESERIES_PATH, ALL_COUNTERS)
    if constant:
        print(f"Dropping constant counter(s) (no unique values in normal mode): {constant}")
    COUNTERS = [c for c in ALL_COUNTERS if c not in constant]
    thresholds = thresholds_full.reindex(COUNTERS)

    fig, axes = plt.subplots(
        3, 4, figsize=(20, 10),
        gridspec_kw={"width_ratios": [10, 1, 10, 10]},
    )
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=VMAX)

    col_titles = ["average normal (20 seeds)", "h", f"{TEST_SEED} jump", f"{TEST_SEED} drift"]

    im = None
    for row, (value_col, thresh_col, label) in enumerate(METRICS):
        thresh_vals = thresholds[thresh_col].to_numpy(dtype=float)

        avg_normal_df = pivot_matrix(windowed, value_col, "normal", COUNTERS)
        test_jump_df = pivot_matrix(windowed, value_col, "jump", COUNTERS, TEST_SEED)
        test_drift_df = pivot_matrix(windowed, value_col, "drift", COUNTERS, TEST_SEED)

        with np.errstate(divide="ignore", invalid="ignore"):
            thresh = thresh_vals.reshape(-1, 1)
            avg_normal = avg_normal_df.to_numpy() / thresh
            test_jump = test_jump_df.to_numpy() / thresh
            test_drift = test_drift_df.to_numpy() / thresh

        # Drop (not just mask) any counter whose threshold is degenerate for
        # this metric, OR whose actual data contains a NaN anywhere in any
        # of the three displayed views -- e.g. hpmcounter6's D is NaN for
        # its entire trial in 5/20 normal seeds (SE(beta)=0/0 when z is
        # exactly flat within a window), independent of its threshold being
        # numerically fine. Checked per-row, so a counter can be fine for
        # one metric and dropped for another.
        threshold_bad = ~np.isfinite(thresh_vals) | (thresh_vals <= THRESHOLD_EPS)
        data_has_nan = (
            np.isnan(avg_normal).any(axis=1)
            | np.isnan(test_jump).any(axis=1)
            | np.isnan(test_drift).any(axis=1)
        )
        drop = threshold_bad | data_has_nan
        if drop.any():
            dropped = [c for c, d in zip(COUNTERS, drop) if d]
            print(f"  {label}: dropping counter(s) with NaN data or degenerate threshold: {dropped}")

        row_counters = [c for c, d in zip(COUNTERS, drop) if not d]
        keep = ~drop
        avg_normal = avg_normal[keep]
        test_jump = test_jump[keep]
        test_drift = test_drift[keep]
        thresh_strip = np.ones((len(row_counters), 1))

        panels = [avg_normal, thresh_strip, test_jump, test_drift]
        x_extents = [
            (avg_normal_df.columns.min(), avg_normal_df.columns.max()),
            None,
            (test_jump_df.columns.min(), test_jump_df.columns.max()),
            (test_drift_df.columns.min(), test_drift_df.columns.max()),
        ]

        for col, (data, xext) in enumerate(zip(panels, x_extents)):
            ax = axes[row, col]
            extent = [xext[0], xext[1], len(row_counters), 0] if xext else [0, 1, len(row_counters), 0]
            im = ax.imshow(data, aspect="auto", cmap=CMAP, norm=norm, extent=extent)

            if col in (0, 2, 3) and ATTACK_START is not None and xext and xext[0] <= ATTACK_START <= xext[1]:
                ax.axvline(ATTACK_START, color="black", linewidth=1, linestyle="--", alpha=0.6)

            if col == 0:
                ax.set_yticks(np.arange(len(row_counters)) + 0.5)
                ax.set_yticklabels(row_counters, fontsize=8)
                ax.set_ylabel(f"{label}\ncounter", fontsize=10)
            else:
                ax.set_yticks([])

            if col == 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("EKF iteration", fontsize=8)

            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)

        # one colorbar per metric (row) -- numerically identical across
        # rows (all share the same norm), but visually tied to its own row
        fig.colorbar(im, ax=axes[row, :], shrink=0.85, pad=0.015, label=f"{label} / h_{label}")

    fig.suptitle("HPM attack detection: z / D / V, normalized per-counter by threshold", fontsize=13)

    out_path = os.path.join(PLOT_DIR, f"detection_heatmap_{TEST_SEED}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
