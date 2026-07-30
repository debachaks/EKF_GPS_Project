"""Plots mean z(t) +/- 95% CI (across 20 seeds), by mode, zoomed into the
early part of each run (before the ~167/300 attack onset), for every
counter - to visually check which counters separate from normal starting
near sample_index 0, well before any attack could have caused it.

Reads line_fitting_timeseries.csv (z per counter/mode/seed/sample_index).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TIMESERIES_PATH = os.path.join(SCRIPT_DIR, "line_fitting_timeseries.csv")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

ZOOM_SAMPLES = 100  # first N sample indices (out of 300), well before onset (~167)
MODES = ["normal", "drift", "jump", "replay"]
MODE_COLOR = {"normal": "#2a78d6", "drift": "#1baf7a", "jump": "#eda100", "replay": "#c0392b"}

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
    ax.axhline(0, color=AXIS_COLOR, linewidth=1)
    ax.set_axisbelow(True)


def main():
    ts = pd.read_csv(TIMESERIES_PATH)
    ts = ts[ts["sample_index"] < ZOOM_SAMPLES]
    counters = sorted(ts["counter"].unique(), key=lambda c: int(c.replace("hpmcounter", "")))

    os.makedirs(PLOT_DIR, exist_ok=True)

    for counter in counters:
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
        first_sig_by_mode = {}

        for mode in MODES:
            sub = ts[(ts.counter == counter) & (ts["mode"] == mode)]
            grouped = sub.groupby("sample_index")["z"]
            mean_z = grouped.mean()
            n = grouped.count()
            sem_z = grouped.std() / np.sqrt(n)
            ci95 = sem_z * stats.t.ppf(0.975, n - 1)

            x = mean_z.index.to_numpy()
            ax.plot(x, mean_z.to_numpy(), color=MODE_COLOR[mode], linewidth=2,
                     label=f"{mode} (n={n.iloc[0]})")
            ax.fill_between(x, (mean_z - ci95).to_numpy(), (mean_z + ci95).to_numpy(),
                             color=MODE_COLOR[mode], alpha=0.15, linewidth=0)

            if mode != "normal":
                sig = None
                for idx, grp in sub.groupby("sample_index"):
                    z = grp["z"].to_numpy()
                    if len(z) < 2:
                        continue
                    _, p = stats.ttest_1samp(z, 0.0)
                    if p < 0.05:
                        sig = idx
                        break
                first_sig_by_mode[mode] = sig
                if sig is not None:
                    ax.axvline(sig, color=MODE_COLOR[mode], linestyle=":", linewidth=1, alpha=0.6)

        title_bits = ", ".join(f"{m} sig@{s}" for m, s in first_sig_by_mode.items() if s is not None)
        ax.set_title(f"{counter}: mean z(t) +/- 95% CI, first {ZOOM_SAMPLES} samples\n"
                      f"(first significant divergence from 0: {title_bits})",
                      color=TEXT_PRIMARY, fontsize=11)
        ax.set_xlabel("sample_index (0..299 grid; attack onset is ~167)", color=TEXT_MUTED)
        ax.set_ylabel("z = (x - mu_normal) / sigma_normal", color=TEXT_MUTED)
        ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
        style_axes(ax)
        fig.tight_layout()

        out_path = os.path.join(PLOT_DIR, f"{counter}_early_divergence.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
