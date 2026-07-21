"""Plot how much normal/drift/jump/replay separate on the FP registers flagged
by seed_new_fp_register_analysis.py.

Two figures:
  - constant_registers_by_mode.png: grouped bars for ft1-ft4 (registers that
    are a single fixed value for the whole run, differing by mode).
  - variable_registers_by_mode.png: box+strip distributions for the 7
    row-varying registers (fs0, fa0-fa5), one subplot each, so the
    statistically significant ones (fa0, fa1) can be compared against the
    non-significant ones by eye.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from seed_new_fp_register_analysis import DOUBLE_REGS, MODES, load_mode

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SRC_DIR, "seed_new_fp_plots")

# Fixed categorical color per mode, consistent across every chart.
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

CONSTANT_REGS = ["ft1", "ft2", "ft3", "ft4"]
VARYING_REGS = ["fs0", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5"]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_MUTED)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)


def plot_constant_registers(data):
    fig, axes = plt.subplots(1, len(CONSTANT_REGS), figsize=(16, 4), facecolor=SURFACE)
    fig.suptitle(
        "seed_new: fixed per-run value by mode (registers that never change within a run)",
        color=TEXT_PRIMARY, fontsize=12,
    )
    for ax, reg in zip(axes, CONSTANT_REGS):
        values = [data[mode][reg].iloc[0] for mode in MODES]
        colors = [MODE_COLOR[mode] for mode in MODES]
        bars = ax.bar(MODES, values, color=colors, width=0.6)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8, color=TEXT_PRIMARY,
            )
        ax.set_title(reg, color=TEXT_PRIMARY, fontsize=11)
        style_axes(ax)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(PLOT_DIR, "constant_registers_by_mode.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_variable_registers(data, sig_lookup):
    ncols = 4
    nrows = int(np.ceil(len(VARYING_REGS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), facecolor=SURFACE)
    axes = np.atleast_2d(axes)
    fig.suptitle(
        "seed_new: row-level distribution by mode (registers that vary within a run)",
        color=TEXT_PRIMARY, fontsize=12,
    )

    rng = np.random.default_rng(0)
    for idx, reg in enumerate(VARYING_REGS):
        ax = axes[idx // ncols][idx % ncols]
        series = [data[mode][reg].to_numpy() for mode in MODES]

        bp = ax.boxplot(
            series, positions=range(len(MODES)), widths=0.5, patch_artist=True,
            showfliers=False, medianprops=dict(color=TEXT_PRIMARY, linewidth=2),
        )
        for patch, mode in zip(bp["boxes"], MODES):
            patch.set_facecolor(MODE_COLOR[mode])
            patch.set_alpha(0.35)
            patch.set_edgecolor(MODE_COLOR[mode])
        for element in ["whiskers", "caps"]:
            for line in bp[element]:
                line.set_color(TEXT_MUTED)

        for i, (mode, vals) in enumerate(zip(MODES, series)):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter, vals,
                s=6, color=MODE_COLOR[mode], alpha=0.5, linewidths=0,
            )

        sig_flags = "".join(
            "*" if sig_lookup.get((reg, m)) else "" for m in ["drift", "jump", "replay"]
        )
        title = reg + (f"  {sig_flags}" if sig_flags else "")
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=11)
        ax.set_xticks(range(len(MODES)))
        ax.set_xticklabels(MODES, rotation=20)
        style_axes(ax)

    for idx in range(len(VARYING_REGS), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.text(
        0.5, 0.005,
        "* = significant vs normal after FDR correction (p_fdr < 0.05), single run per mode",
        ha="center", color=TEXT_MUTED, fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    out_path = os.path.join(PLOT_DIR, "variable_registers_by_mode.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    data = {mode: load_mode(mode) for mode in MODES}

    # Significance flags, recomputed the same way as seed_new_fp_register_analysis.py
    import pandas as pd
    from scipy.stats import mannwhitneyu
    from hpmcounter_analysis import benjamini_hochberg, cliffs_delta

    rows = []
    normal = data["normal"]
    for mode in ["drift", "jump", "replay"]:
        for reg in VARYING_REGS:
            x = data[mode][reg].to_numpy()
            y = normal[reg].to_numpy()
            _, p_value = mannwhitneyu(x, y, alternative="two-sided")
            rows.append({"register": reg, "mode": mode, "p_value": p_value})
    results = pd.DataFrame(rows)
    results["p_value_fdr"] = benjamini_hochberg(results["p_value"].to_numpy())
    sig_lookup = {
        (row.register, row.mode): row.p_value_fdr < 0.05
        for row in results.itertuples()
    }

    plot_constant_registers(data)
    plot_variable_registers(data, sig_lookup)


if __name__ == "__main__":
    main()
