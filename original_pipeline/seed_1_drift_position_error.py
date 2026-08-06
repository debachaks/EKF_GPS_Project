"""Estimated position minus true position (filt - true), per axis, over
time for the seed_1 drift/jump/replay runs. One plot per mode."""

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "seed_1_data")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

MODES = ["drift", "jump", "replay"]
AXIS_COLOR_MAP = {"x": "#2a78d6", "y": "#eda100", "z": "#1baf7a"}

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

    for mode in MODES:
        df = pd.read_csv(os.path.join(DATA_DIR, f"ekf_diag_{mode}.csv"))
        for axis in ["x", "y", "z"]:
            df[f"err_{axis}"] = df[f"filt_{axis}"] - df[f"true_{axis}"]

        fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)

        for axis in ["x", "y", "z"]:
            ax.plot(df["t"], df[f"err_{axis}"], color=AXIS_COLOR_MAP[axis], linewidth=1.5, label=f"{axis} error")

        ax.axhline(0, color=AXIS_COLOR, linewidth=1, linestyle="--")
        ax.set_title(f"seed_1 {mode} run: estimated - true position over time", color=TEXT_PRIMARY, fontsize=12)
        ax.set_xlabel("t (entry index)", color=TEXT_MUTED)
        ax.set_ylabel("position error (filt - true)", color=TEXT_MUTED)
        ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
        style_axes(ax)
        fig.tight_layout()

        out_path = os.path.join(PLOT_DIR, f"seed_1_{mode}_position_error.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
