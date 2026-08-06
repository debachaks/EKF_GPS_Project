"""NIS values over time for the seed_1 jump run."""

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "seed_1_data", "ekf_diag_jump.csv")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

LINE_COLOR = "#eda100"
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
    df = pd.read_csv(DATA_PATH)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.plot(df["t"], df["nis"], color=LINE_COLOR, linewidth=1.5, label="jump")

    ax.set_title("seed_1 jump run: NIS over entries", color=TEXT_PRIMARY, fontsize=12)
    ax.set_xlabel("t (entry index)", color=TEXT_MUTED)
    ax.set_ylabel("nis", color=TEXT_MUTED)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
    style_axes(ax)
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, "seed_1_nis_jump.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
