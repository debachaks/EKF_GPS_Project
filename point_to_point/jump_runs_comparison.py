"""Iteration vs. raw counter value for the two jump runs in point_to_point/:
jump_run_full.csv vs jump_run_2_full.csv, one plot per counter
(hpmcounter3, hpmcounter5).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

COUNTERS = ["hpmcounter3", "hpmcounter5"]
RUNS = ["jump_run_full", "jump_run_2_full"]
RUN_LABEL = {"jump_run_full": "jump run 1", "jump_run_2_full": "jump run 2"}
RUN_COLOR = {"jump_run_full": "#eda100", "jump_run_2_full": "#2a78d6"}

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"


def load(run):
    path = os.path.join(DATA_DIR, f"{run}.csv")
    df = pd.read_csv(path)
    for counter in COUNTERS:
        df[counter] = df[counter].apply(lambda x: int(x, 16))
    return df


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
    data = {run: load(run) for run in RUNS}

    for counter in COUNTERS:
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)

        for run in RUNS:
            df = data[run]
            ax.plot(df["iter"], df[counter], color=RUN_COLOR[run], linewidth=1.5, label=RUN_LABEL[run])

        ax.set_title(f"{counter}: iteration vs. value (jump runs)", color=TEXT_PRIMARY, fontsize=12)
        ax.set_xlabel("iteration", color=TEXT_MUTED)
        ax.set_ylabel(counter, color=TEXT_MUTED)
        ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
        style_axes(ax)
        fig.tight_layout()

        out_path = os.path.join(PLOT_DIR, f"{counter}_jump_runs.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
