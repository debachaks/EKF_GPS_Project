"""p_trace (trace of the posterior covariance P) over time for seed_1,
all four modes overlaid. Kalman gain itself is not logged; p_trace is the
closest available proxy for filter uncertainty/gain behavior."""

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "seed_1_data")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

MODES = ["normal", "jump", "drift", "replay"]
MODE_COLOR = {"normal": "#2a78d6", "jump": "#eda100", "drift": "#1baf7a", "replay": "#c0392b"}

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"


def load(mode):
    return pd.read_csv(os.path.join(DATA_DIR, f"ekf_diag_{mode}.csv"))


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

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)

    for mode in MODES:
        df = load(mode)
        ax.plot(df["t"], df["p_trace"], color=MODE_COLOR[mode], linewidth=1.5, label=mode)

    ax.set_title("seed_1: p_trace over entries, all modes", color=TEXT_PRIMARY, fontsize=12)
    ax.set_xlabel("t (entry index)", color=TEXT_MUTED)
    ax.set_ylabel("p_trace (trace of P)", color=TEXT_MUTED)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
    style_axes(ax)
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, "seed_1_p_trace_all_modes.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
