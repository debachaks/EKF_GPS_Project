"""hpmcounter3-10 vs. elapsed time for the STF firmware's real hardware
runs (ekf_<mode>_hpc.csv), normal/jump/drift overlaid per counter.

These are raw periodic register-dump captures (reg_record.py style --
timestamp_ms + the full CSR/register file each sample), not one row
per EKF step, so elapsed time (ms since the run's first sample) is
used as the x-axis rather than iteration index.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

COUNTERS = [f"hpmcounter{i}" for i in range(3, 11)]
MODES = ["normal", "jump", "drift"]   # no replay HPC capture for this STF run
MODE_COLOR = {"normal": "#2a78d6", "jump": "#eda100", "drift": "#1baf7a"}

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"


def hex_to_int(val):
    s = str(val).strip()
    if s.lower().startswith("0x"):
        return int(s, 16)
    return int(float(s))


def load(mode):
    path = os.path.join(DATA_DIR, f"ekf_{mode}_hpc.csv")
    df = pd.read_csv(path)
    df["timestamp_ms"] = df["timestamp_ms"].map(hex_to_int)
    elapsed = df["timestamp_ms"] - df["timestamp_ms"].iloc[0]
    for counter in COUNTERS:
        df[counter] = df[counter].map(hex_to_int)
    df["elapsed_ms"] = elapsed
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
    data = {mode: load(mode) for mode in MODES}

    for counter in COUNTERS:
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)

        for mode in MODES:
            df = data[mode]
            ax.plot(df["elapsed_ms"], df[counter], color=MODE_COLOR[mode], linewidth=1.5, label=mode)

        ax.set_title(f"STF firmware: {counter} over time", color=TEXT_PRIMARY, fontsize=12)
        ax.set_xlabel("elapsed time (ms since run start)", color=TEXT_MUTED)
        ax.set_ylabel(counter, color=TEXT_MUTED)
        ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
        style_axes(ax)
        fig.tight_layout()

        out_path = os.path.join(PLOT_DIR, f"{counter}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
