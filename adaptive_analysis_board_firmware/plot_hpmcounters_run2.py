"""hpmcounter3-10 vs. elapsed time for the SECOND repeat run of each mode
(plots/seed*/ekf_<mode>_hpc_2.csv), normal/jump/drift overlaid per
counter. Output filenames get a "_2" suffix so they sit alongside the
first run's plots without overwriting them.

Repeat-run counterpart to plot_hpmcounters.py -- same raw periodic
register-dump format (timestamp_ms + full CSR/register file per
sample), added to check whether a pattern seen in the first run
reproduces on a second, independent run of the same seed.

Runs over every plots/seed*/ folder that has an ekf_normal_hpc_2.csv in
it, so it only touches seeds where a repeat run actually exists.
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED_DIRS = sorted(
    os.path.dirname(p) for p in
    glob.glob(os.path.join(SCRIPT_DIR, "plots", "seed*", "ekf_normal_hpc_2.csv"))
)

COUNTERS = [f"hpmcounter{i}" for i in range(3, 11)]
MODES = ["normal", "jump", "drift"]   # replay intentionally excluded
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


def load(data_dir, mode):
    path = os.path.join(data_dir, f"ekf_{mode}_hpc_2.csv")
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
    for seed_dir in SEED_DIRS:
        seed_name = os.path.basename(seed_dir)
        data = {mode: load(seed_dir, mode) for mode in MODES}

        for counter in COUNTERS:
            fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)

            for mode in MODES:
                df = data[mode]
                ax.plot(df["elapsed_ms"], df[counter], color=MODE_COLOR[mode], linewidth=1.5, label=mode)

            ax.set_title(f"STF firmware ({seed_name}, run 2): {counter} over time", color=TEXT_PRIMARY, fontsize=12)
            ax.set_xlabel("elapsed time (ms since run start)", color=TEXT_MUTED)
            ax.set_ylabel(counter, color=TEXT_MUTED)
            ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
            style_axes(ax)
            fig.tight_layout()

            out_path = os.path.join(seed_dir, f"{counter}_2.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
