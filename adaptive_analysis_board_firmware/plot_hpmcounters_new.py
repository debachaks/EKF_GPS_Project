"""hpmcounter3-10 vs. EKF iteration, for the exclusive-branch ANIS-gated
STF firmware's captures (plots/seed*_new/<mode>_hpc.csv), normal/jump/
drift overlaid per counter.

Unlike plot_hpmcounters.py's data (a periodic register-dump with many
samples per EKF step, keyed on timestamp_ms), these captures have
exactly one row per EKF iteration (iter, mcycle, minstret, pc,
hpmcounter3..10 -- iter decimal, the rest hex), so iteration index is
used directly as the x-axis instead of elapsed time.

Runs over every plots/seed*/ folder that has a normal_hpc.csv in it
(not ekf_<mode>_hpc.csv -- that filename marks the older timestamp_ms
register-dump format handled by plot_hpmcounters.py instead), so it
only touches seeds captured in this iter-indexed format, regardless
of naming suffix (seed1_new, seed2_mapping2, etc).
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED_DIRS = sorted(
    os.path.dirname(p) for p in
    glob.glob(os.path.join(SCRIPT_DIR, "plots", "seed*", "normal_hpc.csv"))
)

COUNTERS = [f"hpmcounter{i}" for i in range(3, 11)]
MODES = ["normal", "jump", "drift"]
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
    path = os.path.join(data_dir, f"{mode}_hpc.csv")
    df = pd.read_csv(path)
    for counter in COUNTERS:
        df[counter] = df[counter].map(hex_to_int)
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
                ax.plot(df["iter"], df[counter], color=MODE_COLOR[mode], linewidth=1.5, label=mode)

            ax.set_title(f"STF firmware ({seed_name}): {counter} over EKF iteration", color=TEXT_PRIMARY, fontsize=12)
            ax.set_xlabel("EKF iteration", color=TEXT_MUTED)
            ax.set_ylabel(counter, color=TEXT_MUTED)
            ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
            style_axes(ax)
            fig.tight_layout()

            out_path = os.path.join(seed_dir, f"{counter}.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
