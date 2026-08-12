"""hpmcounter3-10 RATE (counts/iteration) vs. EKF iteration, for the
exclusive-branch ANIS-gated STF firmware's captures
(plots/seed*_new/<mode>_hpc.csv), normal/jump/drift overlaid per
counter.

Rate counterpart to plot_hpmcounters_new.py. These captures have
exactly one row per EKF iteration (not a sub-sampled register dump),
so the sample-to-sample delta already IS the per-iteration rate --
no elapsed-time division needed. A light rolling mean (ROLLING_WINDOW
samples) is still applied since each individual step's rate can
still be noisy (e.g. a division only happening on some iterations
of the exclusive branch).

Runs over every plots/seed*/ folder that has a normal_hpc.csv in it
(not ekf_<mode>_hpc.csv -- that filename marks the older timestamp_ms
register-dump format handled by plot_hpmcounters_rate.py instead), so
it only touches seeds captured in this iter-indexed format, regardless
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

ROLLING_WINDOW = 5

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
        df[f"{counter}_rate"] = df[counter].diff().rolling(ROLLING_WINDOW, center=True).mean()
    return df.iloc[1:]   # drop first row -- no prior sample to diff against


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
        plot_dir = os.path.join(seed_dir, "rate")
        os.makedirs(plot_dir, exist_ok=True)
        data = {mode: load(seed_dir, mode) for mode in MODES}

        for counter in COUNTERS:
            fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)

            for mode in MODES:
                df = data[mode]
                ax.plot(df["iter"], df[f"{counter}_rate"], color=MODE_COLOR[mode], linewidth=1.2, label=mode)

            ax.set_title(f"STF firmware ({seed_name}): {counter} rate over EKF iteration", color=TEXT_PRIMARY, fontsize=12)
            ax.set_xlabel("EKF iteration", color=TEXT_MUTED)
            ax.set_ylabel(f"{counter} (counts/iteration)", color=TEXT_MUTED)
            ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
            style_axes(ax)
            fig.tight_layout()

            out_path = os.path.join(plot_dir, f"{counter}_rate.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
