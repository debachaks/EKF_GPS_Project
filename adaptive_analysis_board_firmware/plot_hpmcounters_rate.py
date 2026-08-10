"""hpmcounter3-10 RATE (counts/sec) vs. elapsed time, for the STF
firmware's real hardware runs (ekf_<mode>_hpc.csv), normal/jump/drift
overlaid per counter.

The raw counters are cumulative (never reset), so an attack's effect
shows up as a change in SLOPE, not a spike -- see plot_hpmcounters.py.
This script differentiates each counter against elapsed time (counts
per second between consecutive samples) so the attack window, if it
changes the counter's growth rate, shows up as a visible bump/step
right at ATTACK_START instead of being spread across the whole run.

Raw sample-to-sample deltas are very noisy -- each register-dump
snapshot lands at an essentially random point within the current EKF
loop iteration, so successive deltas capture wildly varying partial
iterations. A rolling mean (ROLLING_WINDOW samples) is applied on top
of the raw rate to reveal the underlying trend, same idea as the
firmware's own ANIS window smoothing raw per-step NIS.

Runs over every plots/seed*/ folder - each seed's raw data csvs and its
plots live together in that same folder. Replay is intentionally
excluded even for seeds that have replay data (e.g. seed2), so all
seeds are plotted the same way.
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED_DIRS = sorted(glob.glob(os.path.join(SCRIPT_DIR, "plots", "seed*")))

ROLLING_WINDOW = 15

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
    path = os.path.join(data_dir, f"ekf_{mode}_hpc.csv")
    df = pd.read_csv(path)
    df["timestamp_ms"] = df["timestamp_ms"].map(hex_to_int)
    df["elapsed_ms"] = df["timestamp_ms"] - df["timestamp_ms"].iloc[0]
    for counter in COUNTERS:
        df[counter] = df[counter].map(hex_to_int)

    dt_s = df["elapsed_ms"].diff() / 1000.0
    for counter in COUNTERS:
        raw_rate = df[counter].diff() / dt_s
        df[f"{counter}_rate"] = raw_rate.rolling(ROLLING_WINDOW, center=True).mean()

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
                ax.plot(df["elapsed_ms"], df[f"{counter}_rate"], color=MODE_COLOR[mode], linewidth=1.2, label=mode)

            ax.set_title(f"STF firmware ({seed_name}): {counter} rate over time", color=TEXT_PRIMARY, fontsize=12)
            ax.set_xlabel("elapsed time (ms since run start)", color=TEXT_MUTED)
            ax.set_ylabel(f"{counter} (counts/sec)", color=TEXT_MUTED)
            ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
            style_axes(ax)
            fig.tight_layout()

            out_path = os.path.join(plot_dir, f"{counter}_rate.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
