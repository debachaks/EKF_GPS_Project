"""hpmcounter3-10 vs. iteration (raw, and per-iteration rate) for the STF
firmware's point-to-point runs (plots/*_point_to_point/<mode>_hpc.csv),
normal/jump/drift overlaid per counter.

Unlike plot_hpmcounters.py's periodic register-dump captures, these are
one row per EKF iteration (aligned "iter" column, same point in execution
across modes), so iteration index is used directly as the x-axis - no
elapsed-time/timestamp involved.

The raw counters are cumulative, so an attack's effect (if any) shows up
as a change in slope, invisible on the raw plot at full-run scale (see
raw plot_column output). The rate plot differentiates each counter
against iteration (counts per iteration, dt=1 exactly - no JTAG timing
jitter to divide out here, unlike plot_hpmcounters_rate.py) so a
per-iteration change stands out directly. A rolling mean
(ROLLING_WINDOW samples) is applied on top of the raw rate, same idea as
plot_hpmcounters_rate.py.

Runs over every plots/*_point_to_point/ folder found. Replay is
intentionally excluded even if present, so every folder is plotted the
same way.
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIRS = sorted(glob.glob(os.path.join(SCRIPT_DIR, "plots", "*_point_to_point")))

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


def load(run_dir, mode):
    path = os.path.join(run_dir, f"{mode}_hpc.csv")
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


def plot_column(data, run_name, column, ylabel, title_suffix, out_path):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)

    for mode in MODES:
        df = data[mode]
        ax.plot(df["iter"], df[column], color=MODE_COLOR[mode], linewidth=1.5, label=mode)

    ax.set_title(f"{run_name}: {ylabel}{title_suffix}", color=TEXT_PRIMARY, fontsize=12)
    ax.set_xlabel("iteration", color=TEXT_MUTED)
    ax.set_ylabel(ylabel, color=TEXT_MUTED)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
    style_axes(ax)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    for run_dir in RUN_DIRS:
        run_name = os.path.basename(run_dir)
        rate_dir = os.path.join(run_dir, "rate")
        os.makedirs(rate_dir, exist_ok=True)
        data = {mode: load(run_dir, mode) for mode in MODES}

        for counter in COUNTERS:
            plot_column(
                data, run_name, counter, counter,
                " vs. iteration", os.path.join(run_dir, f"{counter}.png"),
            )
            plot_column(
                data, run_name, f"{counter}_rate", f"{counter} (counts/iteration)",
                " rate vs. iteration", os.path.join(rate_dir, f"{counter}_rate.png"),
            )


if __name__ == "__main__":
    main()
