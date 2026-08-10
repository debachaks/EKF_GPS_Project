"""hpmcounter3-10 vs. iteration (raw, and per-iteration rate) for the
SECOND repeat run of each mode (plots/*/<mode>_hpc_2.csv), normal/jump/
drift overlaid per counter. Output filenames get a "_2" suffix so they
sit alongside the first run's plots without overwriting them.

This is the repeat-run counterpart to point_to_point_traces.py -- added
to check whether a pattern seen in the first run (e.g. normal sitting
above jump/drift on hpmcounter5) reproduces on a second, independent
run of the same trajectories, or whether it was run-to-run board noise.

Runs over every plots/*/ folder that has a normal_hpc_2.csv in it, same
file-presence-based discovery as point_to_point_traces.py.
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIRS = sorted(
    os.path.dirname(p) for p in
    glob.glob(os.path.join(SCRIPT_DIR, "plots", "*", "normal_hpc_2.csv"))
)

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
    path = os.path.join(run_dir, f"{mode}_hpc_2.csv")
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

    ax.set_title(f"{run_name} (run 2): {ylabel}{title_suffix}", color=TEXT_PRIMARY, fontsize=12)
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
                " vs. iteration", os.path.join(run_dir, f"{counter}_2.png"),
            )
            plot_column(
                data, run_name, f"{counter}_rate", f"{counter} (counts/iteration)",
                " rate vs. iteration", os.path.join(rate_dir, f"{counter}_rate_2.png"),
            )


if __name__ == "__main__":
    main()
