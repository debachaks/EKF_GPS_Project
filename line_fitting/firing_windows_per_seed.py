"""Firing-window plots, one per (counter, attack_type, seed) - no
cross-seed aggregation, each plot shows exactly one run's own behavior.

4 layers, stacked bottom to top:
    z_flag      - raw-z windowed metric fired at this window
    d_flag      - D (trend) metric fired at this window
    v_flag      - V (variability) metric fired at this window
    flag_and    - "attack alarm": this window's z_flag AND d_flag AND v_flag

X axis is window index (0, 1, 2, ...) within this run, not the raw sample
index.

Reads combined_detection_flags.csv (from combined_detection.py).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FLAGS_PATH = os.path.join(SCRIPT_DIR, "combined_detection_flags.csv")
PLOT_DIR = os.path.join(SCRIPT_DIR, "firing_windows_plots", "per_seed")

ATTACK_MODES = ["drift", "jump", "replay"]
LAYERS = [
    ("z_flag", "z"),
    ("d_flag", "D"),
    ("v_flag", "V"),
    ("flag_and", "attack alarm\n(AND of 3)"),
]
LAYER_COLOR = {
    "z_flag": "#2a78d6",
    "d_flag": "#1baf7a",
    "v_flag": "#eda100",
    "flag_and": "#c0392b",
}
LAYER_HEIGHT = 0.8
LAYER_GAP = 0.4

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_MUTED)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)


def add_window_index(df):
    df = df.sort_values(["counter", "mode", "seed", "window_end_sample_index"])
    df["window_idx"] = df.groupby(["counter", "mode", "seed"]).cumcount()
    return df


def plot_one(counter, mode, seed, sub, out_path):
    sub = sub.sort_values("window_idx")
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)

    for layer_i, (col, label) in enumerate(LAYERS):
        x = sub["window_idx"].to_numpy()
        y_bin = sub[col].to_numpy().astype(float)
        offset = layer_i * (LAYER_HEIGHT + LAYER_GAP)
        y = offset + y_bin * LAYER_HEIGHT
        ax.step(x, y, where="post", color=LAYER_COLOR[col], linewidth=1.5)
        ax.fill_between(x, offset, y, step="post", color=LAYER_COLOR[col], alpha=0.35)
        ax.axhline(offset, color=AXIS_COLOR, linewidth=0.8)

    yticks = [i * (LAYER_HEIGHT + LAYER_GAP) + LAYER_HEIGHT / 2 for i in range(len(LAYERS))]
    ax.set_yticks(yticks)
    ax.set_yticklabels([label for _col, label in LAYERS], color=TEXT_PRIMARY, fontsize=9)
    ax.set_ylim(-0.2, len(LAYERS) * (LAYER_HEIGHT + LAYER_GAP))

    ax.set_xlabel("window index", color=TEXT_MUTED)
    ax.set_title(f"{counter}: {mode} - {seed}", color=TEXT_PRIMARY, fontsize=12)
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    df = pd.read_csv(FLAGS_PATH)
    df = add_window_index(df)

    os.makedirs(PLOT_DIR, exist_ok=True)
    counters = sorted(df["counter"].unique(), key=lambda c: int(c.replace("hpmcounter", "")))

    for counter in counters:
        out_dir = os.path.join(PLOT_DIR, counter)
        os.makedirs(out_dir, exist_ok=True)
        for mode in ATTACK_MODES:
            seeds = sorted(df.loc[(df["counter"] == counter) & (df["mode"] == mode), "seed"].unique())
            for seed in seeds:
                sub = df[(df["counter"] == counter) & (df["mode"] == mode) & (df["seed"] == seed)]
                out_path = os.path.join(out_dir, f"{counter}_{mode}_{seed}.png")
                plot_one(counter, mode, seed, sub, out_path)


if __name__ == "__main__":
    main()
