"""Compare a rerun against its original run: new_test_1/ vs seed_old/test_seed_1/.

Both are raw (uncleaned) board dumps with the exact same column schema, so they
can be compared directly. For each of the 8 hpmcounters, one figure with a
2x2 grid (normal/jump/drift/replay) overlays the original run (solid) against
the rerun (dashed).

Plots raw hpmcounter values as-is (no baselining). Note hpmcounters are NOT
reset at board start (only mcycle/minstret are, see main_ekf.c), so their
absolute starting value carries an arbitrary offset left over from whatever
ran on the board before, and that offset can differ between the original run
and the rerun.

Also writes a small numeric summary (raw start/end values per run) so the
plots can be sanity-checked against actual numbers.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
while not os.path.isdir(os.path.join(PROJECT_ROOT, "original_pipeline")):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from data_preprocessing import hex_to_int  # noqa: E402

ORIGINAL_DIR = os.path.join(PROJECT_ROOT, "seed_old", "test_seed_1")
RERUN_DIR = os.path.join(PROJECT_ROOT, "new_test_1")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots", "rerun_comparison_test_seed_1")

MODES = ["normal", "drift", "jump", "replay"]
MODE_TITLE = {"normal": "normal", "drift": "drift", "jump": "jump", "replay": "replay"}
COUNTERS = [f"hpmcounter{i}" for i in [3, 4, 5, 6, 7, 8, 9, 10]]

ORIGINAL_COLOR = "#2a78d6"
RERUN_COLOR = "#c0392b"

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"


def load_trace(run_dir, mode, counter):
    path = os.path.join(run_dir, f"ekf_{mode}_hpc.csv")
    df = pd.read_csv(path)
    ts = df["timestamp_ms"].map(hex_to_int)
    elapsed = (ts - ts.iloc[0]).to_numpy()
    values = df[counter].map(hex_to_int).to_numpy()
    return elapsed, values


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    summary_rows = []

    for counter in COUNTERS:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)
        fig.suptitle(f"{counter}: test_seed_1 original vs rerun (new_test_1)",
                     color=TEXT_PRIMARY, fontsize=13)

        for ax, mode in zip(axes.flat, MODES):
            orig_t, orig_v = load_trace(ORIGINAL_DIR, mode, counter)
            new_t, new_v = load_trace(RERUN_DIR, mode, counter)

            ax.plot(orig_t, orig_v, color=ORIGINAL_COLOR, linewidth=1.8, label="original")
            ax.plot(new_t, new_v, color=RERUN_COLOR, linewidth=1.5, linestyle="--", label="rerun")

            ax.set_title(MODE_TITLE[mode], color=TEXT_PRIMARY, fontsize=10)
            ax.set_xlabel("elapsed time (ms)", color=TEXT_MUTED, fontsize=8)
            ax.set_ylabel(counter, color=TEXT_MUTED, fontsize=8)
            ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=8)
            style_axes(ax)

            summary_rows.append({
                "counter": counter,
                "mode": mode,
                "original_start": int(orig_v[0]),
                "original_end": int(orig_v[-1]),
                "rerun_start": int(new_v[0]),
                "rerun_end": int(new_v[-1]),
                "original_n_samples": len(orig_v),
                "rerun_n_samples": len(new_v),
            })

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = os.path.join(PLOT_DIR, f"{counter}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(PLOT_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved numeric summary -> {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
