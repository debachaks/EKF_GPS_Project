"""Absolute-difference trace plots for a single seed: one PNG per (seed,
counter), plotting |normal - drift|, |normal - jump|, |normal - replay| over
elapsed time.

Reuses test_seed_raw_trace_per_seed.py's per-mode (elapsed_ms, value) traces.
Since normal/drift/jump/replay are independent runs with slightly different
row counts and elapsed-time grids, each mode's trace is linearly interpolated
onto a shared time grid (0 to the shortest run's elapsed time, so all four
modes have real data to interpolate from) before taking the difference -
counters are monotonically increasing, so interpolation between samples is
a safe way to estimate the value at a time no mode happened to sample.

Output: test_seed_kde_plots/<counter>_per_seed/<seed>_<counter>_diff_trace.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from test_seed_raw_trace_per_seed import (  # noqa: E402
    MODE_COLOR,
    PLOT_ROOT,
    SURFACE,
    find_counters,
    find_seed_dirs,
    seed_trace,
    style_axes,
)

TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
ANOMALY_TYPES = ["drift", "jump", "replay"]


def diff_trace(seed_dir, counter):
    traces = {mode: seed_trace(seed_dir, mode, counter) for mode in ["normal"] + ANOMALY_TYPES}
    grid_end = min(elapsed.max() for elapsed, _ in traces.values())
    grid = np.linspace(0, grid_end, 500)

    interp = {
        mode: np.interp(grid, elapsed, values)
        for mode, (elapsed, values) in traces.items()
    }
    diffs = {anomaly: np.abs(interp[anomaly] - interp["normal"]) for anomaly in ANOMALY_TYPES}
    return grid, diffs


def plot_seed_counter_diff(seed_name, counter, grid, diffs, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    for anomaly in ANOMALY_TYPES:
        ax.plot(grid, diffs[anomaly], color=MODE_COLOR[anomaly], linewidth=1.5, label=f"|normal - {anomaly}|")

    ax.set_title(f"{counter}: |normal - attack| over elapsed time - {seed_name} only",
                 color=TEXT_PRIMARY, fontsize=12)
    ax.set_xlabel("elapsed time (ms since run start)", color=TEXT_MUTED)
    ax.set_ylabel("absolute counter value difference", color=TEXT_MUTED)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    style_axes(ax)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{seed_name}_{counter}_diff_trace.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    seed_dirs = find_seed_dirs()
    if not seed_dirs:
        print("No cleaned test_seed_N folders found - run test_seed_clean_hpc.py first")
        return

    counters = find_counters(seed_dirs)
    print(f"hpmcounters present after cleaning: {counters}")

    for counter in counters:
        out_dir = os.path.join(PLOT_ROOT, f"{counter}_per_seed")
        os.makedirs(out_dir, exist_ok=True)
        for seed_dir in seed_dirs:
            seed_name = os.path.basename(seed_dir)
            grid, diffs = diff_trace(seed_dir, counter)
            plot_seed_counter_diff(seed_name, counter, grid, diffs, out_dir)


if __name__ == "__main__":
    main()
