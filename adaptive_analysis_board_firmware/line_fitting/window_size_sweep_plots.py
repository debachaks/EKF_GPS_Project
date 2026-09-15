"""Window-size justification plots for D, G, V: one plot per metric,
x-axis = swept window size W, y-axis = mean number of trials detected
(out of 20 seeds), one line for jump and one for drift.

Averaged across the 5 well-behaved counters (hpmcounter3/4/5/8/10) --
hpmcounter9 excluded, same convention as Table 3/4 in the manuscript,
since it is the established near-negative-control outlier and its
window-size trend runs in the OPPOSITE direction (its recall INCREASES
with window size for the D metric), which would muddy the justification
these plots are meant to give for the chosen W=10 (D) / W=5 (G, V).

Source data: results/{d,g,v}_final_sweep.csv (per-counter, per-window,
per-mode detected-count-out-of-20, from d_final_sweep.py / g_final_sweep.py
/ v_final_sweep.py).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots_heatmap")

FIVE_COUNTERS = ["hpmcounter3", "hpmcounter4", "hpmcounter5", "hpmcounter8", "hpmcounter10"]
MODE_COLORS = {"jump": "#E07B1A", "drift": "#2E8B3D"}
CHOSEN_WINDOW = {"D": 10, "G": 5, "V": 5}

METRICS = [
    ("D", "results/d_final_sweep.csv"),
    ("G", "results/g_final_sweep.csv"),
    ("V", "results/v_final_sweep.csv"),
]


def make_plot(metric, csv_path):
    df = pd.read_csv(os.path.join(SCRIPT_DIR, csv_path))
    df = df[df["counter"].isin(FIVE_COUNTERS)]

    fig, ax = plt.subplots(figsize=(5, 4))
    for mode in ["jump", "drift"]:
        sub = df[df["mode"] == mode]
        agg = sub.groupby("window")["detected"].mean().sort_index()
        ax.plot(agg.index, agg.values, marker="o", color=MODE_COLORS[mode],
                linewidth=2, label=mode)
        for w, v in agg.items():
            ax.annotate(f"{v:.1f}", (w, v), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=MODE_COLORS[mode])

    ax.axvline(CHOSEN_WINDOW[metric], color="gray", linestyle="--", linewidth=1,
               label=f"Chosen W={CHOSEN_WINDOW[metric]}")
    ax.set_xlabel("Window size $W$")
    ax.set_ylabel("Mean trials detected (/ 20)")
    ax.set_ylim(0, 21)
    ax.set_title(f"{metric} metric -- window-size sensitivity\n(mean over hpmcounter3/4/5/8/10)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"window_size_sweep_{metric}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    for metric, csv_path in METRICS:
        make_plot(metric, csv_path)


if __name__ == "__main__":
    main()
