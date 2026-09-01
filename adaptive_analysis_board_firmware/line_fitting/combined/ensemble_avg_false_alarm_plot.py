"""One false-alarm plot per counter: x-axis = ensemble scenario (I/II/III),
y-axis = average false-alarm ratio, one line for jump and one for drift.

Reads results/ensemble_avg_false_alarm.csv (from
ensemble_per_trial_confusion.py's per-trial false_alarm column, averaged
across the 20 jump trials and 20 drift trials separately, per counter
and scenario) -- window-level ground truth within each trial
(ground_truth=0 before iteration 150, =1 from 150 on), false_alarm =
FP/(FP+TN) computed per trial then averaged, not pooled across trials.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
PLOT_DIR = os.path.join(LINE_FITTING_DIR, "plots_heatmap")

SCENARIOS = ["I", "II", "III"]
SCENARIO_LABEL = {
    "I": "I) A or B or C",
    "II": "II) >=2 fire",
    "III": "III) A and B and C",
}
ATTACK_TYPES = ["jump", "drift"]
ATTACK_COLORS = {"jump": "#E07B1A", "drift": "#2E8B3D"}


def make_figure(df, counter):
    sub = df[df["counter"] == counter]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for attack_type in ATTACK_TYPES:
        s = sub[sub["attack_type"] == attack_type].set_index("scenario").loc[SCENARIOS]
        ax.plot(SCENARIOS, s["false_alarm"], marker="o", color=ATTACK_COLORS[attack_type],
                linewidth=2, label=attack_type)
        for x, y in enumerate(s["false_alarm"]):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=ATTACK_COLORS[attack_type])

    ax.set_xticks(range(len(SCENARIOS)))
    ax.set_xticklabels([SCENARIO_LABEL[s] for s in SCENARIOS], fontsize=9)
    ax.set_ylabel("Average false-alarm ratio")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

    ax.set_title(f"{counter} -- average false-alarm ratio vs. voting rule\n"
                 "(window-level ground truth, averaged per trial: 20 jump, 20 drift)")
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"ensemble_avg_false_alarm_{counter}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(RESULTS_DIR, "ensemble_avg_false_alarm.csv"))

    counter_order = sorted(df["counter"].unique(), key=lambda c: int(c.replace("hpmcounter", "")))
    for counter in counter_order:
        make_figure(df, counter)


if __name__ == "__main__":
    main()
