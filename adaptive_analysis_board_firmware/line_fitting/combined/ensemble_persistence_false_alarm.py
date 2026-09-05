"""False-alarm plots for the ensemble scenarios (I/II/III), using the
FINAL selected detection rule (per counter/trial, pre-onset windows):

    I)   A or B or C, sustained for >=3 CONSECUTIVE windows
    II)  at least 2 fire, sustained for >=3 CONSECUTIVE windows
    III) A and B and C, single window (no persistence requirement)

Chosen after comparing plain single-window Scenario I/II/III against
this mixed rule: plain Scenario I/II had a severe pre-onset false-alarm
rate (I: 75%/55% jump/drift, pooled across counters -- see
ensemble_per_trial_confusion.py), which a 3-consecutive-window
persistence filter brings down sharply (I: 15%/15%) while barely
touching Scenario I's already-strong recall. The same persistence
filter, tried on Scenario III, badly damaged its recall (90%->5% jump,
50%->10% drift) since III's simultaneous-3-metric-agreement is already
rare and brief -- stacking a second strictness requirement on top left
almost nothing. So III keeps its original single-window rule; I/II gain
the persistence filter.

False alarm here = per counter, per (attack_type, scenario): out of the
20 trials of that attack type, what fraction have >=1 pre-onset
(window_end_iter < 150) window satisfying the rule above, in that
SINGLE counter (not pooled across counters -- this is the per-counter
view, matching heatmap_4counters_midthreshold.py / ensemble_detection.py's
per-counter convention, as opposed to the pooled-across-5-counters
number used earlier to diagnose the problem).

Detection rate (recall) is computed the same way but on POST-onset
windows (window_end_iter >= 150), and saved to the same CSV for
reference, though the plot itself shows false alarm only, per request.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
PLOT_DIR = os.path.join(LINE_FITTING_DIR, "plots_heatmap")

sys.path.insert(0, SCRIPT_DIR)
from ensemble_firing_timeline import TARGET_COUNTERS, ONSET_ITER, SCENARIOS, scenario_fire_for_counter  # noqa: E402

ATTACK_TYPES = ["jump", "drift"]
ATTACK_COLORS = {"jump": "#E07B1A", "drift": "#2E8B3D"}
SEEDS = [f"seed{n}" for n in range(1, 21)]
N_CONSEC = 3

# per-scenario rule: "3consec" = require >=N_CONSEC consecutive fires, "any" = any single window
RULE = {"I": "3consec", "II": "3consec", "III": "any"}
SCENARIO_LABEL = {
    "I": f"I) A or B or C\n(>={N_CONSEC} consecutive)",
    "II": f"II) >=2 fire\n(>={N_CONSEC} consecutive)",
    "III": "III) A and B and C\n(single window)",
}


def has_n_consecutive(bool_arr, n):
    run = 0
    for v in bool_arr:
        run = run + 1 if v else 0
        if run >= n:
            return True
    return False


def rule_satisfied(bool_arr, scenario):
    if RULE[scenario] == "3consec":
        return has_n_consecutive(bool_arr, N_CONSEC)
    return bool(np.any(bool_arr))


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    rows = []
    for counter in TARGET_COUNTERS:
        for attack_type in ATTACK_TYPES:
            for scenario in SCENARIOS:
                n_pre_flagged, n_post_detected = 0, 0
                for seed in SEEDS:
                    common_iters, scenario_fire = scenario_fire_for_counter(counter, attack_type, seed)
                    fire = scenario_fire[scenario]

                    pre_fire = fire[common_iters < ONSET_ITER]
                    post_fire = fire[common_iters >= ONSET_ITER]

                    if rule_satisfied(pre_fire, scenario):
                        n_pre_flagged += 1
                    if rule_satisfied(post_fire, scenario):
                        n_post_detected += 1

                rows.append({
                    "counter": counter, "attack_type": attack_type, "scenario": scenario,
                    "n_pre_onset_false_alarm": n_pre_flagged, "false_alarm_rate": n_pre_flagged / len(SEEDS),
                    "n_post_onset_detected": n_post_detected, "detection_rate": n_post_detected / len(SEEDS),
                })

    out = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "ensemble_persistence_false_alarm.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved {out_path}")

    for counter in TARGET_COUNTERS:
        sub = out[out["counter"] == counter]

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for attack_type in ATTACK_TYPES:
            s = sub[sub["attack_type"] == attack_type].set_index("scenario").loc[SCENARIOS]
            ax.plot(SCENARIOS, s["false_alarm_rate"], marker="o", color=ATTACK_COLORS[attack_type],
                    linewidth=2, label=attack_type)
            for x, y in enumerate(s["false_alarm_rate"]):
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8, color=ATTACK_COLORS[attack_type])

        ax.set_xticks(range(len(SCENARIOS)))
        ax.set_xticklabels([SCENARIO_LABEL[s] for s in SCENARIOS], fontsize=8)
        ax.set_ylabel("Pre-onset false-alarm rate\n(fraction of 20 trials with >=1 false flag)")
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9, frameon=False)

        ax.set_title(f"{counter} -- false alarm vs. voting rule\n"
                     "(I/II require 3 consecutive windows; III single window)")
        fig.tight_layout()

        plot_path = os.path.join(PLOT_DIR, f"ensemble_persistence_false_alarm_{counter}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
