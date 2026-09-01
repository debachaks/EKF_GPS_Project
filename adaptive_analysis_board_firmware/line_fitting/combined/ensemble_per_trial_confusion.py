"""Per-trial, window-level confusion matrix for the ensemble scenarios
(I/II/III), computed WITHIN each jump/drift trial rather than across the
20-normal-vs-40-attack run-level pooling used by ensemble_detection.py.

Scope, per explicit instruction: only jump and drift trials (40 total,
20 each) -- normal-mode trials are not part of this evaluation at all
(they still implicitly matter, since the H thresholds used to decide
"detected" were calibrated from normal-mode Q95, but they don't
contribute rows here).

Ground truth is now WINDOW-level, not run-level: within a single trial,
every window with window_end_iter < ONSET_ITER has ground_truth=0 (no
attack yet), every window with window_end_iter >= ONSET_ITER has
ground_truth=1 (attack active) -- this replaces the post-onset-only
rule used everywhere else in the project (which simply discarded
pre-onset windows rather than scoring them as ground_truth=0).

For each (counter, scenario, trial): TP = ground_truth 1 & detected 1,
FP = ground_truth 0 & detected 1, FN = ground_truth 1 & detected 0,
TN = ground_truth 0 & detected 0, counted across that trial's windows
(the D/G/V-aligned intersection of window_end_iter, same as
ensemble_firing_timeline.py -- D starts at iter=10, G/V at iter=5, so
the common range starts at 10). Sigma-fragile positions are treated as
not-fired (consistent with scenario_fire_for_counter, not dropped from
the window count).

This script only computes and saves the raw per-trial values -- no
averaging, no plotting, per explicit instruction.
"""

import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")

sys.path.insert(0, SCRIPT_DIR)
from ensemble_firing_timeline import (  # noqa: E402
    ONSET_ITER, TARGET_COUNTERS, SCENARIOS, scenario_fire_for_counter,
)

ATTACK_TYPES = ["jump", "drift"]
SEEDS = [f"seed{n}" for n in range(1, 21)]


def confusion_for_trial(ground_truth, detected):
    tp = int(((ground_truth == 1) & detected).sum())
    fp = int(((ground_truth == 0) & detected).sum())
    fn = int(((ground_truth == 1) & (~detected)).sum())
    tn = int(((ground_truth == 0) & (~detected)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "n_windows": tp + fp + fn + tn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "false_alarm": round(false_alarm, 4),
    }


def main():
    rows = []
    for counter in TARGET_COUNTERS:
        for attack_type in ATTACK_TYPES:
            for seed in SEEDS:
                common_iters, scenario_fire = scenario_fire_for_counter(counter, attack_type, seed)
                ground_truth = (common_iters >= ONSET_ITER).astype(int)

                for scenario in SCENARIOS:
                    detected = scenario_fire[scenario]
                    stats = confusion_for_trial(ground_truth, detected)
                    rows.append({
                        "counter": counter, "attack_type": attack_type, "seed": seed,
                        "scenario": scenario, **stats,
                    })

    out = pd.DataFrame(rows)
    counter_order = sorted(out["counter"].unique(), key=lambda c: int(c.replace("hpmcounter", "")))
    out["counter"] = pd.Categorical(out["counter"], categories=counter_order, ordered=True)
    out["seed"] = pd.Categorical(out["seed"], categories=SEEDS, ordered=True)
    out["scenario"] = pd.Categorical(out["scenario"], categories=SCENARIOS, ordered=True)
    out = out.sort_values(["counter", "attack_type", "scenario", "seed"]).reset_index(drop=True)

    out_path = os.path.join(RESULTS_DIR, "ensemble_per_trial_confusion.csv")
    out.to_csv(out_path, index=False)

    print(f"Computed {len(out)} rows: {len(TARGET_COUNTERS)} counters x {len(ATTACK_TYPES)} attack types x "
          f"{len(SEEDS)} seeds x {len(SCENARIOS)} scenarios")
    print(f"\nSaved {out_path}")
    print("\n=== First 15 rows (hpmcounter3, jump, scenario III/II/I x first 5 seeds) ===")
    print(out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
