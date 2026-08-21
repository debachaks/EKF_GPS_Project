"""Ensemble detection across the three "final" detectors -- per the
whiteboard, A=G_final, B=D_final, C=V_final -- combined at the run level
under three voting rules:

    I)   A or B or C        (any one fires)
    II)  at least 2 fire    (majority vote)
    III) A and B and C      (all three must fire)

Each metric's own per-run "detected" flag is exactly the post-onset-flag
rule already used throughout this project (run_detected in
detection_confusion_matrix.py); this script reuses that, then combines
the three boolean columns with OR / majority / AND before recomputing
the same confusion matrix (TP/FP/FN/TN) and derived Precision/Recall/F1/
False-Alarm per counter per scenario.

Jump and drift are scored as SEPARATE binary classification problems
(jump vs. normal, and drift vs. normal), not pooled into one "attack"
class -- so every F1/False-Alarm figure has two lines. Note that the
False-Alarm line is expected to come out identical for jump and drift:
FA = FP/(FP+TN) depends only on the 20 normal-mode runs' detected/
not-detected status, which doesn't change depending on which attack
type they're being scored against.

Same three metric configurations as heatmap_4counters_midthreshold.py /
detection_confusion_matrix.py: D_final at W=10, G_final and V_final at
W=5.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
PLOT_DIR = os.path.join(LINE_FITTING_DIR, "plots_heatmap")

ONSET_ITER = 150

# (short_name, score_col, thresh_col, score_path, thresh_path)
METRICS = [
    ("G", "g", "H_g", "g_final_w5_gscore.csv", "g_final_w5_thresholds.csv"),
    ("D", "d", "H_d", "d_final_dscore.csv", "d_final_thresholds.csv"),
    ("V", "v", "H_v", "v_final_w5_vscore.csv", "v_final_w5_thresholds.csv"),
]

SCENARIOS = ["I", "II", "III"]
SCENARIO_LABEL = {
    "I": "I) A or B or C",
    "II": "II) at least 2 fire",
    "III": "III) A and B and C",
}

ATTACK_TYPES = ["jump", "drift"]
ATTACK_COLORS = {"jump": "#E07B1A", "drift": "#2E8B3D"}


def run_detected_for_metric(score_col, thresh_col, score_path, thresh_path):
    scored = pd.read_csv(os.path.join(RESULTS_DIR, score_path))
    thresholds = pd.read_csv(os.path.join(RESULTS_DIR, thresh_path))

    df = scored[~scored["sigma_fragile"]].merge(thresholds, on="counter")
    df["flagged"] = df[score_col].abs() > df[thresh_col]
    df["post_onset_flag"] = df["flagged"] & (df["window_end_iter"] >= ONSET_ITER)

    return (
        df.groupby(["counter", "mode", "seed"])["post_onset_flag"]
        .any()
        .reset_index(name="detected")
    )


def confusion_stats(grp):
    tp = int(((grp["ground_truth"] == 1) & (grp["detected"])).sum())
    fn = int(((grp["ground_truth"] == 1) & (~grp["detected"])).sum())
    fp = int(((grp["ground_truth"] == 0) & (grp["detected"])).sum())
    tn = int(((grp["ground_truth"] == 0) & (~grp["detected"])).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    return tp, fp, fn, tn, precision, recall, f1, false_alarm


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    merged = None
    for short_name, score_col, thresh_col, score_path, thresh_path in METRICS:
        rd = run_detected_for_metric(score_col, thresh_col, score_path, thresh_path)
        rd = rd.rename(columns={"detected": f"detected_{short_name}"})
        merged = rd if merged is None else merged.merge(rd, on=["counter", "mode", "seed"])

    n_fire = merged[["detected_G", "detected_D", "detected_V"]].sum(axis=1)
    merged["I"] = n_fire >= 1
    merged["II"] = n_fire >= 2
    merged["III"] = n_fire >= 3

    rows = []
    for counter, cgrp in merged.groupby("counter"):
        for attack_type in ATTACK_TYPES:
            sub_grp = cgrp[cgrp["mode"].isin([attack_type, "normal"])].copy()
            sub_grp["ground_truth"] = (sub_grp["mode"] == attack_type).astype(int)
            for scenario in SCENARIOS:
                g = sub_grp[["ground_truth"]].copy()
                g["detected"] = sub_grp[scenario]
                tp, fp, fn, tn, precision, recall, f1, fa = confusion_stats(g)
                rows.append({
                    "counter": counter, "attack_type": attack_type, "scenario": scenario,
                    "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                    "precision": round(precision, 3), "recall": round(recall, 3),
                    "f1": round(f1, 3), "false_alarm": round(fa, 3),
                })

    out = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "ensemble_detection_confusion_matrix.csv")
    out.to_csv(out_path, index=False)

    counter_order = sorted(out["counter"].unique(), key=lambda c: int(c.replace("hpmcounter", "")))
    for counter in counter_order:
        print(f"\n=== {counter} ===")
        print(out[out["counter"] == counter].drop(columns="counter").to_string(index=False))
    print(f"\nSaved {out_path}")

    # one figure per counter: (a) F1 vs scenario, (b) False Alarm vs scenario -- one line per attack type
    for counter in counter_order:
        csub = out[out["counter"] == counter]

        fig, axes = plt.subplots(2, 1, figsize=(6, 8))
        for ax, metric_col, ylabel in [(axes[0], "f1", "F1"), (axes[1], "false_alarm", "False Alarm (FA)")]:
            for attack_type in ATTACK_TYPES:
                sub = csub[csub["attack_type"] == attack_type].set_index("scenario").loc[SCENARIOS]
                ax.plot(SCENARIOS, sub[metric_col], marker="o", color=ATTACK_COLORS[attack_type],
                        linewidth=2, label=attack_type)
                for x, y in enumerate(sub[metric_col]):
                    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                                ha="center", fontsize=8, color=ATTACK_COLORS[attack_type])
            ax.set_xticks(range(len(SCENARIOS)))
            ax.set_xticklabels([SCENARIO_LABEL[s] for s in SCENARIOS], fontsize=9)
            ax.set_ylabel(ylabel)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=9, frameon=False)

        fig.suptitle(f"{counter} -- ensemble detection (A=G_final, B=D_final, C=V_final)\nF1 and False Alarm vs. voting rule, jump vs. drift")
        fig.tight_layout()
        plot_path = os.path.join(PLOT_DIR, f"ensemble_detection_f1_fa_{counter}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
