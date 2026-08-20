"""Standard detection confusion matrix (TP/FP/FN/TN) and derived metrics
(Precision, Recall, F1, False Alarm rate) for D_final/G_final/V_final,
per counter -- per the whiteboard design:

    ground truth: attack (jump/drift) = 1, normal = 0
    detection:    run_detected (any post-onset-flag in that run) = 1, else 0

    TP = attack run, detected      FP = normal run, detected
    FN = attack run, not detected  TN = normal run, not detected

    P  = Precision = TP / (TP + FP)
    R  = Recall    = TP / (TP + FN)
    F1 = 2*TP / (2*TP + FP + FN)
    FA = False Alarm rate = FP / (FP + TN)

Jump and drift are pooled into the single "attack" class (40 attack runs
+ 20 normal runs = 60 per counter), matching the whiteboard's generic
attack=1/normal=0 framing rather than scoring jump/drift separately.

Uses the same three metric configurations as heatmap_4counters_midthreshold.py:
D_final at W=10, G_final and V_final at W=5 (g_final_metric_w5.py /
v_final_metric_w5.py), with the same sigma-fragile exclusion and
post-onset-only detection rule as each metric's own script.
"""

import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

ONSET_ITER = 150

# (label, score_col, thresh_col, score_path, thresh_path)
METRICS = [
    ("D_final (W=10)", "d", "H_d", "d_final_dscore.csv", "d_final_thresholds.csv"),
    ("G_final (W=5)", "g", "H_g", "g_final_w5_gscore.csv", "g_final_w5_thresholds.csv"),
    ("V_final (W=5)", "v", "H_v", "v_final_w5_vscore.csv", "v_final_w5_thresholds.csv"),
]


def confusion_for_metric(label, score_col, thresh_col, score_path, thresh_path):
    scored = pd.read_csv(os.path.join(RESULTS_DIR, score_path))
    thresholds = pd.read_csv(os.path.join(RESULTS_DIR, thresh_path))

    df = scored[~scored["sigma_fragile"]].merge(thresholds, on="counter")
    df["flagged"] = df[score_col].abs() > df[thresh_col]
    df["post_onset_flag"] = df["flagged"] & (df["window_end_iter"] >= ONSET_ITER)

    run_detected = (
        df.groupby(["counter", "mode", "seed"])["post_onset_flag"]
        .any()
        .reset_index(name="detected")
    )
    run_detected["ground_truth"] = run_detected["mode"].isin(["jump", "drift"]).astype(int)

    rows = []
    for counter, grp in run_detected.groupby("counter"):
        tp = int(((grp["ground_truth"] == 1) & (grp["detected"])).sum())
        fn = int(((grp["ground_truth"] == 1) & (~grp["detected"])).sum())
        fp = int(((grp["ground_truth"] == 0) & (grp["detected"])).sum())
        tn = int(((grp["ground_truth"] == 0) & (~grp["detected"])).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
        false_alarm = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

        rows.append({
            "metric": label, "counter": counter,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "false_alarm": round(false_alarm, 3),
        })
    return rows


def main():
    all_rows = []
    for label, score_col, thresh_col, score_path, thresh_path in METRICS:
        all_rows.extend(confusion_for_metric(label, score_col, thresh_col, score_path, thresh_path))

    out = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, "detection_confusion_matrix.csv")
    out.to_csv(out_path, index=False)

    counter_order = sorted(out["counter"].unique(), key=lambda c: int(c.replace("hpmcounter", "")))
    out["counter"] = pd.Categorical(out["counter"], categories=counter_order, ordered=True)
    out = out.sort_values(["metric", "counter"])

    for metric in [m[0] for m in METRICS]:
        print(f"\n=== {metric} ===")
        print(out[out["metric"] == metric].drop(columns="metric").to_string(index=False))

    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
