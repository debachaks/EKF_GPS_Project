"""Audit of pre-onset (iter < 150) flags, across ALL modes -- normal,
jump, AND drift -- for D_final/G_final/V_final. The confusion-matrix
scripts (detection_confusion_matrix.py, ensemble_detection.py) apply the
post-onset-only rule uniformly, which silently discards any flag before
iter 150 regardless of mode:

  - in NORMAL runs, a pre-onset flag is a genuine false alarm that never
    gets counted as FP (FP only counts post-onset normal flags).
  - in JUMP/DRIFT runs, a pre-onset flag is ALSO a false alarm in the
    sense that no real attack has started yet at that point in the run
    -- but it's invisible in the confusion matrix too, since ground
    truth is assigned per whole run (mode=jump/drift -> 1), not per
    window, so it just quietly doesn't help or hurt that run's TP/FN
    classification unless it's the run's ONLY flag (in which case the
    run becomes an FN, not an FP).

This script reports, per metric per counter per mode: how many runs have
at least one pre-onset flag, and what fraction of all flagged windows
(in that mode) are pre-onset -- the same style of check used earlier in
this project for individual metrics, now applied consistently across
the whole D_final/G_final/V_final set for direct comparison.
"""

import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")

ONSET_ITER = 150

# (label, score_col, thresh_col, score_path, thresh_path)
METRICS = [
    ("D_final (W=10)", "d", "H_d", "d_final_dscore.csv", "d_final_thresholds.csv"),
    ("G_final (W=5)", "g", "H_g", "g_final_w5_gscore.csv", "g_final_w5_thresholds.csv"),
    ("V_final (W=5)", "v", "H_v", "v_final_w5_vscore.csv", "v_final_w5_thresholds.csv"),
]
MODES = ["normal", "jump", "drift"]

# hpmcounter9 excluded: established near-negative-control, noise-contaminated
# (see the per-counter audit above) -- pooling it in would swamp the
# mode-comparison signal this second audit is actually looking for.
USABLE_COUNTERS = ["hpmcounter3", "hpmcounter4", "hpmcounter5", "hpmcounter8", "hpmcounter10"]


def audit_metric(label, score_col, thresh_col, score_path, thresh_path):
    scored = pd.read_csv(os.path.join(RESULTS_DIR, score_path))
    thresholds = pd.read_csv(os.path.join(RESULTS_DIR, thresh_path))

    df = scored[~scored["sigma_fragile"]].merge(thresholds, on="counter")
    df["flagged"] = df[score_col].abs() > df[thresh_col]
    df["pre_onset"] = df["window_end_iter"] < ONSET_ITER

    rows = []
    for (counter, mode), grp in df.groupby(["counter", "mode"]):
        flagged = grp[grp["flagged"]]
        n_flags = len(flagged)
        n_pre_onset_flags = int(flagged["pre_onset"].sum())
        pct_pre_onset = round(100 * n_pre_onset_flags / n_flags, 1) if n_flags > 0 else float("nan")

        per_run_any_pre = (
            flagged[flagged["pre_onset"]]
            .groupby("seed")
            .size()
        )
        n_runs_with_pre_onset_flag = per_run_any_pre.shape[0]
        n_runs_total = grp["seed"].nunique()

        rows.append({
            "metric": label, "counter": counter, "mode": mode,
            "n_runs_total": n_runs_total,
            "n_runs_with_pre_onset_flag": n_runs_with_pre_onset_flag,
            "n_flagged_windows": n_flags,
            "n_pre_onset_flagged_windows": n_pre_onset_flags,
            "pct_flags_pre_onset": pct_pre_onset,
        })
    return rows


def audit_mode_pooled_rate(label, score_col, thresh_col, score_path, thresh_path):
    """Pre-onset flag RATE (not share-of-flags) by mode, pooled across
    USABLE_COUNTERS -- checks whether pre-onset activation is actually
    mode-independent, which the post-onset-only detection rule implicitly
    assumes (paper Section 9, item 6 / Table 4).
    """
    scored = pd.read_csv(os.path.join(RESULTS_DIR, score_path))
    thresholds = pd.read_csv(os.path.join(RESULTS_DIR, thresh_path))

    df = scored[scored["counter"].isin(USABLE_COUNTERS)].merge(thresholds, on="counter")
    df["flagged"] = (~df["sigma_fragile"]) & (df[score_col].abs() > df[thresh_col])
    pre = df[df["window_end_iter"] < ONSET_ITER]

    rows = []
    for mode, grp in pre.groupby("mode"):
        n_flagged = int(grp["flagged"].sum())
        n_total = len(grp)
        rows.append({
            "metric": label, "mode": mode,
            "n_pre_onset_flagged_windows": n_flagged,
            "n_pre_onset_windows_total": n_total,
            "pct_pre_onset_windows_flagged": round(100 * n_flagged / n_total, 4) if n_total else float("nan"),
        })
    return rows


def main():
    all_rows = []
    for label, score_col, thresh_col, score_path, thresh_path in METRICS:
        all_rows.extend(audit_metric(label, score_col, thresh_col, score_path, thresh_path))

    out = pd.DataFrame(all_rows)
    counter_order = sorted(out["counter"].unique(), key=lambda c: int(c.replace("hpmcounter", "")))
    out["counter"] = pd.Categorical(out["counter"], categories=counter_order, ordered=True)
    out["mode"] = pd.Categorical(out["mode"], categories=MODES, ordered=True)
    out = out.sort_values(["metric", "counter", "mode"])

    out_path = os.path.join(RESULTS_DIR, "pre_onset_audit.csv")
    out.to_csv(out_path, index=False)

    for label, _, _, _, _ in METRICS:
        print(f"\n=== {label} ===")
        print(out[out["metric"] == label].drop(columns="metric").to_string(index=False))
    print(f"\nSaved {out_path}")

    mode_rows = []
    for label, score_col, thresh_col, score_path, thresh_path in METRICS:
        mode_rows.extend(audit_mode_pooled_rate(label, score_col, thresh_col, score_path, thresh_path))

    mode_out = pd.DataFrame(mode_rows)
    mode_out["mode"] = pd.Categorical(mode_out["mode"], categories=MODES, ordered=True)
    mode_out = mode_out.sort_values(["metric", "mode"])

    mode_out_path = os.path.join(RESULTS_DIR, "pre_onset_mode_rate.csv")
    mode_out.to_csv(mode_out_path, index=False)

    print(f"\n=== Pre-onset flag rate by mode, pooled across {USABLE_COUNTERS} (paper Table 4) ===")
    print(mode_out.to_string(index=False))
    print(f"\nSaved {mode_out_path}")


if __name__ == "__main__":
    main()
