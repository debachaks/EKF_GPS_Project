"""ML fusion classifier built on ENSEMBLE-AGREEMENT-FILTERED features,
instead of ml_fusion_classifier.py's 15 independent per-metric max(|score|)
features (5 counters x 3 metrics D/G/V, each scored on its own).

Motivation (see notes/conversation_2026-08-27.txt Section 8, and
ml_fusion_expanded_normal.py): testing the independent-per-metric
features against jump/drift's PRE-onset segment (treated as a
"should be normal" test case) produced a severe false-alarm rate
(Mahalanobis FA=0.583, LOF FA=0.500 -- jump_pre flagged 85%, drift_pre
70% of the time). Tracing this down, the SAME pattern showed up using
the plain rule-based D/G/V flags with no ML involved at all: "did any of
the 3 metrics fire anywhere" is a permissive aggregation that surfaces
rare pre-onset noise as a high percentage. But ensemble_detection.py's
Scenario III (all three metrics fire AT THE SAME WINDOW) essentially
eliminates this: per-trial pre-onset false-alarm rate under Scenario III
is 0% for both jump and drift (vs. 75%/55% under Scenario I).

This script tests whether feeding that same simultaneous-agreement
signal into the ML models (instead of each metric's independent score)
inherits Scenario III's pre-onset robustness.

Feature construction: for each of the 5 usable counters, at every window
position where D/G/V all have a defined value (inner join on
window_end_iter -- reusing ensemble_detection.py's flagged_frame/METRICS
so the alignment and per-metric thresholds are identical to the
already-fixed, already-tested ensemble logic), compute
    n_agree(t) = flagged_D(t) + flagged_G(t) + flagged_V(t)   in {0,1,2,3}
The feature per counter is max(n_agree) over the relevant window range
(5 features total, not 15 -- also directly helps the n-approx-p problem
flagged in ml_fusion_classifier.py's Mahalanobis docstring, since 19
training points in a 5-d space is far better conditioned than 15-d).

Two evaluations, mirroring the two existing scripts exactly so results
are directly comparable:
  1. Standard (mirrors ml_fusion_classifier.py): post-onset-only
     features, normal/jump/drift as 3 whole-run classes, all 4 models
     (logreg, RF, Mahalanobis, LOF) via LOSO-CV.
  2. Expanded/pre-onset stress test (mirrors ml_fusion_expanded_normal.py):
     normal-mode trained on its full window range; jump/drift each
     contribute a post-onset test row (ground_truth=1) AND a pre-onset
     test row (ground_truth=0) -- Mahalanobis/LOF only, since that's
     what the original stress test exercised.

Model-fitting functions (run_supervised_loso, run_one_class_loso,
run_lof_loso from ml_fusion_classifier.py; the expanded-table variants
from ml_fusion_expanded_normal.py) are reused as-is, unmodified -- they
already operate generically on (table, feature_cols) / ground_truth
columns, so no feature-count-specific logic needed changing.
"""

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(LINE_FITTING_DIR, "combined"))

from ensemble_detection import METRICS as ENSEMBLE_METRICS, USABLE_COUNTERS, flagged_frame  # noqa: E402
from ml_fusion_classifier import (  # noqa: E402
    ONSET_ITER, MODES, run_supervised_loso, run_one_class_loso, run_lof_loso,
)
from ml_fusion_expanded_normal import (  # noqa: E402
    ATTACK_MODES,
    run_one_class_loso as run_expanded_one_class_loso,
    run_lof_loso as run_expanded_lof_loso,
)


def build_windowed_agreement():
    """[counter, mode, seed, window_end_iter, n_agree] -- n_agree in
    {0,1,2,3}, from the same D/G/V inner-join alignment and per-metric
    thresholds as ensemble_detection.py."""
    windowed = None
    for short_name, score_col, thresh_col, score_path, thresh_path in ENSEMBLE_METRICS:
        fdf = flagged_frame(short_name, score_col, thresh_col, score_path, thresh_path)
        windowed = fdf if windowed is None else windowed.merge(
            fdf, on=["counter", "mode", "seed", "window_end_iter"], how="inner"
        )
    windowed["n_agree"] = windowed[["flagged_G", "flagged_D", "flagged_V"]].sum(axis=1)
    return windowed[["counter", "mode", "seed", "window_end_iter", "n_agree"]]


def build_feature_table(windowed):
    """Standard evaluation: one row per (mode, seed), 5 features (one per
    counter) = max(n_agree) over post-onset windows."""
    post = windowed[windowed["window_end_iter"] >= ONSET_ITER]
    agg = post.groupby(["counter", "mode", "seed"])["n_agree"].max()

    seeds = sorted(windowed["seed"].unique(), key=lambda s: int(s.replace("seed", "")))
    rows = []
    for mode in MODES:
        for seed in seeds:
            row = {"mode": mode, "seed": seed}
            for counter in USABLE_COUNTERS:
                row[f"agree_{counter}"] = agg.get((counter, mode, seed), np.nan)
            rows.append(row)

    table = pd.DataFrame(rows)
    feature_cols = [c for c in table.columns if c not in ("mode", "seed")]
    table[feature_cols] = table[feature_cols].fillna(table[feature_cols].median())
    return table, feature_cols


def build_expanded_feature_table(windowed):
    """Expanded/pre-onset stress test: normal_full (train-eligible, full
    window range) + jump/drift post-onset (ground_truth=1) and pre-onset
    (ground_truth=0, test-only) rows -- same shape as
    ml_fusion_expanded_normal.build_expanded_feature_table."""
    seeds = sorted(windowed["seed"].unique(), key=lambda s: int(s.replace("seed", "")))

    def feature_row(mode, seed, iter_mask_fn):
        row = {}
        sub_all = windowed[(windowed["mode"] == mode) & (windowed["seed"] == seed)]
        for counter in USABLE_COUNTERS:
            sub = sub_all[(sub_all["counter"] == counter) & iter_mask_fn(sub_all["window_end_iter"])]
            row[f"agree_{counter}"] = sub["n_agree"].max() if len(sub) else float("nan")
        return row

    rows = []
    for seed in seeds:
        rows.append({
            "row_type": "normal_full", "seed": seed, "ground_truth": 0, "train_eligible": True,
            **feature_row("normal", seed, lambda it: it >= 0),
        })
        for mode in ATTACK_MODES:
            rows.append({
                "row_type": f"{mode}_post", "seed": seed, "ground_truth": 1, "train_eligible": False,
                **feature_row(mode, seed, lambda it: it >= ONSET_ITER),
            })
            rows.append({
                "row_type": f"{mode}_pre", "seed": seed, "ground_truth": 0, "train_eligible": False,
                **feature_row(mode, seed, lambda it: it < ONSET_ITER),
            })

    table = pd.DataFrame(rows)
    feature_cols = [c for c in table.columns if c not in ("row_type", "seed", "ground_truth", "train_eligible")]
    table[feature_cols] = table[feature_cols].fillna(table[feature_cols].median())
    return table, feature_cols


def main():
    windowed = build_windowed_agreement()

    # --- standard evaluation ---
    table, feature_cols = build_feature_table(windowed)
    print(f"Standard feature table: {table.shape[0]} runs x {len(feature_cols)} features (agreement-filtered)")
    print(f"  mode counts: {table['mode'].value_counts().to_dict()}")

    standard_results = {
        "logreg (supervised)": run_supervised_loso(table, feature_cols, "logreg"),
        "random_forest (supervised)": run_supervised_loso(table, feature_cols, "rf"),
        "mahalanobis (one-class, normal-only)": run_one_class_loso(table, feature_cols),
        "lof (one-class, normal-only)": run_lof_loso(table, feature_cols),
    }
    standard_out = pd.DataFrame([{"method": name, **s} for name, s in standard_results.items()])
    standard_path = os.path.join(RESULTS_DIR, "ml_fusion_agreement_classifier_loso.csv")
    standard_out.to_csv(standard_path, index=False)
    print("\n=== Standard LOSO-CV, agreement-filtered features (n=60) ===")
    print(standard_out.to_string(index=False))
    print(f"Saved {standard_path}")

    # --- expanded / pre-onset stress test ---
    expanded_table, expanded_cols = build_expanded_feature_table(windowed)
    print(f"\nExpanded feature table: {expanded_table.shape[0]} rows x {len(expanded_cols)} features")
    print(f"  row_type counts: {expanded_table['row_type'].value_counts().to_dict()}")

    expanded_results = {
        "mahalanobis (agreement features, pre-onset-as-normal test)": run_expanded_one_class_loso(expanded_table, expanded_cols),
        "lof (agreement features, pre-onset-as-normal test)": run_expanded_lof_loso(expanded_table, expanded_cols),
    }
    expanded_out = pd.DataFrame([{"method": name, **s} for name, s in expanded_results.items()])
    expanded_path = os.path.join(RESULTS_DIR, "ml_fusion_agreement_expanded_normal_loso.csv")
    expanded_out.to_csv(expanded_path, index=False)
    print(f"\n=== Expanded/pre-onset LOSO-CV, agreement-filtered features (n={expanded_table.shape[0]}) ===")
    print(expanded_out.to_string(index=False))
    print(f"Saved {expanded_path}")


if __name__ == "__main__":
    main()
