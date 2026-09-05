"""ML fusion classifier on a CONTINUOUS simultaneous-agreement feature --
no boolean thresholding baked into the feature at all, unlike
ml_fusion_agreement_classifier.py's n_agree count (which pre-decides
"did this metric cross its own Q95 threshold" before the model ever
sees the data -- a legitimate objection: that's hand-designed voting
wearing an ML costume).

ml_fusion_pca_classifier.py already showed the raw 15 independent-metric
features can't be fixed by better downstream modeling (PCA barely moved
the pre-onset stress-test false-alarm rate: 0.583->0.467 for
Mahalanobis) -- the problem isn't model capacity, it's that
max(|score|) taken independently per metric over the whole run discards
WHEN each metric fired, and that timing information can't be recovered
after the fact by any model. Some form of window-alignment has to be
preserved in the FEATURES, not decided in advance as attack/not-attack
-- preserving "when" is feature engineering, not a rule about "whether."

So this version keeps everything continuous: for each counter, at every
window position where D/G/V all have a defined value (same inner join
as ensemble_detection.py / ml_fusion_agreement_classifier.py), normalize
each metric's score by ITS OWN threshold (|score|/H -- a scale-fixing
step, not a boolean decision), then take the ELEMENTWISE MINIMUM of the
three normalized values at that window. This "concurrent minimum" is
large only when ALL THREE metrics are simultaneously elevated relative
to their own scale -- the same "did they agree in time" signal the
n_agree count captures -- but as a smooth magnitude, with no threshold
comparison, no True/False, anywhere in the feature construction. The
one-class model sees raw (normalized) numbers and has to learn for
itself what magnitude of concurrent elevation is anomalous.

Feature per counter: max over the relevant window range of
concurrent_min(t) -- 5 features total, same dimensionality benefit as
the agreement-count version, for the same fair standard-vs-expanded
comparison across all three feature designs (raw/agreement/continuous).
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

from ensemble_detection import METRICS as ENSEMBLE_METRICS, USABLE_COUNTERS  # noqa: E402
from ml_fusion_classifier import (  # noqa: E402
    ONSET_ITER, MODES, run_supervised_loso, run_one_class_loso, run_lof_loso,
)
from ml_fusion_expanded_normal import (  # noqa: E402
    ATTACK_MODES,
    run_one_class_loso as run_expanded_one_class_loso,
    run_lof_loso as run_expanded_lof_loso,
)


def normalized_frame(short_name, score_col, thresh_col, score_path, thresh_path):
    """Like ensemble_detection.flagged_frame, but keeps the continuous
    normalized magnitude (|score|/threshold) instead of a boolean flag."""
    scored = pd.read_csv(os.path.join(RESULTS_DIR, score_path))
    thresholds = pd.read_csv(os.path.join(RESULTS_DIR, thresh_path))

    df = scored[~scored["sigma_fragile"] & scored["counter"].isin(USABLE_COUNTERS)].merge(thresholds, on="counter")
    df[f"norm_{short_name}"] = df[score_col].abs() / df[thresh_col]
    return df[["counter", "mode", "seed", "window_end_iter", f"norm_{short_name}"]]


def build_windowed_concurrent_min():
    """[counter, mode, seed, window_end_iter, concurrent_min] --
    concurrent_min(t) = min(norm_D(t), norm_G(t), norm_V(t)), continuous,
    no thresholding of the combined value."""
    windowed = None
    for short_name, score_col, thresh_col, score_path, thresh_path in ENSEMBLE_METRICS:
        ndf = normalized_frame(short_name, score_col, thresh_col, score_path, thresh_path)
        windowed = ndf if windowed is None else windowed.merge(
            ndf, on=["counter", "mode", "seed", "window_end_iter"], how="inner"
        )
    windowed["concurrent_min"] = windowed[["norm_G", "norm_D", "norm_V"]].min(axis=1)
    return windowed[["counter", "mode", "seed", "window_end_iter", "concurrent_min"]]


def build_feature_table(windowed):
    post = windowed[windowed["window_end_iter"] >= ONSET_ITER]
    agg = post.groupby(["counter", "mode", "seed"])["concurrent_min"].max()

    seeds = sorted(windowed["seed"].unique(), key=lambda s: int(s.replace("seed", "")))
    rows = []
    for mode in MODES:
        for seed in seeds:
            row = {"mode": mode, "seed": seed}
            for counter in USABLE_COUNTERS:
                row[f"cmin_{counter}"] = agg.get((counter, mode, seed), np.nan)
            rows.append(row)

    table = pd.DataFrame(rows)
    feature_cols = [c for c in table.columns if c not in ("mode", "seed")]
    table[feature_cols] = table[feature_cols].fillna(table[feature_cols].median())
    return table, feature_cols


def build_expanded_feature_table(windowed):
    seeds = sorted(windowed["seed"].unique(), key=lambda s: int(s.replace("seed", "")))

    def feature_row(mode, seed, iter_mask_fn):
        row = {}
        sub_all = windowed[(windowed["mode"] == mode) & (windowed["seed"] == seed)]
        for counter in USABLE_COUNTERS:
            sub = sub_all[(sub_all["counter"] == counter) & iter_mask_fn(sub_all["window_end_iter"])]
            row[f"cmin_{counter}"] = sub["concurrent_min"].max() if len(sub) else float("nan")
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
    windowed = build_windowed_concurrent_min()

    table, feature_cols = build_feature_table(windowed)
    print(f"Standard feature table: {table.shape[0]} runs x {len(feature_cols)} features (continuous concurrent-min)")
    print(f"  mode counts: {table['mode'].value_counts().to_dict()}")

    standard_results = {
        "logreg (supervised)": run_supervised_loso(table, feature_cols, "logreg"),
        "random_forest (supervised)": run_supervised_loso(table, feature_cols, "rf"),
        "mahalanobis (one-class, normal-only)": run_one_class_loso(table, feature_cols),
        "lof (one-class, normal-only)": run_lof_loso(table, feature_cols),
    }
    standard_out = pd.DataFrame([{"method": name, **s} for name, s in standard_results.items()])
    standard_path = os.path.join(RESULTS_DIR, "ml_fusion_continuous_agreement_classifier_loso.csv")
    standard_out.to_csv(standard_path, index=False)
    print("\n=== Standard LOSO-CV, continuous concurrent-min features (n=60) ===")
    print(standard_out.to_string(index=False))
    print(f"Saved {standard_path}")

    expanded_table, expanded_cols = build_expanded_feature_table(windowed)
    print(f"\nExpanded feature table: {expanded_table.shape[0]} rows x {len(expanded_cols)} features")
    print(f"  row_type counts: {expanded_table['row_type'].value_counts().to_dict()}")

    expanded_results = {
        "mahalanobis (continuous features, pre-onset-as-normal test)": run_expanded_one_class_loso(expanded_table, expanded_cols),
        "lof (continuous features, pre-onset-as-normal test)": run_expanded_lof_loso(expanded_table, expanded_cols),
    }
    expanded_out = pd.DataFrame([{"method": name, **s} for name, s in expanded_results.items()])
    expanded_path = os.path.join(RESULTS_DIR, "ml_fusion_continuous_agreement_expanded_normal_loso.csv")
    expanded_out.to_csv(expanded_path, index=False)
    print(f"\n=== Expanded/pre-onset LOSO-CV, continuous concurrent-min features (n={expanded_table.shape[0]}) ===")
    print(expanded_out.to_string(index=False))
    print(f"Saved {expanded_path}")


if __name__ == "__main__":
    main()
