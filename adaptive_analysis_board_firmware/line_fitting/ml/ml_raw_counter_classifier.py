"""Ablation of ml_fusion_classifier.py: does the hand-engineered D/G/V
window-diff feature construction actually help, or does a classifier
learn just as well straight off the raw (z-scored) hardware counters?

Same LOSO-CV protocol, same 15-feature-per-run shape, same models --
only the feature-construction step changes. Instead of max(|D or G or V
score|) over post-onset windows (the windowed slope/variance/mean
statistics from d_final_metric.py etc.), this uses mean/max/std of the
raw per-iteration z-scored counter value itself (results/zscore_
timeseries.csv, the same baseline zscore_baseline.py computes and every
D/G/V metric is built on top of) over the post-onset region, with no
windowing or differencing at all.

Caveat carried over from ml_fusion_classifier.py: this skips the
sigma-fragility guard the D/G/V pipeline was specifically built to
handle (Section 6.3 -- z-scores blowing up near-singular between-trial
variance). Raw per-iteration z-scores don't have that guard applied, so
any resulting instability here is expected to show up as noisier
features, not as a numerical crash (mean/max/std over ~150 iterations
is far less sensitive to a handful of extreme values than a windowed
score built directly from a near-zero denominator would be).
"""

import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")

sys.path.insert(0, SCRIPT_DIR)
from ml_fusion_classifier import (  # noqa: E402
    ONSET_ITER, USABLE_COUNTERS, run_supervised_loso, run_one_class_loso, run_lof_loso,
)

STATS = ["mean", "max", "std"]


def build_feature_table():
    """One row per (mode, seed): 15 features (5 counters x 3 stats:
    mean/max/std of the raw z-scored counter value over post-onset
    iterations, no windowing/differencing)."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "zscore_timeseries.csv"))
    df = df[df["counter"].isin(USABLE_COUNTERS) & (df["iter"] >= ONSET_ITER)]

    agg = df.groupby(["counter", "mode", "seed"])["z"].agg(["mean", "max", "std"])

    seeds = sorted(df["seed"].unique(), key=lambda s: int(s.replace("seed", "")))
    modes = sorted(df["mode"].unique())

    rows = []
    for mode in modes:
        for seed in seeds:
            row = {"mode": mode, "seed": seed}
            for counter in USABLE_COUNTERS:
                for stat in STATS:
                    key = (counter, mode, seed)
                    row[f"{stat}_{counter}"] = agg.loc[key, stat] if key in agg.index else float("nan")
            rows.append(row)

    table = pd.DataFrame(rows)
    feature_cols = [c for c in table.columns if c not in ("mode", "seed")]
    table[feature_cols] = table[feature_cols].fillna(table[feature_cols].median())
    return table, feature_cols


def main():
    table, feature_cols = build_feature_table()
    print(f"Feature table: {table.shape[0]} runs x {len(feature_cols)} features (raw counter z-scores)")
    print(f"  mode counts: {table['mode'].value_counts().to_dict()}")

    results = {
        "logreg (supervised, raw)": run_supervised_loso(table, feature_cols, "logreg"),
        "random_forest (supervised, raw)": run_supervised_loso(table, feature_cols, "rf"),
        "mahalanobis (one-class, raw)": run_one_class_loso(table, feature_cols),
        "lof (one-class, raw)": run_lof_loso(table, feature_cols),
    }

    rows = [{"method": name, **stats_} for name, stats_ in results.items()]
    out = pd.DataFrame(rows)

    out_path = os.path.join(RESULTS_DIR, "ml_raw_counter_classifier_loso.csv")
    out.to_csv(out_path, index=False)

    print("\n=== LOSO-CV results, raw z-scored counters (attack=1, normal=0; n=60) ===")
    print(out.to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
