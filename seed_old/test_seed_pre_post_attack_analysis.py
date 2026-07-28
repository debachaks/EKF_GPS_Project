"""Within-run before/after attack-onset comparison.

The existing run-level tests (test_seed_run_level_hpmcounter_analysis.py /
test_seed_run_level_rate_analysis.py) compare whole-run summaries between
attack runs and normal runs - they answer "is this run different overall?",
not "does something change partway through, right at the attack?". This
script targets the sharper question: within EACH run, does the hpmcounter
rate shift right at the known attack onset point?

Attack onset: ekf_diag_<mode>.csv has a 't' step column (1..299) and an
attack_active flag. Checking every seed/mode confirms attack_active flips
0->1 at exactly t=150/299 (fraction 0.5000) in every case - including
"normal" runs, where it's a harness artifact (no real attack is injected
there, but the flag still flips and spoof_error still becomes nonzero
afterward). Every run - normal included - is split at that same relative
point (halfway through elapsed time), so normal acts as a negative control:
if normal also shows a significant pre/post shift, that's a structural
artifact, not a genuine attack effect.

For each run, per counter: compute row-wise rate (counter delta / mcycle
delta, as in test_seed_run_level_rate_analysis.py), split rows into
pre-onset and post-onset by elapsed time, and take the median rate in each
half. Across all seeds of a given run_type this gives paired
(median_pre, median_post) samples - one pair per seed - compared with a
paired Wilcoxon signed-rank test. Pairing removes between-seed baseline
noise (the dominant noise source we found in the raw/diff trace plots),
unlike the unpaired Mann-Whitney used in the other run-level scripts.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from data_preprocessing import hex_to_int  # noqa: E402
from hpmcounter_analysis import benjamini_hochberg  # noqa: E402

RAW_ROOT = SCRIPT_DIR
CLEAN_ROOT = os.path.join(SCRIPT_DIR, "CLEAN_HPC_TEST_SEED")
STATS_OUT = os.path.join(SCRIPT_DIR, "test_seed_pre_post_attack_stats.csv")

ALL_TYPES = ["normal", "drift", "jump", "replay"]
DEFAULT_ONSET_FRACTION = 0.5


def find_seed_names():
    clean_dirs = sorted(d for d in glob.glob(os.path.join(CLEAN_ROOT, "test_seed_[0-9]*")) if os.path.isdir(d))
    return [os.path.basename(d) for d in clean_dirs]


def find_counters(seed_names):
    common = None
    for seed_name in seed_names:
        for run_type in ALL_TYPES:
            path = os.path.join(CLEAN_ROOT, seed_name, f"ekf_{run_type}_hpc.csv")
            cols = frozenset(c for c in pd.read_csv(path, nrows=1).columns if c.startswith("hpmcounter"))
            common = cols if common is None else (common & cols)
    return sorted(common, key=lambda c: int(c.replace("hpmcounter", "")))


def onset_fraction(seed_name, run_type):
    path = os.path.join(RAW_ROOT, seed_name, f"ekf_diag_{run_type}.csv")
    diag = pd.read_csv(path)
    t_min, t_max = diag["t"].min(), diag["t"].max()
    active = diag[diag["attack_active"] == 1]
    if active.empty:
        return DEFAULT_ONSET_FRACTION
    onset_t = active["t"].min()
    return (onset_t - t_min) / (t_max - t_min)


def pre_post_medians(seed_name, run_type, counters):
    path = os.path.join(CLEAN_ROOT, seed_name, f"ekf_{run_type}_hpc.csv")
    df = pd.read_csv(path)

    mcycle = df["mcycle"].map(hex_to_int)
    mcycle_delta = mcycle.diff()
    counter_deltas = df[counters].apply(lambda col: col.map(hex_to_int)).diff()
    rates = counter_deltas.div(mcycle_delta, axis=0).iloc[1:].replace([np.inf, -np.inf], np.nan)

    elapsed = df["timestamp_ms"].map(hex_to_int) - df["timestamp_ms"].map(hex_to_int).iloc[0]
    elapsed_aligned = elapsed.iloc[1:]

    cutoff = onset_fraction(seed_name, run_type) * elapsed.iloc[-1]
    pre_mask = elapsed_aligned < cutoff
    post_mask = ~pre_mask

    return rates[pre_mask].median(), rates[post_mask].median()


def build_pairs(seed_names, run_type, counters):
    rows = []
    for seed_name in seed_names:
        pre, post = pre_post_medians(seed_name, run_type, counters)
        for counter in counters:
            rows.append({"seed": seed_name, "counter": counter, "pre": pre[counter], "post": post[counter]})
    return pd.DataFrame(rows)


def run_tests(seed_names, counters):
    rows = []
    for run_type in ALL_TYPES:
        pairs = build_pairs(seed_names, run_type, counters)
        for counter in counters:
            sub = pairs[pairs["counter"] == counter]
            pre_vals = sub["pre"].to_numpy()
            post_vals = sub["post"].to_numpy()

            stat, p_value = wilcoxon(post_vals, pre_vals, alternative="two-sided")
            diffs = post_vals - pre_vals
            paired_sign_delta = (np.sum(diffs > 0) - np.sum(diffs < 0)) / len(diffs)

            rows.append({
                "counter": counter,
                "run_type": run_type,
                "n_pairs": len(sub),
                "median_pre": np.median(pre_vals),
                "median_post": np.median(post_vals),
                "median_diff_post_minus_pre": np.median(diffs),
                "paired_sign_delta": paired_sign_delta,
                "wilcoxon_stat": stat,
                "p_value": p_value,
            })

    results = pd.DataFrame(rows)
    results["p_value_fdr"] = benjamini_hochberg(results["p_value"].to_numpy())
    results["significant_fdr_0.05"] = results["p_value_fdr"] < 0.05
    return results.sort_values("p_value_fdr").reset_index(drop=True)


def main():
    seed_names = find_seed_names()
    if not seed_names:
        print(f"No cleaned test_seed_N folders found under {CLEAN_ROOT} - run test_seed_clean_hpc.py first")
        return
    print(f"Found {len(seed_names)} seeds: {seed_names}")

    counters = find_counters(seed_names)
    print(f"hpmcounters present after cleaning: {counters}")

    results = run_tests(seed_names, counters)
    results.to_csv(STATS_OUT, index=False)
    print(results.to_string(index=False))
    print(f"\nSaved pre/post attack-onset test results to {STATS_OUT}")

    n_seeds = len(seed_names)
    if n_seeds < 15:
        print(
            f"\nCaveat: only {n_seeds} run(s) per type. Treat any 'significant' "
            "result above as a pilot-stage hypothesis, not a firm conclusion."
        )


if __name__ == "__main__":
    main()
