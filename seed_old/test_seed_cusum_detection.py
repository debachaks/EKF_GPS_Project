"""CUSUM (cumulative sum) sequential change detection, per counter.

Unlike the whole-run/pre-post tests elsewhere in this folder, CUSUM is an
online, per-sample method: it accumulates deviations from a normal-mode
reference (mu0, sigma0) as a run streams in, and signals as soon as that
accumulated evidence crosses a threshold. It's specifically suited to a
small, sustained shift rather than a single dramatic spike (see the
per-timestep hpmcounter5 plot, which showed no localized spike - just a
diffuse, whole-run bias), which is exactly the kind of signal this dataset
has shown so far.

Reference (mu0, sigma0) is fit on pooled row-level rate from normal
training runs. The alarm threshold h is calibrated via leave-one-out on
the normal runs themselves (fit on the other 19, test on the held-out
one), taking the max of the 20 held-out runs' peak CUSUM statistics - by
construction this gives 0/20 false alarms on the calibration set itself,
so the reported false-alarm rate on attack-run "pre-onset" segments below
is the more honest number to look at.

Crucially: every detection is checked against the KNOWN attack onset
(t=150/299, fraction 0.5 - see test_seed_pre_post_attack_analysis.py for
how this was established) to see whether the alarm fired before or after
the attack actually started. A real attack detector should fire after; if
it mostly fires before, that's the same run-order confound showing up in
a new form, not genuine detection.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from data_preprocessing import hex_to_int  # noqa: E402

RAW_ROOT = SCRIPT_DIR
CLEAN_ROOT = os.path.join(SCRIPT_DIR, "CLEAN_HPC_TEST_SEED")
ANOMALY_TYPES = ["drift", "jump", "replay"]
ALL_TYPES = ["normal"] + ANOMALY_TYPES
K_FACTOR = 0.5  # CUSUM slack, in units of sigma0 (standard default: half the shift you want to catch)
DEFAULT_ONSET_FRACTION = 0.5


def find_seed_names():
    dirs = sorted(d for d in glob.glob(os.path.join(CLEAN_ROOT, "test_seed_[0-9]*")) if os.path.isdir(d))
    return [os.path.basename(d) for d in dirs]


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


def row_rate_series(seed_name, run_type, counter):
    path = os.path.join(CLEAN_ROOT, seed_name, f"ekf_{run_type}_hpc.csv")
    df = pd.read_csv(path)
    mcycle_delta = df["mcycle"].map(hex_to_int).diff()
    counter_delta = df[counter].map(hex_to_int).diff()
    rate = (counter_delta / mcycle_delta).iloc[1:].replace([np.inf, -np.inf], np.nan)
    valid = rate.notna()
    return rate[valid].to_numpy()


def cusum(series, mu0, sigma0, k_factor=K_FACTOR):
    k = k_factor * sigma0
    n = len(series)
    c_pos = np.zeros(n)
    c_neg = np.zeros(n)
    for i in range(n):
        prev_pos = c_pos[i - 1] if i > 0 else 0.0
        prev_neg = c_neg[i - 1] if i > 0 else 0.0
        c_pos[i] = max(0.0, prev_pos + (series[i] - mu0 - k))
        c_neg[i] = max(0.0, prev_neg + (mu0 - series[i] - k))
    return c_pos, c_neg


def first_crossing(c_pos, c_neg, h):
    combined = np.maximum(c_pos, c_neg)
    over = np.where(combined > h)[0]
    return int(over[0]) if len(over) else None


def calibrate_threshold(seed_names, counter):
    """Leave-one-out: fit (mu0, sigma0) on 19 normal runs, find the peak
    CUSUM statistic on the held-out one. Threshold = max across all 20
    held-out peaks, so the calibration set itself has 0/20 false alarms."""
    series_by_seed = {s: row_rate_series(s, "normal", counter) for s in seed_names}
    peak_stats = []
    for held_out in seed_names:
        train_pool = np.concatenate([series_by_seed[s] for s in seed_names if s != held_out])
        mu0, sigma0 = train_pool.mean(), train_pool.std()
        c_pos, c_neg = cusum(series_by_seed[held_out], mu0, sigma0)
        peak_stats.append(np.maximum(c_pos, c_neg).max())
    return max(peak_stats), np.array(peak_stats)


def evaluate_counter(seed_names, counter):
    threshold, loo_peaks = calibrate_threshold(seed_names, counter)

    all_normal = np.concatenate([row_rate_series(s, "normal", counter) for s in seed_names])
    mu0_final, sigma0_final = all_normal.mean(), all_normal.std()

    rows = []
    for run_type in ANOMALY_TYPES:
        for seed_name in seed_names:
            series = row_rate_series(seed_name, run_type, counter)
            c_pos, c_neg = cusum(series, mu0_final, sigma0_final)
            crossing = first_crossing(c_pos, c_neg, threshold)
            onset_row = int(round(onset_fraction(seed_name, run_type) * len(series)))
            detected = crossing is not None
            fired_before_onset = detected and crossing < onset_row
            rows.append({
                "counter": counter, "run_type": run_type, "seed": seed_name,
                "detected": detected,
                "crossing_row": crossing,
                "onset_row": onset_row,
                "n_rows": len(series),
                "fired_before_onset": fired_before_onset,
                "latency_rows_after_onset": (crossing - onset_row) if (detected and not fired_before_onset) else None,
            })

    return pd.DataFrame(rows), threshold, loo_peaks


def main():
    seed_names = find_seed_names()
    if not seed_names:
        print(f"No cleaned test_seed_N folders found under {CLEAN_ROOT} - run test_seed_clean_hpc.py first")
        return
    print(f"Found {len(seed_names)} seeds\n")

    counters = find_counters(seed_names)
    all_results = []

    for counter in counters:
        results, threshold, loo_peaks = evaluate_counter(seed_names, counter)
        all_results.append(results)

        print(f"=== {counter} (threshold={threshold:.4g}, calibrated via leave-one-out normal) ===")
        for run_type in ANOMALY_TYPES:
            sub = results[results.run_type == run_type]
            n_detected = sub.detected.sum()
            n_before = sub.fired_before_onset.sum()
            n_after = n_detected - n_before
            after_latencies = sub["latency_rows_after_onset"].dropna()
            latency_str = f"median latency={after_latencies.median():.1f} rows after onset" if len(after_latencies) else "n/a"
            print(f"  {run_type}: {n_detected}/{len(sub)} detected "
                  f"({n_before} fired BEFORE onset [confound-like], {n_after} fired AFTER onset [real detection]) - {latency_str}")
        print()

    out_path = os.path.join(SCRIPT_DIR, "test_seed_cusum_detection_results.csv")
    pd.concat(all_results, ignore_index=True).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
