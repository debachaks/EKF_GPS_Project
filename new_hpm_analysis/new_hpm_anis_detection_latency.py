"""ANIS-based anomaly detection latency for the new-HPM-mapping runs
(seed_new_1..5). Same method as anis_detection_latency.py at the project
root, just pointed at seed_new_1..5/ instead of seed_1_data..seed_5_data/,
kept separate since this is a distinct set of runs.

diag files aren't affected by HPM event mapping (they're EKF filter
diagnostics, not hardware counters), so no cleaning step applies here.
"""

import glob
import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DIAG_GLOB = os.path.join(PROJECT_ROOT, "seed_new_[0-9]*", "ekf_diag_*.csv")
OUT_PATH = os.path.join(SCRIPT_DIR, "new_hpm_anis_detection_latency.csv")


def find_diag_files():
    files = glob.glob(DIAG_GLOB)
    return sorted(f for f in files if "normal" not in os.path.basename(f))


def analyze_file(path):
    df = pd.read_csv(path)

    onset_rows = df.index[df["attack_active"] == 1]
    if len(onset_rows) == 0:
        return None
    onset_t = df.loc[onset_rows.min(), "t"]

    before_onset = df[df["t"] < onset_t]
    false_alarms_before_onset = int(before_onset["anis_alarm"].sum())

    after_onset = df[df["t"] >= onset_t]
    alarms_after_onset = after_onset[after_onset["anis_alarm"] == 1]
    total_alarms_after_onset = len(alarms_after_onset)

    if total_alarms_after_onset > 0:
        detection_t = alarms_after_onset["t"].min()
        latency = detection_t - onset_t
    else:
        detection_t = None
        latency = None

    seed = os.path.basename(os.path.dirname(path))
    anomaly_type = os.path.basename(path).replace("ekf_diag_", "").replace(".csv", "")

    return {
        "seed": seed,
        "anomaly_type": anomaly_type,
        "onset_t": onset_t,
        "detection_t": detection_t,
        "latency": latency,
        "total_alarms_after_onset": total_alarms_after_onset,
        "false_alarms_before_onset": false_alarms_before_onset,
    }


def main():
    files = find_diag_files()
    if not files:
        print(f"No diag files found matching {DIAG_GLOB}")
        return

    rows = [analyze_file(f) for f in files]
    results = pd.DataFrame([r for r in rows if r is not None])
    results = results.sort_values(["anomaly_type", "seed"]).reset_index(drop=True)
    results.to_csv(OUT_PATH, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved to {OUT_PATH}\n")

    print("Latency summary by anomaly type (rows where detected):")
    detected = results.dropna(subset=["latency"])
    summary = detected.groupby("anomaly_type")["latency"].agg(["count", "mean", "median", "min", "max"])
    print(summary.to_string())

    never_detected = results[results["latency"].isna()]
    if len(never_detected):
        print("\nNever detected within the run:")
        print(never_detected[["seed", "anomaly_type"]].to_string(index=False))


if __name__ == "__main__":
    main()
