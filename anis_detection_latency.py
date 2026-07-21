"""ANIS-based anomaly detection latency for every seed_*_data/seed_new run.

For each ekf_diag_{drift,jump,replay}.csv (normal is excluded - there's no
attack onset to detect), this finds:
  - onset_t: the first t where attack_active flips to 1 (ground truth)
  - detection_t: the first t >= onset_t where anis_alarm fires
  - latency: detection_t - onset_t (in row/time-step units of `t`)
  - false_alarms_before_onset: anis_alarm firing while attack_active is still 0
  - total_alarms_after_onset: how many rows after onset had anis_alarm set
    (not just the first) - a rough sense of how persistent the alarm is

If anis_alarm never fires at/after onset, latency is left as NaN (not
detected within the run).
"""

import glob
import os

import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DIAG_GLOB_PATTERNS = [
    os.path.join(SRC_DIR, "seed_*_data", "ekf_diag_*.csv"),
    os.path.join(SRC_DIR, "seed_new", "ekf_diag_*.csv"),
]
OUT_PATH = os.path.join(SRC_DIR, "anis_detection_latency.csv")


def find_diag_files():
    files = []
    for pattern in DIAG_GLOB_PATTERNS:
        files.extend(glob.glob(pattern))
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
    rows = [analyze_file(f) for f in find_diag_files()]
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
