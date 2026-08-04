import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Inputs
ATTACK_WINDOWED_PATH = os.path.join(RESULTS_DIR, "trend_score_windowed_results.csv")
NORMAL_THRESH_PATH = os.path.join(RESULTS_DIR, "normal_trend_score_thresholds.csv")

# Outputs
FLAGGED_WINDOWS_PATH = os.path.join(RESULTS_DIR, "attack_flagged_windows.csv")
WINDOW_LEVEL_OUTPUT_PATH = os.path.join(RESULTS_DIR, "attack_with_thresholds.csv")
TRIAL_SUMMARY_PATH = os.path.join(RESULTS_DIR, "attack_trial_summary.csv")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    attack_df = pd.read_csv(ATTACK_WINDOWED_PATH)
    thresh_df = pd.read_csv(NORMAL_THRESH_PATH)

    required_attack_cols = {"counter", "mode", "seed", "D"}
    required_thresh_cols = {"counter", "threshold"}

    missing_attack = required_attack_cols - set(attack_df.columns)
    missing_thresh = required_thresh_cols - set(thresh_df.columns)

    if missing_attack:
        raise ValueError(f"Attack file missing columns: {sorted(missing_attack)}")
    if missing_thresh:
        raise ValueError(f"Threshold file missing columns: {sorted(missing_thresh)}")

    thresh_df = thresh_df[["counter", "threshold"]].copy()

    merged = attack_df.merge(thresh_df, on="counter", how="left")

    if merged["threshold"].isna().any():
        missing_counters = sorted(merged.loc[merged["threshold"].isna(), "counter"].unique().tolist())
        raise ValueError(f"Missing thresholds for counters: {missing_counters}")

    # Compare every attack window D against the normal threshold for that counter
    merged["exceeds_threshold"] = merged["D"] > merged["threshold"]

    # Save full window-level output
    merged.to_csv(WINDOW_LEVEL_OUTPUT_PATH, index=False)

    # Save only flagged windows
    flagged = merged[merged["exceeds_threshold"]].copy()
    flagged = flagged.sort_values(["counter", "mode", "seed", "D"], ascending=[True, True, True, False])
    flagged.to_csv(FLAGGED_WINDOWS_PATH, index=False)

    # Trial-level summary: how many windows exceeded threshold in each trial
    trial_summary = (
        merged.groupby(["counter", "mode", "seed"], as_index=False)
        .agg(
            num_windows=("D", "size"),
            num_flagged_windows=("exceeds_threshold", "sum"),
            max_D=("D", "max"),
            threshold=("threshold", "first"),
        )
    )
    trial_summary["trial_exceeds_threshold"] = trial_summary["max_D"] > trial_summary["threshold"]
    trial_summary = trial_summary.sort_values(["counter", "mode", "seed"])
    trial_summary.to_csv(TRIAL_SUMMARY_PATH, index=False)

    print(f"Saved full window-level comparison to: {WINDOW_LEVEL_OUTPUT_PATH}")
    print(f"Saved flagged windows to: {FLAGGED_WINDOWS_PATH}")
    print(f"Saved trial summary to: {TRIAL_SUMMARY_PATH}")

    print("\nFlagged windows by counter/mode:")
    print(
        flagged.groupby(["counter", "mode"], as_index=False)
        .size()
        .rename(columns={"size": "num_flagged_windows"})
        .to_string(index=False)
    )

    print("\nTrial summary:")
    print(trial_summary.to_string(index=False))


if __name__ == "__main__":
    main()