import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MAX_D_PATH = os.path.join(SCRIPT_DIR, "normal_trend_score_max_per_trial.csv")
THRESHOLD_PATH = os.path.join(SCRIPT_DIR, "normal_trend_score_thresholds.csv")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "normal_D_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

max_df = pd.read_csv(MAX_D_PATH)
thresh_df = pd.read_csv(THRESHOLD_PATH)

for counter in sorted(max_df["counter"].unique()):

    df = max_df[max_df["counter"] == counter].sort_values("seed")

    threshold = thresh_df.loc[
        thresh_df["counter"] == counter, "threshold"
    ].values[0]

    ###############################################
    # Plot 1: Max D vs Seed
    ###############################################
    plt.figure(figsize=(8,4))

    plt.plot(df["seed"], df["max_D"], marker="o")
    plt.axhline(
        threshold,
        color="red",
        linestyle="--",
        label=f"95% Threshold = {threshold:.2f}"
    )

    plt.xlabel("Seed")
    plt.ylabel("Maximum D")
    plt.title(f"Counter {counter}: Maximum Normal D")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"counter_{counter}_maxD_vs_seed.png"
        )
    )
    plt.close()

    ###############################################
    # Plot 2: Histogram
    ###############################################
    plt.figure(figsize=(6,4))

    plt.hist(df["max_D"], bins=8)

    plt.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"95% Threshold = {threshold:.2f}"
    )

    plt.xlabel("Maximum D")
    plt.ylabel("Count")
    plt.title(f"Counter {counter}: Distribution of Max D")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"counter_{counter}_histogram.png"
        )
    )
    plt.close()

print(f"Plots saved to {OUTPUT_DIR}")