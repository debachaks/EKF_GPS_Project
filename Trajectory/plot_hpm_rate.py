"""Per the whiteboard sketch (item 5): for each counter, two panels --

  a) HPM_x(t): raw counter value vs. EKF iteration.
  b) rate(t) = (HPM_x(k) - HPM_x(k-1)) / (mcycle(k) - mcycle(k-1))

Both panels: black = mean across all 20 normal seeds (REFERENCE_SEEDS),
orange = jump for SEED, green = drift for SEED.

Reads adaptive_analysis_board_firmware/plots/seed<N>_newMapping/<mode>_hpc<N>.csv
(iter, mcycle, hpmcounter3-10, all hex-encoded -- same hex_to_int
convention as zscore_baseline.py).
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PLOTS_ROOT = os.path.join(PROJECT_ROOT, "adaptive_analysis_board_firmware", "plots")
OUT_DIR = os.path.join(SCRIPT_DIR, "plots")

SEED = 1
REFERENCE_SEEDS = list(range(1, 21))  # seed1-20_newMapping -- full set, hpc data confirmed present for all 20
COUNTERS = [3, 4, 5, 8, 10]

MODE_COLORS = {
    "normal": "#111111",
    "jump": "#E07B1A",
    "drift": "#2E8B3D",
}


def hex_to_int(val):
    s = str(val).strip()
    if s.lower().startswith("0x"):
        return int(s, 16)
    return int(float(s))


def load_hpc(seed, mode):
    seed_dir = os.path.join(PLOTS_ROOT, f"seed{seed}_newMapping")
    matches = glob.glob(os.path.join(seed_dir, f"{mode}_hpc*.csv"))
    df = pd.read_csv(matches[0])
    df["iter"] = df["iter"].map(hex_to_int)
    df["mcycle"] = df["mcycle"].map(hex_to_int)
    for c in range(3, 11):
        df[f"hpmcounter{c}"] = df[f"hpmcounter{c}"].map(hex_to_int)
    return df.sort_values("iter").reset_index(drop=True)


def rate_curve(df, counter_col):
    hpm = df[counter_col].to_numpy()
    mcycle = df["mcycle"].to_numpy()
    d_hpm = np.diff(hpm)
    d_mcycle = np.diff(mcycle)
    return df["iter"].to_numpy()[1:], d_hpm / d_mcycle


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    normal_dfs = {s: load_hpc(s, "normal") for s in REFERENCE_SEEDS}
    attack_dfs = {mode: load_hpc(SEED, mode) for mode in ("jump", "drift")}
    seed1_normal = normal_dfs[SEED]

    plt.rcParams.update({"font.size": 12, "font.family": "serif"})

    for c in COUNTERS:
        col = f"hpmcounter{c}"

        # panel a: raw value
        normal_stack = np.stack([normal_dfs[s][col].to_numpy() for s in REFERENCE_SEEDS])
        mean_raw = normal_stack.mean(axis=0)
        iters = seed1_normal["iter"].to_numpy()

        raw_curves = {"normal": mean_raw}
        for mode in ("jump", "drift"):
            raw_curves[mode] = attack_dfs[mode][col].to_numpy()

        # panel b: rate
        normal_rates = np.stack([rate_curve(normal_dfs[s], col)[1] for s in REFERENCE_SEEDS])
        mean_rate = normal_rates.mean(axis=0)
        rate_iters = rate_curve(seed1_normal, col)[0]

        rate_curves = {"normal": mean_rate}
        for mode in ("jump", "drift"):
            rate_curves[mode] = rate_curve(attack_dfs[mode], col)[1]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

        ax = axes[0]
        for mode, color in MODE_COLORS.items():
            ax.plot(iters, raw_curves[mode], color=color, linewidth=1.8, label=mode)
        ax.set_xlabel("EKF iteration")
        ax.set_ylabel(f"HPM$_{{{c}}}$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", fontsize=10, frameon=False)

        ax = axes[1]
        for mode, color in MODE_COLORS.items():
            ax.plot(rate_iters, rate_curves[mode], color=color, linewidth=1.8, label=mode)
        ax.set_xlabel("EKF iteration")
        ax.set_ylabel(r"$\frac{\Delta HPM_{%d}}{\Delta mcycle}$" % c)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", fontsize=10, frameon=False)

        fig.suptitle(f"hpmcounter{c} — seed {SEED} vs. {len(REFERENCE_SEEDS)}-seed normal mean", y=1.02)
        fig.tight_layout()

        out_path = os.path.join(OUT_DIR, f"hpmcounter{c}_raw_rate.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
