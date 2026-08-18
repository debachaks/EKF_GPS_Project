"""Two paper figures, per the whiteboard sketch, for one seed:

  1) e_GPS(t) = || z_t^spoofed - z_t^normal || [m]  -- the raw GPS
     measurement's deviation from ground truth (spoof_error column in
     ekf_diag_<mode>.csv, i.e. ||measured - true||; the closest available
     proxy to a spoofed-vs-normal measurement comparison, since the raw
     per-mode measurement stream itself isn't a logged column here --
     see trajectory_<mode>_seed<N>_new.h's traj_measured for that exact
     comparison if needed instead).

  2) e_EKF(t) = || P_hat_t - P_t^true || [m]  -- EKF position estimate's
     distance from ground truth, computed from filt_x/y/z and
     true_x/y/z (not a stored column).

Reads from adaptive_analysis_board_firmware/plots/seed<N>_newMapping/.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

SEED = 1
SEED_DIR = os.path.join(PROJECT_ROOT, "adaptive_analysis_board_firmware", "plots", f"seed{SEED}_newMapping")

MODE_COLORS = {
    "normal": "#111111",
    "jump": "#E07B1A",
    "drift": "#2E8B3D",
}


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    dfs = {mode: pd.read_csv(os.path.join(SEED_DIR, f"ekf_diag_{mode}.csv")) for mode in MODE_COLORS}
    for df in dfs.values():
        filt = df[["filt_x", "filt_y", "filt_z"]].to_numpy()
        true = df[["true_x", "true_y", "true_z"]].to_numpy()
        df["e_ekf"] = np.linalg.norm(filt - true, axis=1)

    plt.rcParams.update({"font.size": 12, "font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    panels = [
        ("spoof_error", r"$e_{GPS}(t) = \Vert z_t^{spoofed} - z_t^{normal} \Vert$ [m]"),
        ("e_ekf", r"$e_{EKF}(t) = \Vert \hat{P}_t - P_t^{true} \Vert$ [m]"),
    ]

    for ax, (col, ylabel) in zip(axes, panels):
        for mode, color in MODE_COLORS.items():
            df = dfs[mode]
            ax.plot(df["t"], df[col], color=color, linewidth=1.8, label=mode)
        ax.set_xlabel("EKF iteration")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", fontsize=10, frameon=False)

    fig.suptitle(f"Seed {SEED} — GPS measurement error vs. EKF estimation error", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"seed{SEED}_gps_ekf_error.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
