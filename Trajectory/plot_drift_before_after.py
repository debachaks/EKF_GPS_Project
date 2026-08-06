"""Before/after comparison: drift seed1 NIS with the old GPS-only
filter vs the new GPS+IMU filter. Same underlying GPS noise draw in
both (generate_measured_trajectory's RNG logic is unchanged), so this
is an apples-to-apples comparison of filter design, not different data.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from generate_trajectories_new import (
    parse_ekf_config, generate_true_trajectory, compute_true_accel,
    generate_imu_accel, generate_measured_trajectory, EKF_CONFIG_PATH,
)
from simulate_ekf_new import run_ekf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"
OLD_COLOR = "#c0392b"
NEW_COLOR = "#2a78d6"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_MUTED)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    old = pd.read_csv(os.path.join(
        SCRIPT_DIR, "..", "original_pipeline", "seed_1_data", "ekf_diag_drift.csv"))

    cfg = parse_ekf_config(EKF_CONFIG_PATH)
    true_traj = generate_true_trajectory()
    true_accel = compute_true_accel(true_traj, cfg["T_STEP"])
    measured_traj = generate_measured_trajectory(true_traj, "drift", cfg, seed=1)
    accel_traj = generate_imu_accel(true_accel, cfg, seed=1)
    new_nis, new_anis = run_ekf(true_traj, measured_traj, accel_traj, cfg)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.plot(old["t"], old["nis"], color=OLD_COLOR, linewidth=1.5, label="old (GPS-only)")
    ax.plot(range(1, len(new_nis) + 1), new_nis, color=NEW_COLOR, linewidth=1.5, label="new (GPS+IMU)")
    ax.axhline(11.345, color=AXIS_COLOR, linewidth=1, linestyle="--", label="NIS_THRESHOLD")

    ax.set_title("drift seed1: NIS before vs after adding IMU", color=TEXT_PRIMARY, fontsize=12)
    ax.set_xlabel("t (entry index)", color=TEXT_MUTED)
    ax.set_ylabel("nis", color=TEXT_MUTED)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
    style_axes(ax)
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, "drift_seed1_nis_before_after.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
