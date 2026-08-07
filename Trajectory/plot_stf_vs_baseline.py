"""Drift seed1 NIS: baseline GPS+IMU EKF vs the same filter with the
STF fading factor applied. Same underlying GPS/accel data in both --
isolates the effect of the fading factor itself."""

import os

import matplotlib.pyplot as plt

from generate_trajectories_new import (
    parse_ekf_config, generate_true_trajectory, compute_true_accel,
    generate_imu_accel, generate_measured_trajectory, EKF_CONFIG_PATH,
)
from simulate_ekf_new import run_ekf
from simulate_ekf_stf import run_ekf_stf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"
BASE_COLOR = "#2a78d6"
STF_COLOR = "#c0392b"


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

    cfg = parse_ekf_config(EKF_CONFIG_PATH)
    true_traj = generate_true_trajectory()
    true_accel = compute_true_accel(true_traj, cfg["T_STEP"])
    measured_traj = generate_measured_trajectory(true_traj, "drift", cfg, seed=1)
    accel_traj = generate_imu_accel(true_accel, cfg, seed=1)

    base_nis, base_anis = run_ekf(true_traj, measured_traj, accel_traj, cfg)
    stf_nis, stf_anis, lam = run_ekf_stf(true_traj, measured_traj, accel_traj, cfg)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), facecolor=SURFACE)

    ax1.plot(range(1, len(base_nis)+1), base_nis, color=BASE_COLOR, linewidth=1.5, label="baseline (no STF)")
    ax1.plot(range(1, len(stf_nis)+1), stf_nis, color=STF_COLOR, linewidth=1.5, label="with STF fading factor")
    ax1.axhline(11.345, color=AXIS_COLOR, linewidth=1, linestyle="--", label="NIS_THRESHOLD")
    ax1.set_title("drift seed1: NIS, baseline vs STF", color=TEXT_PRIMARY, fontsize=12)
    ax1.set_xlabel("t (entry index)", color=TEXT_MUTED)
    ax1.set_ylabel("nis", color=TEXT_MUTED)
    ax1.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
    style_axes(ax1)

    ax2.plot(range(1, len(lam)+1), lam, color=STF_COLOR, linewidth=1.5, label="lambda_k")
    ax2.axhline(1.5, color=AXIS_COLOR, linewidth=1, linestyle="--", label="candidate lambda alarm (1.5)")
    ax2.set_title("drift seed1: fading factor lambda_k over time", color=TEXT_PRIMARY, fontsize=12)
    ax2.set_xlabel("t (entry index)", color=TEXT_MUTED)
    ax2.set_ylabel("lambda_k", color=TEXT_MUTED)
    ax2.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
    style_axes(ax2)

    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, "drift_seed1_stf_vs_baseline.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
