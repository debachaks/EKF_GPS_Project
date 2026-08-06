"""
simulate_ekf_new.py

Python replica of main_ekf_new.c's exact EKF math (predict/update/NIS),
run entirely off-device against the trajectories this session already
generates in-memory (same true/measured/accel generation as
generate_trajectories_new.py -- imported directly, not re-typed).

Purpose: validate the GPS+IMU design (ACCEL_SIGMA-driven Q, accel-driven
predict step) BEFORE spending a hardware build cycle on it. Checks two
things per mode:
  1. Does drift's NIS actually rise above baseline now (the whole point
     of adding the accelerometer)?
  2. Does normal-mode NIS stay well-calibrated (mean near the chi2(3)
     expectation of ~3, not blown up by an overconfident Q)?

Mirrors main_ekf_new.c's ekf_step() ordering exactly, including using
the POST-update P (not the predicted/prior P) to compute NIS -- that's
what the real firmware does, so this replica matches it faithfully
rather than silently "fixing" it.
"""

import numpy as np

from generate_trajectories_new import (
    parse_ekf_config, generate_true_trajectory, compute_true_accel,
    generate_imu_accel, generate_measured_trajectory,
    EKF_CONFIG_PATH, TRAJ_LEN,
)

NIS_THRESHOLD = 11.345
ANIS_THRESHOLD = 4.377
WINDOW_SIZE = 10

MODES = ["normal", "jump", "drift", "replay"]
SEEDS = list(range(1, 11))   # 10 independent seeds per mode


def run_ekf(true_traj, measured_traj, accel_traj, cfg):
    T = cfg["T_STEP"]
    accel_sigma = cfg["ACCEL_SIGMA"]

    # Q_mat, matching ekf_config_new.h's XYZ0..XYZ3 formula
    xyz0 = accel_sigma**2 * T**3 / 3.0
    xyz1 = accel_sigma**2 * T**2 / 2.0
    xyz2 = xyz1
    xyz3 = accel_sigma**2 * T
    Q = np.zeros((6, 6))
    for j in range(3):
        Q[2*j, 2*j]     = xyz0
        Q[2*j, 2*j+1]   = xyz1
        Q[2*j+1, 2*j]   = xyz2
        Q[2*j+1, 2*j+1] = xyz3

    sigma_h = 5.0
    sigma_v = 10.0
    R = np.diag([sigma_h**2, sigma_h**2, sigma_v**2])

    H = np.zeros((3, 6))
    H[0, 0] = 1.0
    H[1, 2] = 1.0
    H[2, 4] = 1.0

    x = np.zeros(6)
    x[0], x[2], x[4] = true_traj[0]
    P = np.eye(6) * 10.0   # P0_INIT

    nis_buf = []
    nis_list = []
    anis_list = []

    for t in range(1, TRAJ_LEN):
        ax, ay, az = accel_traj[t]
        accel = [ax, ay, az]

        # run_model: accel-driven predict
        fx = np.zeros(6)
        F = np.eye(6)
        for j in range(3):
            v = x[2*j+1]
            a = accel[j]
            fx[2*j]   = x[2*j] + T*v + 0.5*T*T*a
            fx[2*j+1] = v + T*a
            F[2*j, 2*j+1] = T
        hx = np.array([fx[0], fx[2], fx[4]])

        # ekf_predict
        x = fx.copy()
        P = F @ P @ F.T + Q

        # ekf_update
        z = np.array(measured_traj[t])
        S = H @ P @ H.T + R
        G = P @ H.T @ np.linalg.inv(S)
        inn = z - hx
        x = x + G @ inn
        P = (np.eye(6) - G @ H) @ P

        # NIS -- uses POST-update P, matching main_ekf_new.c exactly
        S_nis = H @ P @ H.T + R
        nis = float(inn @ np.linalg.inv(S_nis) @ inn)

        nis_buf.append(nis)
        if len(nis_buf) > WINDOW_SIZE:
            nis_buf.pop(0)
        anis = sum(nis_buf) / len(nis_buf)

        nis_list.append(nis)
        anis_list.append(anis)

    return np.array(nis_list), np.array(anis_list)


def main():
    cfg = parse_ekf_config(EKF_CONFIG_PATH)
    true_traj = generate_true_trajectory()
    true_accel = compute_true_accel(true_traj, cfg["T_STEP"])

    print(f"\n{'mode':8s} {'seed':>4s} {'NIS mean':>10s} {'NIS max':>12s} "
          f"{'nis_alarm':>10s} {'ANIS mean':>10s} {'anis_alarm':>11s}")

    summary = {mode: {"nis_mean": [], "nis_alarm": [], "anis_alarm": []} for mode in MODES}

    for mode in MODES:
        for seed in SEEDS:
            measured_traj = generate_measured_trajectory(true_traj, mode, cfg, seed)
            accel_traj = generate_imu_accel(true_accel, cfg, seed)
            nis, anis = run_ekf(true_traj, measured_traj, accel_traj, cfg)

            nis_alarm_count = int((nis > NIS_THRESHOLD).sum())
            anis_alarm_count = int((anis > ANIS_THRESHOLD).sum())

            print(f"{mode:8s} {seed:4d} {nis.mean():10.3f} {nis.max():12.3f} "
                  f"{nis_alarm_count:10d} {anis.mean():10.3f} {anis_alarm_count:11d}")

            summary[mode]["nis_mean"].append(nis.mean())
            summary[mode]["nis_alarm"].append(nis_alarm_count)
            summary[mode]["anis_alarm"].append(anis_alarm_count)

    print(f"\n{'='*70}\nSUMMARY (avg over {len(SEEDS)} seeds per mode)\n{'='*70}")
    print(f"{'mode':8s} {'avg NIS mean':>14s} {'avg nis_alarm':>14s} {'avg anis_alarm':>15s}")
    for mode in MODES:
        s = summary[mode]
        print(f"{mode:8s} {np.mean(s['nis_mean']):14.3f} "
              f"{np.mean(s['nis_alarm']):14.2f} {np.mean(s['anis_alarm']):15.2f}")


if __name__ == "__main__":
    main()
