"""
simulate_ekf_stf.py

Python-only test of a Strong Tracking Filter (STF) fading factor
lambda_k layered on top of the GPS+IMU EKF from simulate_ekf_new.py,
BEFORE writing any C. Purpose: find out whether the adaptive idea is
actually worth the extra computation, using the same offline
validation approach that de-risked the IMU addition.

STF mechanics (classic single-scalar fading factor, Zhou & Frank
1996, simplified):

  1. Raw innovation e_k = z_k - H*fx_k  (fx_k = accel-driven
     prediction, BEFORE any lambda scaling -- lambda must be computed
     from an unbiased residual, not one it has already influenced).
  2. Windowed empirical innovation covariance:
       V_k = mean(e_i @ e_i^T) over the last WINDOW steps.
  3. "Theoretical" contribution from propagated state uncertainty
     (isolated from Q and R):
       M_k = H @ (F P F^T) @ H^T
       N_k = V_k - R
  4. lambda_k = max(1, trace(N_k) / trace(M_k))
     Floored at 1 -- STF only ever INFLATES uncertainty, never
     shrinks it below the nominal filter's.
  5. Predict step becomes:
       P_pred = lambda_k * (F P F^T) + Q
     instead of the plain P_pred = F P F^T + Q.

Everything else (update step, NIS/ANIS formulas, thresholds) stays
identical to simulate_ekf_new.py's run_ekf(), so results are directly
comparable.
"""

import numpy as np

from generate_trajectories_new import (
    parse_ekf_config, generate_true_trajectory, compute_true_accel,
    generate_imu_accel, generate_measured_trajectory,
    EKF_CONFIG_PATH, TRAJ_LEN,
)
from simulate_ekf_new import run_ekf   # baseline (non-adaptive) GPS+IMU EKF

NIS_THRESHOLD = 11.345
ANIS_THRESHOLD = 4.377
WINDOW_SIZE = 10
LAMBDA_WINDOW = 10          # window for the innovation-covariance estimate
LAMBDA_ALARM_THRESHOLD = 1.5  # candidate alarm threshold on lambda_k itself

MODES = ["normal", "jump", "drift", "replay"]
SEEDS = list(range(1, 11))


def run_ekf_stf(true_traj, measured_traj, accel_traj, cfg):
    T = cfg["T_STEP"]
    accel_sigma = cfg["ACCEL_SIGMA"]

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
    P = np.eye(6) * 10.0

    nis_buf, nis_list, anis_list = [], [], []
    lambda_buf_inn = []   # raw innovations for the lambda_k covariance window
    lambda_list = []

    for t in range(1, TRAJ_LEN):
        ax, ay, az = accel_traj[t]
        accel = [ax, ay, az]

        fx = np.zeros(6)
        F = np.eye(6)
        for j in range(3):
            v = x[2*j+1]
            a = accel[j]
            fx[2*j]   = x[2*j] + T*v + 0.5*T*T*a
            fx[2*j+1] = v + T*a
            F[2*j, 2*j+1] = T
        hx = np.array([fx[0], fx[2], fx[4]])

        z = np.array(measured_traj[t])
        e_raw = z - hx   # raw innovation, BEFORE lambda -- drives lambda_k itself

        lambda_buf_inn.append(e_raw)
        if len(lambda_buf_inn) > LAMBDA_WINDOW:
            lambda_buf_inn.pop(0)

        FPFt = F @ P @ F.T
        if len(lambda_buf_inn) >= 3:   # need a few samples before trusting V_k
            V_k = np.mean([np.outer(e, e) for e in lambda_buf_inn], axis=0)
            M_k = H @ FPFt @ H.T
            N_k = V_k - R
            m_tr = np.trace(M_k)
            lam = max(1.0, np.trace(N_k) / m_tr) if m_tr > 1e-9 else 1.0
        else:
            lam = 1.0

        lambda_list.append(lam)

        # predict, WITH lambda scaling on the propagated-uncertainty term only
        x = fx.copy()
        P = lam * FPFt + Q

        # update (unchanged)
        S = H @ P @ H.T + R
        G = P @ H.T @ np.linalg.inv(S)
        x = x + G @ e_raw
        P = (np.eye(6) - G @ H) @ P

        S_nis = H @ P @ H.T + R
        nis = float(e_raw @ np.linalg.inv(S_nis) @ e_raw)

        nis_buf.append(nis)
        if len(nis_buf) > WINDOW_SIZE:
            nis_buf.pop(0)
        anis = sum(nis_buf) / len(nis_buf)

        nis_list.append(nis)
        anis_list.append(anis)

    return np.array(nis_list), np.array(anis_list), np.array(lambda_list)


def main():
    cfg = parse_ekf_config(EKF_CONFIG_PATH)
    true_traj = generate_true_trajectory()
    true_accel = compute_true_accel(true_traj, cfg["T_STEP"])

    print(f"\n{'mode':8s} {'seed':>4s} "
          f"{'NIS mean':>9s} {'nis_alrm':>9s} {'anis_alrm':>10s} | "
          f"{'STF nis_alrm':>13s} {'STF anis_alrm':>14s} {'lam mean':>9s} {'lam max':>8s} {'lam_alrm':>9s}")

    summary = {m: {"base_nis_alarm": [], "base_anis_alarm": [],
                    "stf_nis_alarm": [], "stf_anis_alarm": [], "lam_alarm": []} for m in MODES}

    for mode in MODES:
        for seed in SEEDS:
            measured_traj = generate_measured_trajectory(true_traj, mode, cfg, seed)
            accel_traj = generate_imu_accel(true_accel, cfg, seed)

            base_nis, base_anis = run_ekf(true_traj, measured_traj, accel_traj, cfg)
            stf_nis, stf_anis, lam = run_ekf_stf(true_traj, measured_traj, accel_traj, cfg)

            base_nis_alarm = int((base_nis > NIS_THRESHOLD).sum())
            base_anis_alarm = int((base_anis > ANIS_THRESHOLD).sum())
            stf_nis_alarm = int((stf_nis > NIS_THRESHOLD).sum())
            stf_anis_alarm = int((stf_anis > ANIS_THRESHOLD).sum())
            lam_alarm = int((lam > LAMBDA_ALARM_THRESHOLD).sum())

            print(f"{mode:8s} {seed:4d} "
                  f"{base_nis.mean():9.3f} {base_nis_alarm:9d} {base_anis_alarm:10d} | "
                  f"{stf_nis_alarm:13d} {stf_anis_alarm:14d} {lam.mean():9.4f} {lam.max():8.3f} {lam_alarm:9d}")

            s = summary[mode]
            s["base_nis_alarm"].append(base_nis_alarm)
            s["base_anis_alarm"].append(base_anis_alarm)
            s["stf_nis_alarm"].append(stf_nis_alarm)
            s["stf_anis_alarm"].append(stf_anis_alarm)
            s["lam_alarm"].append(lam_alarm)

    print(f"\n{'='*95}\nSUMMARY (avg over {len(SEEDS)} seeds; alarm counts out of {TRAJ_LEN-1} steps)\n{'='*95}")
    print(f"{'mode':8s} {'base nis_alrm':>14s} {'base anis_alrm':>15s} "
          f"{'STF nis_alrm':>13s} {'STF anis_alrm':>14s} {'lam_alrm':>9s}")
    for mode in MODES:
        s = summary[mode]
        print(f"{mode:8s} {np.mean(s['base_nis_alarm']):14.2f} {np.mean(s['base_anis_alarm']):15.2f} "
              f"{np.mean(s['stf_nis_alarm']):13.2f} {np.mean(s['stf_anis_alarm']):14.2f} "
              f"{np.mean(s['lam_alarm']):9.2f}")


if __name__ == "__main__":
    main()
