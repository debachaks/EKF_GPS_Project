/*
 * ekf_config_new.h
 *
 * NEW VERSION of ekf_config.h, extended for IMU (accelerometer)
 * integration. All EKF configuration for GPS+IMU spoofing
 * experiment.
 * Platform: SiFive 8-stage dual-issue RISC-V soft-core
 *           on Xilinx Versal SoC-FPGA
 *
 * CRITICAL: Include this file BEFORE tinyekf.h
 *
 * WHAT CHANGED FROM ekf_config.h:
 *   - Added ACCEL_SIGMA: real accelerometer sensor noise std,
 *     used both by generate_trajectories_new.py (to synthesize
 *     imu_accel_meas) and here (to build Q_mat).
 *   - Q_mat's XYZ0..XYZ3 now use ACCEL_SIGMA instead of
 *     SIGMA_MODEL -- previously Q represented "how much unmodeled
 *     acceleration could there be" (pure guesswork, since there
 *     was no accelerometer). Now that acceleration is genuinely
 *     measured and fed into the predict step (see main_ekf_new.c's
 *     run_model()), Q represents "how much do I trust that
 *     measurement" -- a real sensor noise spec, not a guess.
 *   - EKF_N/EKF_M UNCHANGED (still 6/3) -- the accelerometer is
 *     used as a control input to the existing 6-state model, not
 *     as new states. See main_ekf_new.c for why this is sufficient
 *     given the straight-line, constant-velocity true trajectory.
 */

#ifndef EKF_CONFIG_NEW_H_
#define EKF_CONFIG_NEW_H_

/* ============================================================
 * EKF DIMENSIONS
 * EKF_N = 6  state: [x, x_vel, y, y_vel, z, z_vel]
 * EKF_M = 3  measurement: [GPS_x, GPS_y, GPS_z] ECEF meters
 * ============================================================ */
#define EKF_N   6
#define EKF_M   3

/* ============================================================
 * FLOATING POINT PRECISION
 * Must be double — ECEF coords ~4,000,000m
 * float gives 400m rounding error
 * ============================================================ */
#define _float_t double

/* ============================================================
 * TIMESTEP
 * T_STEP = seconds between GPS fixes (1Hz update rate)
 * ============================================================ */
#define T_STEP   1.0

/* ============================================================
 * INITIAL COVARIANCE DIAGONAL
 * ============================================================ */
#define P0_INIT  10.0

/* ============================================================
 * PROCESS NOISE MATRIX Q (6x6)
 * ACCEL_SIGMA = std deviation of the accelerometer's OWN sensor
 * noise (m/s^2). Derived from a representative MEMS accelerometer
 * noise density (~180 ug/sqrt(Hz), e.g. BMI160-class) evaluated
 * at ~1 Hz effective bandwidth:
 *   sigma = noise_density * 1e-6 * 9.81 * sqrt(bandwidth_Hz)
 *         ~= 180e-6 * 9.81 * sqrt(1) ~= 0.0018 m/s^2
 * Rounded up slightly for margin.
 * ============================================================ */
#define ACCEL_SIGMA  0.003

#define XYZ0  (ACCEL_SIGMA*ACCEL_SIGMA*T_STEP*T_STEP*T_STEP/3.0)
#define XYZ1  (ACCEL_SIGMA*ACCEL_SIGMA*T_STEP*T_STEP/2.0)
#define XYZ2  (ACCEL_SIGMA*ACCEL_SIGMA*T_STEP*T_STEP/2.0)
#define XYZ3  (ACCEL_SIGMA*ACCEL_SIGMA*T_STEP)

static const double Q_mat[6*6] = {
    XYZ0, XYZ1,  0,    0,    0,    0,
    XYZ2, XYZ3,  0,    0,    0,    0,
    0,    0,    XYZ0, XYZ1,  0,    0,
    0,    0,    XYZ2, XYZ3,  0,    0,
    0,    0,    0,    0,    XYZ0, XYZ1,
    0,    0,    0,    0,    XYZ2, XYZ3,
};

/* ============================================================
 * MEASUREMENT NOISE MATRIX R (3x3)
 * SIGMA_H = horizontal std ~5m
 * SIGMA_V = vertical std  ~10m
 * ============================================================ */
#define SIGMA_H   5.0
#define SIGMA_V   10.0

static const double R_mat[3*3] = {
    SIGMA_H*SIGMA_H,  0,               0,
    0,                SIGMA_H*SIGMA_H,  0,
    0,                0,               SIGMA_V*SIGMA_V,
};

/* ============================================================
 * INNOVATION MONITOR THRESHOLDS
 * chi2(3, 0.99) = 11.345  single step
 * chi2(3, 0.95) =  7.815  averaged NIS
 * ============================================================ */
#define NIS_THRESHOLD    11.345
#define ANIS_THRESHOLD    4.377
#define WINDOW_SIZE       10

/* ============================================================
 * ATTACK PARAMETERS
 * ============================================================ */
#define JUMP_OFFSET_M   500.0
#define DRIFT_RATE_MS     1.0
#define DRIFT_MAX_M     200.0
#define REPLAY_DELAY       10
#define HISTORY_LEN        32

/* ============================================================
 * GPS NOISE PARAMETERS
 * sigma_h = horizontal noise std (meters)
 * sigma_v = vertical noise std (meters)
 * ============================================================ */
#define GPS_SIGMA_H   5.0
#define GPS_SIGMA_V   10.0

/* ============================================================
 * ATTACK TIMING
 * Attack switches on at this timestep
 * ============================================================ */
#define ATTACK_START  150

/* ============================================================
 * ATTACK MODE — change this line for each experiment
 * Options: "normal" "jump" "drift" "replay"
 * ============================================================ */
#define ATTACK_MODE   "replay"

#endif /* EKF_CONFIG_NEW_H_ */
