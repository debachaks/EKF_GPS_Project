/*
 * ekf_config_stf.h
 *
 * Config for the STF (Strong Tracking Filter) hardware-signature
 * experiment. Identical to board_firmware/ekf_config_new.h, plus
 * LAMBDA_WINDOW for the fading-factor's innovation-covariance
 * estimate and LAMBDA_APPLY_THRESHOLD for main_ekf_stf_applied.c's
 * gated P-scaling.
 *
 * IMPORTANT: this is a HARDWARE-SIGNATURE test only. The Python
 * simulation (Trajectory/simulate_ekf_stf.py) showed that applying
 * lambda_k to the real state covariance ACTIVELY SUPPRESSES drift
 * detection (nis_alarm dropped from 51/299 to 2/299) -- the fading
 * factor inflates P, which increases the Kalman gain, which lets
 * the filter absorb the drift faster, closing the very gap that
 * made it detectable. So main_ekf_stf.c computes lambda_k every
 * step (for the extra division-heavy computation and its hardware
 * footprint) but does NOT use it to scale the real P/x update --
 * detection behavior stays identical to main_ekf_new.c. See
 * main_ekf_stf.c's header comment for the full rationale.
 *
 * CRITICAL: Include this file BEFORE tinyekf.h
 */

#ifndef EKF_CONFIG_STF_H_
#define EKF_CONFIG_STF_H_

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
 * noise (m/s^2). See board_firmware/ekf_config_new.h for the
 * derivation (~180 ug/sqrt(Hz) MEMS noise density at ~1 Hz).
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
 * STRONG TRACKING FILTER (lambda_k) WINDOW
 * Size of the rolling window used to estimate the empirical
 * innovation covariance V_k that lambda_k is derived from.
 * Matches WINDOW_SIZE by default; kept as its own constant since
 * it's conceptually a separate knob (ANIS smooths NIS, this
 * smooths the raw innovation covariance -- no reason they must be
 * the same size).
 * ============================================================ */
#define LAMBDA_WINDOW     10

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
#define ATTACK_MODE   "drift"

#endif /* EKF_CONFIG_STF_H_ */
