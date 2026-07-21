/*
 * ekf_config.h
 *
 * All EKF configuration for GPS spoofing experiment.
 * Platform: SiFive 8-stage dual-issue RISC-V soft-core
 *           on Xilinx Versal SoC-FPGA
 *
 * CRITICAL: Include this file BEFORE tinyekf.h
 */

#ifndef EKF_CONFIG_H_
#define EKF_CONFIG_H_

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
 * SIGMA_MODEL = std deviation of unexpected acceleration
 * ============================================================ */
#define SIGMA_MODEL  0.1

#define XYZ0  (SIGMA_MODEL*SIGMA_MODEL*T_STEP*T_STEP*T_STEP/3.0)
#define XYZ1  (SIGMA_MODEL*SIGMA_MODEL*T_STEP*T_STEP/2.0)
#define XYZ2  (SIGMA_MODEL*SIGMA_MODEL*T_STEP*T_STEP/2.0)
#define XYZ3  (SIGMA_MODEL*SIGMA_MODEL*T_STEP)

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

#endif /* EKF_CONFIG_H_ */
