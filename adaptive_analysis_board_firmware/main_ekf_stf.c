/*
 * main_ekf_stf.c
 *
 * HARDWARE-SIGNATURE TEST for a Strong Tracking Filter (STF) fading
 * factor lambda_k, layered on top of the GPS+IMU EKF from
 * board_firmware/main_ekf_new.c.
 *
 * ============================================================
 * WHY THIS DOES NOT USE lambda_k TO CORRECT THE STATE
 * ============================================================
 * Trajectory/simulate_ekf_stf.py tested the "adapt to track faster"
 * version of STF (scale the predicted covariance by lambda_k, let
 * it increase the Kalman gain) against 10 seeds x 4 modes, BEFORE
 * any of this was written in C. Result: it made detection WORSE,
 * not better --
 *
 *   drift nis_alarm:  baseline 51/299  ->  with lambda_k  2/299
 *   jump  nis_alarm: baseline 147/299  ->  with lambda_k  6.6/299
 *
 * Mechanism: inflating P increases the Kalman gain, which lets the
 * filter absorb the spoofed trajectory FASTER -- closing the very
 * prediction/measurement gap that made drift and jump detectable
 * in the first place. lambda_k itself also turned out to fire at
 * roughly the same rate in EVERY mode (~48-58 times/299 regardless
 * of attack), so it isn't a clean standalone detector either, in
 * this per-step-threshold form.
 *
 * SO WHAT IS THIS FILE FOR: the open question left after that
 * result was whether the EXTRA COMPUTATION lambda_k requires --
 * more divisions, more matrix multiplies, with attack-correlated
 * operand magnitudes (lambda_k's own mean DOES separate jump
 * cleanly: ~3.8-11.2 vs ~1.2 baseline) -- shows up as a hardware
 * signature on hpmcounter3/5, independent of whether it helps the
 * software-level NIS/ANIS statistic. Python can't answer that
 * (no pipeline/interlock timing model), so this file computes
 * lambda_k EVERY step, unconditionally (same instruction count
 * regardless of attack, so mcycle/minstret should stay
 * attack-invariant, same as the ablation study's finding for the
 * core EKF math), logs it to diag_log for offline analysis, but
 * NEVER applies it to ekf.P or ekf.x. Detection behavior
 * (nis/anis/alarms) should come out IDENTICAL to main_ekf_new.c on
 * the same trajectory -- only the extra lambda_k computation and
 * its hardware footprint are new.
 *
 * ============================================================
 * WHAT CHANGED FROM main_ekf_new.c
 * ============================================================
 *   - Added a LAMBDA_WINDOW-sized ring buffer of raw innovations,
 *     used to build V_k (empirical innovation covariance).
 *   - Added compute_lambda_k(): builds F*P*F^T (shadow copy, using
 *     the SAME F and PRE-predict P that ekf_predict() is about to
 *     consume), then M_k = H*(F P F^T)*H^T, N_k = V_k - R,
 *     lambda_k = max(1, trace(N_k)/trace(M_k)).
 *   - diag_t gained a `lambda_k` field; DIAG line format gained one
 *     more %.6f.
 *   - ekf_step() calls compute_lambda_k() BEFORE ekf_predict()
 *     (since ekf_predict() overwrites ekf.P), stores the result,
 *     then proceeds with predict/update exactly as main_ekf_new.c
 *     does -- Q_mat, R_mat, x, P are all untouched by lambda_k.
 * ============================================================
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include <metal/cpu.h>
#include <metal/hpm.h>
#include <metal/machine.h>

#include "uart.h"
#include "hpm_configure.h"
#include "ekf_config_stf.h"    /* MUST come before tinyekf.h. */
#include "tinyekf.h"
#include "trajectory.h"     /* provides traj_true[][3], traj_measured[][3],
                              and imu_accel_meas[][3] (drift/seed1 by
                              default in this folder -- swap for any
                              other Trajectory/trajectory_<mode>_seed<N>_new.h
                              to test other modes/seeds). */

/* ============================================================
 * EKF STATE
 * ============================================================ */

static ekf_t ekf;

/* ============================================================
 * SLIDING WINDOW FOR ANIS (unchanged from main_ekf_new.c)
 * ============================================================ */

static double nis_buf[WINDOW_SIZE];
static int    nis_head  = 0;
static int    nis_count = 0;
static double nis_sum   = 0.0;

static void window_push(double val) {
    if (nis_count == WINDOW_SIZE)
        nis_sum -= nis_buf[nis_head];
    else
        nis_count++;
    nis_buf[nis_head] = val;
    nis_sum  += val;
    nis_head  = (nis_head + 1) % WINDOW_SIZE;
}

static double window_mean(void) {
    return (nis_count > 0) ? (nis_sum / nis_count) : 0.0;
}

/* ============================================================
 * SLIDING WINDOW FOR lambda_k'S EMPIRICAL INNOVATION COVARIANCE
 * Stores the last LAMBDA_WINDOW raw innovation vectors (EKF_M each)
 * so compute_lambda_k() can build V_k = mean(e_i * e_i^T).
 * ============================================================ */

static double lam_buf[LAMBDA_WINDOW][EKF_M];
static int    lam_head  = 0;
static int    lam_count = 0;

static void lambda_window_push(const double e[EKF_M]) {
    int i;
    for (i = 0; i < EKF_M; i++)
        lam_buf[lam_head][i] = e[i];
    lam_head = (lam_head + 1) % LAMBDA_WINDOW;
    if (lam_count < LAMBDA_WINDOW)
        lam_count++;
}

/* ============================================================
 * EKF MODEL -- CONSTANT VELOCITY, ACCEL-DRIVEN PREDICTION
 * (unchanged from main_ekf_new.c)
 * ============================================================ */

static void run_model(ekf_t  *ep,
                      double  fx[EKF_N],
                      double  F[EKF_N*EKF_N],
                      double  hx[EKF_M],
                      double  H[EKF_M*EKF_N],
                      double  ax, double ay, double az)
{
    int j;
    double accel[3] = {ax, ay, az};

    for (j = 0; j < 3; j++) {
        double v = ep->x[2*j+1];
        double a = accel[j];
        fx[2*j]   = ep->x[2*j] + T_STEP * v + 0.5 * T_STEP * T_STEP * a;
        fx[2*j+1] = v + T_STEP * a;
    }

    memset(F, 0, EKF_N * EKF_N * sizeof(double));
    for (j = 0; j < 6; j++)
        F[j*6+j] = 1.0;
    for (j = 0; j < 3; j++)
        F[2*j*6 + 2*j+1] = T_STEP;

    hx[0] = fx[0];
    hx[1] = fx[2];
    hx[2] = fx[4];

    memset(H, 0, EKF_M * EKF_N * sizeof(double));
    H[0*6+0] = 1.0;
    H[1*6+2] = 1.0;
    H[2*6+4] = 1.0;
}

/* ============================================================
 * NIS -- inn' * S^(-1) * inn, S = H*P*H' + R  (unchanged)
 * ============================================================ */

static double compute_nis(const double inn[EKF_M],
                          const double H[EKF_M*EKF_N],
                          const double P[EKF_N*EKF_N],
                          const double R[EKF_M*EKF_M])
{
    int i, j, k;

    double HP[EKF_M*EKF_N];
    for (i = 0; i < EKF_M; i++)
        for (j = 0; j < EKF_N; j++) {
            double s = 0.0;
            for (k = 0; k < EKF_N; k++)
                s += H[i*EKF_N+k] * P[k*EKF_N+j];
            HP[i*EKF_N+j] = s;
        }

    double S[EKF_M*EKF_M];
    for (i = 0; i < EKF_M; i++)
        for (j = 0; j < EKF_M; j++) {
            double s = 0.0;
            for (k = 0; k < EKF_N; k++)
                s += HP[i*EKF_N+k] * H[j*EKF_N+k];
            S[i*EKF_M+j] = s + R[i*EKF_M+j];
        }

    double a=S[0], b=S[1], c=S[2];
    double d=S[3], e=S[4], f=S[5];
    double g=S[6], h=S[7], ii=S[8];

    double A =  (e*ii - f*h);
    double B = -(d*ii - f*g);
    double C =  (d*h  - e*g);
    double det = a*A + b*B + c*C;

    if (fabs(det) < 1e-12) return 1e6;

    double invdet = 1.0 / det;
    double Sinv[9];
    Sinv[0] =  A * invdet;
    Sinv[1] = -(b*ii - c*h) * invdet;
    Sinv[2] =  (b*f  - c*e) * invdet;
    Sinv[3] =  B * invdet;
    Sinv[4] =  (a*ii - c*g) * invdet;
    Sinv[5] = -(a*f  - c*d) * invdet;
    Sinv[6] =  C * invdet;
    Sinv[7] = -(a*h  - b*g) * invdet;
    Sinv[8] =  (a*e  - b*d) * invdet;

    double Sinv_inn[EKF_M];
    for (i = 0; i < EKF_M; i++) {
        double s = 0.0;
        for (j = 0; j < EKF_M; j++)
            s += Sinv[i*EKF_M+j] * inn[j];
        Sinv_inn[i] = s;
    }
    double nis = 0.0;
    for (i = 0; i < EKF_M; i++)
        nis += inn[i] * Sinv_inn[i];

    return nis;
}

/* ============================================================
 * STRONG TRACKING FILTER -- lambda_k (DIAGNOSTIC ONLY)
 *
 * Computed every step, unconditionally, purely to exercise the
 * extra division-heavy computation on hardware. NEVER used to
 * scale ekf.P or ekf.x -- see file header for why (it actively
 * suppressed detection when Python-tested that way).
 *
 *   V_k = mean(e_i * e_i^T) over the last LAMBDA_WINDOW raw
 *         innovations (e = z_meas - hx, from THIS step's
 *         prediction, pushed into the window before this call)
 *   M_k = H * (F * P_prior * F^T) * H^T   -- P_prior is the
 *         PRE-predict P (this step's starting P, before
 *         ekf_predict() overwrites it with F P F^T + Q)
 *   N_k = V_k - R
 *   lambda_k = max(1, trace(N_k) / trace(M_k))
 * ============================================================ */

static double compute_lambda_k(const double F[EKF_N*EKF_N],
                               const double P_prior[EKF_N*EKF_N],
                               const double H[EKF_M*EKF_N],
                               const double R[EKF_M*EKF_M])
{
    int i, j, k;

    /* FP = F * P_prior */
    double FP[EKF_N*EKF_N];
    for (i = 0; i < EKF_N; i++)
        for (j = 0; j < EKF_N; j++) {
            double s = 0.0;
            for (k = 0; k < EKF_N; k++)
                s += F[i*EKF_N+k] * P_prior[k*EKF_N+j];
            FP[i*EKF_N+j] = s;
        }

    /* FPFt = FP * F^T */
    double FPFt[EKF_N*EKF_N];
    for (i = 0; i < EKF_N; i++)
        for (j = 0; j < EKF_N; j++) {
            double s = 0.0;
            for (k = 0; k < EKF_N; k++)
                s += FP[i*EKF_N+k] * F[j*EKF_N+k];
            FPFt[i*EKF_N+j] = s;
        }

    /* M_k = H * FPFt * H^T */
    double HFPFt[EKF_M*EKF_N];
    for (i = 0; i < EKF_M; i++)
        for (j = 0; j < EKF_N; j++) {
            double s = 0.0;
            for (k = 0; k < EKF_N; k++)
                s += H[i*EKF_N+k] * FPFt[k*EKF_N+j];
            HFPFt[i*EKF_N+j] = s;
        }

    double M_k[EKF_M*EKF_M];
    for (i = 0; i < EKF_M; i++)
        for (j = 0; j < EKF_M; j++) {
            double s = 0.0;
            for (k = 0; k < EKF_N; k++)
                s += HFPFt[i*EKF_N+k] * H[j*EKF_N+k];
            M_k[i*EKF_M+j] = s;
        }

    /* V_k = mean(e_i * e_i^T) over the lambda window */
    double V_k[EKF_M*EKF_M] = {0};
    if (lam_count > 0) {
        for (int n = 0; n < lam_count; n++) {
            for (i = 0; i < EKF_M; i++)
                for (j = 0; j < EKF_M; j++)
                    V_k[i*EKF_M+j] += lam_buf[n][i] * lam_buf[n][j];
        }
        for (i = 0; i < EKF_M*EKF_M; i++)
            V_k[i] /= (double)lam_count;
    }

    /* N_k = V_k - R; trace ratio */
    double trace_N = 0.0, trace_M = 0.0;
    for (i = 0; i < EKF_M; i++) {
        trace_N += V_k[i*EKF_M+i] - R[i*EKF_M+i];
        trace_M += M_k[i*EKF_M+i];
    }

    double lambda_k;
    if (fabs(trace_M) < 1e-9) {
        lambda_k = 1.0;
    } else {
        double ratio = trace_N / trace_M;
        lambda_k = (ratio > 1.0) ? ratio : 1.0;   /* floor at 1 -- design
                                                       choice: this is a
                                                       branch, so its
                                                       taken/not-taken
                                                       pattern is itself
                                                       attack-correlated.
                                                       See file header. */
    }

    return lambda_k;
}

/* ============================================================
 * DIAGNOSTICS LOG
 * Same TRAJ_LEN+64 padding as main_ekf_new.c -- see that file's
 * comment for why. lambda_k added as a new field.
 * ============================================================ */

typedef struct {
    int    t;
    int    attack_active;
    double innovation_norm;
    double nis;
    double anis;
    int    nis_alarm;
    int    anis_alarm;
    double p_trace;
    int    invert_ok;
    double filt_x;
    double filt_y;
    double filt_z;
    double true_x;
    double true_y;
    double true_z;
    double spoof_error;
    double lambda_k;
} diag_t;

static diag_t diag_log[TRAJ_LEN + 64];

/* ============================================================
 * ONE EKF STEP
 * ============================================================ */

static void ekf_step(int t,
                     double mx, double my, double mz,
                     double tx, double ty, double tz,
                     double ax, double ay, double az)
{
    int    i;
    double inn[EKF_M]       = {0};
    double fx[EKF_N]        = {0};
    double F[EKF_N*EKF_N]  = {0};
    double hx[EKF_M]       = {0};
    double H[EKF_M*EKF_N] = {0};

    double z_meas[EKF_M] = {mx, my, mz};

    run_model(&ekf, fx, F, hx, H, ax, ay, az);

    /* Raw innovation for THIS step, using the pre-predict hx --
       needed both for the real update below and for lambda_k. */
    double e_raw[EKF_M];
    for (i = 0; i < EKF_M; i++)
        e_raw[i] = z_meas[i] - hx[i];

    /* lambda_k: computed from the PRE-predict P (ekf.P still holds
       last step's post-update value here), then pushed into the
       window AFTER use so this step's own innovation feeds next
       step's V_k, not this one's -- lambda_k for step t is based
       only on evidence up to t-1, which is the causally-correct
       choice for an online filter (you can't use the residual
       you're trying to explain to decide how much to trust it).
       NOTE: Trajectory/simulate_ekf_stf.py's quick Python prototype
       pushed the current step's innovation into the window BEFORE
       computing lambda_k for that same step -- a shortcut that's
       fine for an offline what-if check but not how this version
       does it. Expect lambda_k values here to differ slightly
       (one-step lag) from the Python prototype's numbers; this
       doesn't affect the "does P/x stay untouched" property either
       version relies on. */
    double lambda_k = compute_lambda_k(F, ekf.P, H, R_mat);
    lambda_window_push(e_raw);

    /* Real predict/update -- IDENTICAL to main_ekf_new.c. Q_mat,
       R_mat, x, P are never touched by lambda_k. */
    ekf_predict(&ekf, fx, F, Q_mat);
    int ok = ekf_update(&ekf, z_meas, hx, H, R_mat);

    for (i = 0; i < EKF_M; i++)
        inn[i] = e_raw[i];
    double inn_norm = 0.0;
    for (i = 0; i < EKF_M; i++)
        inn_norm += inn[i] * inn[i];
    inn_norm = sqrt(inn_norm);

    double nis  = compute_nis(inn, H, ekf.P, R_mat);
    window_push(nis);
    double anis = window_mean();

    int nis_alarm  = (nis  > NIS_THRESHOLD)  ? 1 : 0;
    int anis_alarm = (anis > ANIS_THRESHOLD) ? 1 : 0;

    double p_trace = 0.0;
    for (i = 0; i < EKF_N; i++)
        p_trace += ekf.P[i*EKF_N+i];

    double spoof_err = sqrt(
        (mx-tx)*(mx-tx) + (my-ty)*(my-ty) + (mz-tz)*(mz-tz));

    diag_log[t].t               = t;
    diag_log[t].attack_active   = (t >= ATTACK_START) ? 1 : 0;
    diag_log[t].innovation_norm = inn_norm;
    diag_log[t].nis             = nis;
    diag_log[t].anis            = anis;
    diag_log[t].nis_alarm       = nis_alarm;
    diag_log[t].anis_alarm      = anis_alarm;
    diag_log[t].p_trace         = p_trace;
    diag_log[t].invert_ok       = ok;
    diag_log[t].filt_x          = ekf.x[0];
    diag_log[t].filt_y          = ekf.x[2];
    diag_log[t].filt_z          = ekf.x[4];
    diag_log[t].true_x          = tx;
    diag_log[t].true_y          = ty;
    diag_log[t].true_z          = tz;
    diag_log[t].spoof_error     = spoof_err;
    diag_log[t].lambda_k        = lambda_k;
}

/* ============================================================
 * MAIN
 * ============================================================ */

int main(void)
{
    int t, i;

    asm volatile ("csrw mcycle,   zero");
    asm volatile ("csrw minstret, zero");

    configure_hpm_events_macro_PCA_selected_set();

    double pdiag[EKF_N];
    for (i = 0; i < EKF_N; i++) pdiag[i] = P0_INIT;
    ekf_initialize(&ekf, pdiag);
    ekf.x[0] = traj_true[0][0];
    ekf.x[2] = traj_true[0][1];
    ekf.x[4] = traj_true[0][2];
    ekf.x[1] = ekf.x[3] = ekf.x[5] = 0.0;

    uart_puts("Start\n");

    for (t = 1; t < TRAJ_LEN; t++) {
        ekf_step(t,
                traj_measured[t][0], traj_measured[t][1], traj_measured[t][2],
                traj_true[t][0],     traj_true[t][1],     traj_true[t][2],
                imu_accel_meas[t][0], imu_accel_meas[t][1], imu_accel_meas[t][2]);
    }

    uart_puts("Stop\n");

    {
        volatile uint32_t pause;
        for (pause = 0; pause < 80000000UL; pause++) {
            __asm__ volatile ("nop");
        }
    }

    uart_puts("DIAG_START\n");

    char buf[512];
    for (t = 1; t < TRAJ_LEN; t++) {
        sprintf(buf,
            "DIAG:%d,%d,"
            "%.6f,%.6f,%.6f,%d,%d,"
            "%.6f,%d,"
            "%.4f,%.4f,%.4f,"
            "%.4f,%.4f,%.4f,"
            "%.4f,"
            "%.6f\n",
            diag_log[t].t,
            diag_log[t].attack_active,
            diag_log[t].innovation_norm,
            diag_log[t].nis,
            diag_log[t].anis,
            diag_log[t].nis_alarm,
            diag_log[t].anis_alarm,
            diag_log[t].p_trace,
            diag_log[t].invert_ok,
            diag_log[t].filt_x,
            diag_log[t].filt_y,
            diag_log[t].filt_z,
            diag_log[t].true_x,
            diag_log[t].true_y,
            diag_log[t].true_z,
            diag_log[t].spoof_error,
            diag_log[t].lambda_k);
        uart_puts(buf);

        volatile uint32_t d;
        for (d = 0; d < 300000UL; d++) {
           __asm__ volatile ("nop");
        }
    }

    uart_puts("DIAG_END\n");

    exit(0);
}
