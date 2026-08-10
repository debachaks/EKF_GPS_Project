/*
 * main_ekf_stf_applied.c
 *
 * APPLIED variant of main_ekf_stf.c: lambda_k now actually scales the
 * predicted covariance, instead of being computed-and-discarded.
 *
 * ============================================================
 * HOW THIS DIFFERS FROM main_ekf_stf.c -- READ THIS FIRST
 * ============================================================
 * main_ekf_stf.c deliberately never let lambda_k touch ekf.P/ekf.x,
 * specifically so the already-working GPS+IMU NIS/ANIS detector
 * (main_ekf_new.c) stayed completely intact while probing for a
 * hardware signature from the extra computation alone. That gave a
 * real but narrow signal: hpm5 (cond-branch mispredictions), tied to
 * a fixed LAMBDA_WINDOW=10-step branch pattern right at ATTACK_START.
 *
 * This file makes the OPPOSITE trade: apply lambda_k for real, which
 * we already know (Trajectory/simulate_ekf_stf.py, 10 seeds x 4
 * modes) DEGRADES the NIS/ANIS detector --
 *
 *   drift nis_alarm:  baseline 51/299  ->  with lambda_k  2/299
 *   jump  nis_alarm: baseline 147/299  ->  with lambda_k  6.6/299
 *
 * -- in exchange for a DIFFERENT, longer-lived hardware-observable
 * event: once lambda_k actually inflates P, P doesn't snap back on a
 * fixed window boundary the way the floor-branch pattern does. It
 * decays according to the filter's own recursive dynamics. Python
 * simulation (jump, seed 1) showed p_trace jump from a pre-attack
 * steady state of ~21.7 to a peak of ~7131 (>300x) at t=151, still
 * ~46% above baseline 34 steps later at t=184, with a second bump
 * around t=199-208 as the filter keeps re-adjusting. That's a much
 * longer, continuously-varying window than the rigid ~10-step branch
 * pattern -- and P feeds every subsequent matrix operation (F P F^T,
 * H P H^T, the gain inversion), so hpm3/hpm4 (FP interlock cycles /
 * FP div-sqrt retired -- your "M1: FP latency" pair) are the
 * predicted targets here, over an extended and VARIABLE-length
 * post-attack window -- not a fixed constant like LAMBDA_WINDOW.
 *
 * THIS IS A DELIBERATE FORK, NOT AN UPGRADE: main_ekf_stf.c's result
 * (hpm5, NIS/ANIS intact) is still valid and kept as its own file.
 * Use this file only if hardware-counter detection is being pursued
 * as the PRIMARY strategy, since NIS/ANIS are expected to degrade
 * here exactly as the Python ablation predicted.
 *
 * ============================================================
 * WHAT ACTUALLY CHANGED FROM main_ekf_stf.c
 * ============================================================
 *   - compute_lambda_k() gained an output parameter, FPFt_out: it
 *     already builds F*P_prior*F^T internally to compute M_k, so
 *     this just hands that same matrix back instead of discarding
 *     it, avoiding a redundant recompute.
 *   - ekf_step() no longer calls the library's ekf_predict(). It
 *     replicates ekf_predict()'s exact logic manually, but with
 *     lambda_k scaling the propagated-uncertainty term:
 *         ekf.x = fx
 *         ekf.P = lambda_k * FPFt + Q_mat      (was: 1.0 * FPFt + Q_mat)
 *     matching Trajectory/simulate_ekf_stf.py's run_ekf_stf() exactly
 *     -- only the FPFt term is scaled, Q is added unscaled after.
 *   - ekf_update() is called exactly as before -- it only reads
 *     ekf->P/ekf->x, so it doesn't need to know how they got there.
 *   - Everything else (run_model, compute_nis, diag_t, DIAG format,
 *     NIS/ANIS window, main() loop structure) is UNCHANGED from
 *     main_ekf_stf.c. NIS/ANIS/alarms are still logged -- kept as a
 *     reference/comparison against the ablation numbers above, not
 *     because they're expected to still work well.
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
 * SLIDING WINDOW FOR ANIS (unchanged from main_ekf_stf.c)
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
 * (unchanged from main_ekf_stf.c)
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
 * (unchanged from main_ekf_stf.c / main_ekf_new.c)
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
 * STRONG TRACKING FILTER -- lambda_k (APPLIED)
 *
 * Same formula as main_ekf_stf.c's diagnostic-only version:
 *   V_k = mean(e_i * e_i^T) over the last LAMBDA_WINDOW raw
 *         innovations (e = z_meas - hx, from THIS step's
 *         prediction, pushed into the window before this call)
 *   M_k = H * (F * P_prior * F^T) * H^T   -- P_prior is the
 *         PRE-predict P (this step's starting P, before this
 *         step's predict overwrites it)
 *   N_k = V_k - R
 *   lambda_k = max(1, trace(N_k) / trace(M_k))
 *
 * DIFFERENCE FROM main_ekf_stf.c: also returns FPFt (via FPFt_out)
 * -- the F*P_prior*F^T it already computes for M_k -- so ekf_step()
 * can reuse it directly for the lambda-scaled predict instead of
 * letting the library's ekf_predict() recompute an unscaled version.
 * ============================================================ */

static double compute_lambda_k(const double F[EKF_N*EKF_N],
                               const double P_prior[EKF_N*EKF_N],
                               const double H[EKF_M*EKF_N],
                               const double R[EKF_M*EKF_M],
                               double FPFt_out[EKF_N*EKF_N])
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

    /* FPFt = FP * F^T -- handed back via FPFt_out for the predict step */
    double FPFt[EKF_N*EKF_N];
    for (i = 0; i < EKF_N; i++)
        for (j = 0; j < EKF_N; j++) {
            double s = 0.0;
            for (k = 0; k < EKF_N; k++)
                s += FP[i*EKF_N+k] * F[j*EKF_N+k];
            FPFt[i*EKF_N+j] = s;
        }
    memcpy(FPFt_out, FPFt, EKF_N * EKF_N * sizeof(double));

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
        lambda_k = (ratio > 1.0) ? ratio : 1.0;   /* floor at 1 -- STF only
                                                       ever INFLATES
                                                       uncertainty, never
                                                       shrinks it below the
                                                       nominal filter's. */
    }

    return lambda_k;
}

/* ============================================================
 * DIAGNOSTICS LOG
 * Same TRAJ_LEN+64 padding as main_ekf_stf.c / main_ekf_new.c.
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
       last step's post-update value here), same one-step-lag
       causal ordering as main_ekf_stf.c -- pushed into the window
       AFTER use, so lambda_k for step t only sees evidence up to
       t-1. FPFt is handed back so the predict below can reuse it. */
    double FPFt[EKF_N*EKF_N];
    double lambda_k = compute_lambda_k(F, ekf.P, H, R_mat, FPFt);
    lambda_window_push(e_raw);

    /* Predict, WITH lambda_k APPLIED -- this is the one substantive
       difference from main_ekf_stf.c. Replicates ekf_predict()'s
       exact logic (x = fx, P = F P F^T + Q) but scales only the
       propagated-uncertainty term by lambda_k, matching
       Trajectory/simulate_ekf_stf.py's run_ekf_stf() exactly. Q is
       added unscaled, same as an unmodified predict. */
    memcpy(ekf.x, fx, EKF_N * sizeof(double));
    for (i = 0; i < EKF_N * EKF_N; i++)
        ekf.P[i] = lambda_k * FPFt[i] + Q_mat[i];

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
