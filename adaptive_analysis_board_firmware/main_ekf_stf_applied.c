/*
 * main_ekf_stf_applied.c
 *
 * APPLIED variant of main_ekf_stf.c: lambda_k can actually scale the
 * predicted covariance -- but only through two EXCLUSIVE, ANIS-gated
 * branches, not unconditionally.
 *
 * ============================================================
 * HOW THIS DIFFERS FROM main_ekf_stf.c -- READ THIS FIRST
 * ============================================================
 * main_ekf_stf.c deliberately never lets lambda_k touch ekf.P/ekf.x,
 * so the already-working GPS+IMU NIS/ANIS detector (main_ekf_new.c)
 * stays completely intact while probing for a hardware signature from
 * the extra computation alone -- but it always computes lambda_k,
 * every step, unconditionally, which caps the achievable signal at a
 * timing-only effect (same instruction count every step, attack or
 * not).
 *
 * This file goes further: it makes the *computation itself* present
 * or absent depending on the real, validated detector, using two
 * separate exclusive branches:
 *
 *   BRANCH 1 (in ekf_step): compute_lambda_k() -- the expensive part,
 *   ~680 multiply-adds + 10 divisions -- only runs when last step's
 *   ANIS already exceeded ANIS_THRESHOLD. ANIS was chosen as the
 *   gate specifically because it's the detector we've already shown
 *   is clean (seed1: drift fires 119/299, jump 151/299, vs normal
 *   15/299), unlike raw per-step lambda_k magnitude, which is NOT
 *   discriminative on its own (drift and normal came out
 *   statistically identical: mean 1.20, max 4.2598, std ~0.45, both
 *   modes). When ANIS didn't flag anything, this step just calls the
 *   plain library ekf_predict() -- identical to main_ekf_stf.c.
 *
 *   BRANCH 2 (inside branch 1): even when compute_lambda_k() DID run,
 *   P is only actually inflated (P = lambda_k*FPFt + Q) if the result
 *   is genuinely above its floor of 1.0. If it landed exactly at 1.0
 *   (ANIS was elevated, but the STF check itself found nothing), the
 *   plain unscaled predict is used instead (P = FPFt + Q, reusing the
 *   FPFt already computed rather than re-deriving it).
 *
 * NOT YET VALIDATED: the NIS/ANIS-degradation numbers that motivated
 * main_ekf_stf.c's decoupled design (drift nis_alarm 51/299 -> 2/299
 * under UNCONDITIONAL lambda_k application) describe the old design,
 * not this one. Since P is now only inflated when ANIS/lambda_k both
 * agree something is happening, the actual detection impact of THIS
 * gated version hasn't been re-checked in
 * Trajectory/simulate_ekf_stf.py yet -- do that before drawing
 * conclusions about whether this version still suppresses detection
 * the way the unconditional one did.
 *
 * ============================================================
 * WHAT ACTUALLY CHANGED FROM main_ekf_stf.c
 * ============================================================
 *   - New static double anis_prev: last step's ANIS, carried forward
 *     to gate this step's branch decision (ANIS for step t isn't
 *     known until AFTER this step's update, so the gate necessarily
 *     uses t-1's value -- same one-step-lag principle as lambda_k's
 *     own windowing).
 *   - compute_lambda_k() gained an output parameter, FPFt_out: it
 *     already builds F*P_prior*F^T internally to compute M_k, so
 *     this just hands that same matrix back instead of discarding
 *     it, avoiding a redundant recompute when branch 2 needs it.
 *   - ekf_step()'s predict logic is now the two-branch structure
 *     described above, instead of always calling the library
 *     ekf_predict() (main_ekf_stf.c) or always applying lambda_k
 *     unconditionally (this file's earlier revision).
 *   - lambda_window_push(e_raw) stays unconditional regardless of
 *     which branch runs -- it's cheap bookkeeping (a 3-value copy),
 *     not the expensive part, and the window needs to reflect the
 *     TRUE most-recent-LAMBDA_WINDOW history so a future
 *     compute_lambda_k() call isn't built from gappy data.
 *   - Everything else (run_model, compute_nis, diag_t, DIAG format,
 *     main() loop structure) is UNCHANGED from main_ekf_stf.c.
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

/* ANIS from the previous step -- gates whether this step's predict
   bothers checking lambda_k at all. Starts at 0.0 (below
   ANIS_THRESHOLD), so step t=1 correctly takes the cheap plain-predict
   path with no prior history to judge from. */
static double anis_prev = 0.0;

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

    /* The window always tracks the true most-recent-LAMBDA_WINDOW
       history, regardless of whether lambda_k gets computed this
       step -- cheap bookkeeping (a 3-value copy), not the expensive
       part, so it stays unconditional. If it were only pushed when
       triggered, a future lambda_k computation would be built from a
       gappy, non-contiguous history instead of the real last 10
       steps. */
    lambda_window_push(e_raw);

    /* EXCLUSIVE BRANCH 1: only bother computing lambda_k at all when
       last step's ANIS already flagged something as unusual. ANIS is
       the validated, clean detector (seed1: drift 119/299, jump
       151/299, vs normal 15/299 -- see the anis_alarm fire-point
       analysis). Raw per-step lambda_k magnitude is NOT clean on its
       own (drift and normal came out statistically identical: same
       mean 1.20, same max 4.2598, same std). So ANIS decides WHETHER
       to look; lambda_k (when computed) decides HOW MUCH.

       When ANIS didn't flag anything, skip compute_lambda_k() (the
       ~680 multiply-adds + 10 divisions) entirely and just call the
       plain library predict -- identical to main_ekf_stf.c/
       main_ekf_new.c for that step. */
    double lambda_k = 1.0;

    if (anis_prev > ANIS_THRESHOLD) {
        double FPFt[EKF_N*EKF_N];
        lambda_k = compute_lambda_k(F, ekf.P, H, R_mat, FPFt);

        /* EXCLUSIVE BRANCH 2: only actually inflate P when lambda_k
           came back genuinely above its floor of 1.0 -- ANIS being
           elevated doesn't guarantee the STF check itself finds
           anything (they're different statistics). If lambda_k
           landed exactly at 1.0, use the FPFt already computed
           rather than re-deriving it via a second call. */
        memcpy(ekf.x, fx, EKF_N * sizeof(double));
        if (lambda_k > 1.0) {
            for (i = 0; i < EKF_N * EKF_N; i++)
                ekf.P[i] = lambda_k * FPFt[i] + Q_mat[i];
        } else {
            for (i = 0; i < EKF_N * EKF_N; i++)
                ekf.P[i] = FPFt[i] + Q_mat[i];
        }
    } else {
        ekf_predict(&ekf, fx, F, Q_mat);
    }

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
    anis_prev = anis;   // for the NEXT call's branch decision

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
