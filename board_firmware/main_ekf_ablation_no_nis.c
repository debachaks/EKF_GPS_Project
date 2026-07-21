/*
 * main_ekf_ablation_no_nis.c
 *
 * ABLATION VARIANT of main_ekf.c, built to answer one question:
 * is hpmcounter5 ("mul/div interlock") separating normal from
 * drift/jump/replay because of the core EKF math (ekf_predict /
 * ekf_update in tinyekf.h), or because of the NIS/ANIS diagnostic
 * block (compute_nis / window_push / threshold checks) that also
 * runs inside the profiled Start..Stop loop?
 *
 * Everything below is IDENTICAL to main_ekf.c except the block
 * marked ABLATION START / ABLATION END inside ekf_step(). That
 * block is what computed NIS, pushed it into the ANIS window, and
 * derived nis_alarm/anis_alarm - it's now skipped, so those
 * diag_log fields are written as 0 instead. run_model(),
 * ekf_predict(), and ekf_update() (the actual EKF - the only
 * things doing matrix math now) are untouched.
 *
 * HOW TO USE
 * Build and run this in place of main_ekf.c for a couple of the
 * seed_new-style trajectory.h headers (e.g. one drift, one normal)
 * and diff the resulting hpmcounter5 values against the original
 * main_ekf.c run on the SAME trajectory.h:
 *   - hpmcounter5 still separates normal vs drift the same way
 *     -> the EKF core (Cholesky inversion in ekf_update) is doing it
 *   - hpmcounter5 separation shrinks/disappears
 *     -> the NIS/ANIS block was a real contributor
 *
 * diag_log's nis/anis/nis_alarm/anis_alarm columns are meaningless
 * in this build (always 0) - only use this build to compare
 * hpmcounter5 (and other HPCs), not to compare detection latency.
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
#include "ekf_config.h"
#include "tinyekf.h"
#include "trajectory.h"

/* ============================================================
 * EKF STATE
 * ============================================================ */

static ekf_t ekf;

/* ============================================================
 * SLIDING WINDOW FOR ANIS
 * Kept only so the struct/build layout matches main_ekf.c - not
 * called in this ablation build (see ekf_step below).
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
 * EKF MODEL -- CONSTANT VELOCITY (unchanged)
 * ============================================================ */

static void run_model(ekf_t  *ep,
                      double  fx[EKF_N],
                      double  F[EKF_N*EKF_N],
                      double  hx[EKF_M],
                      double  H[EKF_M*EKF_N])
{
    int j;

    for (j = 0; j < 6; j += 2) {
        fx[j]   = ep->x[j] + T_STEP * ep->x[j+1];
        fx[j+1] = ep->x[j+1];
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
 * NIS -- inn' * S^(-1) * inn, S = H*P*H' + R
 * Kept (unmodified) but NOT called from ekf_step() in this
 * ablation build - see ABLATION block below.
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
 * DIAGNOSTICS LOG (same +64 padding workaround as main_ekf.c)
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
} diag_t;

static diag_t diag_log[TRAJ_LEN + 64];

/* ============================================================
 * ONE EKF STEP - ABLATION VARIANT
 * ============================================================ */

static void ekf_step(int t,
                     double mx, double my, double mz,
                     double tx, double ty, double tz)
{
    int    i;
    double inn[EKF_M]       = {0};
    double fx[EKF_N]        = {0};
    double F[EKF_N*EKF_N]  = {0};
    double hx[EKF_M]       = {0};
    double H[EKF_M*EKF_N] = {0};

    double z_meas[EKF_M] = {mx, my, mz};

    run_model(&ekf, fx, F, hx, H);

    ekf_predict(&ekf, fx, F, Q_mat);
    int ok = ekf_update(&ekf, z_meas, hx, H, R_mat);

    double inn_norm = 0.0;
    for (i = 0; i < EKF_M; i++) {
        inn[i]    = z_meas[i] - hx[i];
        inn_norm += inn[i] * inn[i];
    }
    inn_norm = sqrt(inn_norm);

    /* ============================================================
     * ABLATION START - this whole block is what main_ekf.c runs
     * here instead:
     *     double nis  = compute_nis(inn, H, ekf.P, R_mat);
     *     window_push(nis);
     *     double anis = window_mean();
     *     int nis_alarm  = (nis  > NIS_THRESHOLD)  ? 1 : 0;
     *     int anis_alarm = (anis > ANIS_THRESHOLD) ? 1 : 0;
     * Skipped here so the profiled loop contains ONLY run_model +
     * ekf_predict + ekf_update - no second matrix inversion, no
     * extra FP division from compute_nis/window_mean.
     * ============================================================ */
    double nis  = 0.0;
    double anis = 0.0;
    int nis_alarm  = 0;
    int anis_alarm = 0;
    /* ABLATION END */

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
}

/* ============================================================
 * MAIN (unchanged from main_ekf.c)
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
                traj_true[t][0],     traj_true[t][1],     traj_true[t][2]);
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
            "%.4f\n",
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
            diag_log[t].spoof_error);
        uart_puts(buf);

        volatile uint32_t d;
        for (d = 0; d < 300000UL; d++) {
           __asm__ volatile ("nop");
        }
    }

    uart_puts("DIAG_END\n");

    exit(0);
}
