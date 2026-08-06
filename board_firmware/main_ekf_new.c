/*
 * main_ekf_new.c
 *
 * NEW VERSION of main_ekf.c, extended with a simulated IMU
 * (accelerometer) as a second, independent measurement source
 * alongside GPS.
 *
 * GPS Spoofing Detection via Hardware Performance Counters
 * PRECOMPUTED-TRAJECTORY VERSION, GPS+IMU
 *
 * Platform: SiFive 8-stage dual-issue RISC-V soft-core
 *           on Xilinx Versal SoC-FPGA
 *
 * WHAT CHANGED FROM main_ekf.c:
 *   - run_model() now takes the current accelerometer reading
 *     (ax, ay, az) and uses it to drive the predicted position and
 *     velocity, instead of just carrying the previous velocity
 *     estimate forward unchanged:
 *         fx_pos = x + T_STEP*v + 0.5*T_STEP^2*a
 *         fx_vel = v + T_STEP*a
 *     F is UNCHANGED -- the accel reading is a known input, not a
 *     state, so it doesn't appear in the Jacobian.
 *   - ekf_step() and the main() loop thread ax/ay/az through from
 *     the new imu_accel_meas[][3] array (see trajectory.h, produced
 *     by generate_trajectories_new.py).
 *   - EKF_N, EKF_M, H, diag_log, compute_nis(), tinyekf.h are ALL
 *     UNCHANGED. The accelerometer augments the predict step only;
 *     the measurement/update step, and everything downstream of it
 *     (NIS/ANIS/diagnostics), works exactly as before.
 *
 * WHY THIS SHOULD MAKE DRIFT MORE VISIBLE:
 *   The true trajectory is a straight line at constant speed, so
 *   true acceleration is ~0 throughout. imu_accel_meas is generated
 *   from the TRUE trajectory, independent of `mode` -- a GPS attack
 *   cannot touch it. During a drift attack, GPS keeps implying a
 *   growing velocity, while the accelerometer keeps reporting ~0
 *   acceleration -- a persistent, physical contradiction between
 *   predict and update that should show up in NIS, unlike before
 *   (where velocity was purely inferred from GPS itself and could
 *   silently absorb a slow drift).
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
#include "ekf_config_new.h"    /* MUST come before tinyekf.h.
                              Still needed for EKF_N/EKF_M/_float_t,
                              ATTACK_START (for the ATTACK_ON marker
                              and the diagnostic attack_active flag),
                              NIS_THRESHOLD/ANIS_THRESHOLD, Q_mat/R_mat,
                              P0_INIT, WINDOW_SIZE, T_STEP, ACCEL_SIGMA.
                              JUMP_OFFSET_M/DRIFT_RATE_MS/DRIFT_MAX_M/
                              REPLAY_DELAY/GPS_SIGMA_H/GPS_SIGMA_V are
                              still defined here too -- they're just
                              unused on the board now, since
                              generate_trajectories_new.py reads them
                              directly for the offline computation. */
#include "tinyekf.h"
#include "trajectory.h"     /* now provides traj_true[][3],
                              traj_measured[][3], AND
                              imu_accel_meas[][3] -- generate this
                              from generate_trajectories_new.py, not
                              the old generate_trajectories.py. */

/* ============================================================
 * EKF STATE
 * ============================================================ */

static ekf_t ekf;

/* ============================================================
 * SLIDING WINDOW FOR ANIS
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
 * EKF MODEL -- CONSTANT VELOCITY, ACCEL-DRIVEN PREDICTION
 *
 * ax, ay, az are the current IMU accelerometer reading (m/s^2),
 * used as a control input to the predict step. F is unchanged --
 * the accel term is a known constant here, not a function of
 * state, so it doesn't enter the Jacobian.
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
 * DIAGNOSTICS LOG
 * Sized TRAJ_LEN+64, not TRAJ_LEN -- this padding works around
 * an out-of-bounds write whose exact source was never isolated
 * (symptom: entries near the end of a plain TRAJ_LEN-sized array
 * were getting zeroed). Padding moved the overflow into harmless
 * memory rather than fixing the root cause. Keep this padding
 * unless/until the actual overflow is found and fixed properly.
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
 * ONE EKF STEP
 *
 * Takes the measured position AND the measured IMU acceleration
 * directly -- no mode string, no on-device spoof application.
 * spoof_error is computed simply as the distance between the
 * measured and true positions, which are both already given.
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

    ekf_predict(&ekf, fx, F, Q_mat);
    int ok = ekf_update(&ekf, z_meas, hx, H, R_mat);

    double inn_norm = 0.0;
    for (i = 0; i < EKF_M; i++) {
        inn[i]    = z_meas[i] - hx[i];
        inn_norm += inn[i] * inn[i];
    }
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
}

/* ============================================================
 * MAIN
 * ============================================================ */

int main(void)
{
    int t, i;

    /* Reset counters at startup, so mcycle=0 corresponds to the
       very start of the run. No RNG seed needed anymore -- noise
       is already baked into trajectory.h. */
    asm volatile ("csrw mcycle,   zero");
    asm volatile ("csrw minstret, zero");

    /* Program HPM3-HPM10 with PCA-selected events */
    configure_hpm_events_macro_PCA_selected_set();

    /* Initialize EKF with the first TRUE trajectory point (not
       measured) -- same as before, seeding the filter at the
       actual starting position. */
    double pdiag[EKF_N];
    for (i = 0; i < EKF_N; i++) pdiag[i] = P0_INIT;
    ekf_initialize(&ekf, pdiag);
    ekf.x[0] = traj_true[0][0];
    ekf.x[2] = traj_true[0][1];
    ekf.x[4] = traj_true[0][2];
    ekf.x[1] = ekf.x[3] = ekf.x[5] = 0.0;

    uart_puts("Start\n");

    /* Run TRAJ_LEN-1 EKF steps -- HPM counters accumulate.
       No noise generation, no attack decision: just read the
       precomputed measured/true/accel triple for this step and
       hand it to the filter. Every mode runs this exact same loop. */
    for (t = 1; t < TRAJ_LEN; t++) {

        /*if (t == ATTACK_START) {
            uint32_t mc;
            asm volatile ("csrr %0, mcycle" : "=r"(mc));
            char m[64];
            sprintf(m, "ATTACK_ON mcycle=%u\n", mc);
            uart_puts(m);
        }*/

        ekf_step(t,
                traj_measured[t][0], traj_measured[t][1], traj_measured[t][2],
                traj_true[t][0],     traj_true[t][1],     traj_true[t][2],
                imu_accel_meas[t][0], imu_accel_meas[t][1], imu_accel_meas[t][2]);
    }

    /* Stop triggers reg_record.py to save CSV and close the port */
    uart_puts("Stop\n");

    /*
     * Give the user time to switch terminals and start reading
     * DIAG lines after reg_record.py has finished and released
     * the port. Tuned empirically -- increase if you still miss
     * the start of the dump.
     */
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
