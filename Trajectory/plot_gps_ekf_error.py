"""Two paper figures, per the whiteboard sketch, for one seed:

  1) e_GPS(t) -- raw GPS measurement deviation from the normal baseline.
     z_t = [x_t; y_t; z_t] is the raw measured position (traj_measured
     in trajectory_<mode>_seed<N>_new.h); reduced to a scalar distance
     from origin |z_t| = sqrt(x_t^2 + y_t^2 + z_t^2) (ECEF, so "origin"
     is Earth's center -- the ~6.37e6 m common offset cancels out in the
     differences below, leaving only the meaningful few-to-few-hundred-
     meter deviations).

     meanz(t)  = mean of |z_t| across the REFERENCE_SEEDS normal seeds
     black(t)  = |z_t|_normal,SEED(t)  - meanz(t)
     orange(t) = |z_t|_jump,SEED(t)    - meanz(t)
     green(t)  = |z_t|_drift,SEED(t)   - meanz(t)

     REFERENCE_SEEDS is currently 9 (seed1-9_newMapping are the only
     ones with data on disk so far, out of an eventual 20) -- rerun once
     more seeds are added.

  2) e_EKF(t) = || P_hat_t - P_t^true || [m]  -- EKF position estimate's
     distance from ground truth, computed from filt_x/y/z and
     true_x/y/z (not a stored column), from ekf_diag_<mode>.csv.
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

SEED = 1
REFERENCE_SEEDS = list(range(1, 10))  # seed1-9_newMapping -- only ones with data so far
SEED_DIR = os.path.join(PROJECT_ROOT, "adaptive_analysis_board_firmware", "plots", f"seed{SEED}_newMapping")

MODE_COLORS = {
    "normal": "#111111",
    "jump": "#E07B1A",
    "drift": "#2E8B3D",
}

ARRAY_RE = re.compile(r"\{\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\}")


def load_traj_measured(mode, seed):
    """Parse traj_measured[TRAJ_LEN][3] out of trajectory_<mode>_seed<N>_new.h."""
    path = os.path.join(SCRIPT_DIR, f"trajectory_{mode}_seed{seed}_new.h")
    with open(path) as f:
        text = f.read()
    block = text.split("traj_measured")[1].split("{", 1)[1].split("};")[0]
    rows = ARRAY_RE.findall(block)
    return np.array(rows, dtype=float)


def dist_from_origin(xyz):
    return np.linalg.norm(xyz, axis=1)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # meanz(t): mean distance-from-origin across the available normal seeds
    normal_dists = np.stack([dist_from_origin(load_traj_measured("normal", s)) for s in REFERENCE_SEEDS])
    meanz = normal_dists.mean(axis=0)

    gps_curves = {}
    for mode in MODE_COLORS:
        xyz = load_traj_measured(mode, SEED)
        gps_curves[mode] = dist_from_origin(xyz) - meanz
    t_gps = np.arange(1, len(meanz) + 1)

    dfs = {mode: pd.read_csv(os.path.join(SEED_DIR, f"ekf_diag_{mode}.csv")) for mode in MODE_COLORS}
    for df in dfs.values():
        filt = df[["filt_x", "filt_y", "filt_z"]].to_numpy()
        true = df[["true_x", "true_y", "true_z"]].to_numpy()
        df["e_ekf"] = np.linalg.norm(filt - true, axis=1)

    plt.rcParams.update({"font.size": 12, "font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax = axes[0]
    for mode, color in MODE_COLORS.items():
        ax.plot(t_gps, gps_curves[mode], color=color, linewidth=1.8, label=mode)
    ax.set_xlabel("EKF iteration")
    ax.set_ylabel(r"$e_{GPS}(t) = |z_t| - \overline{|z_t|}_{normal}$ [m]")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=10, frameon=False)

    ax = axes[1]
    for mode, color in MODE_COLORS.items():
        df = dfs[mode]
        ax.plot(df["t"], df["e_ekf"], color=color, linewidth=1.8, label=mode)
    ax.set_xlabel("EKF iteration")
    ax.set_ylabel(r"$e_{EKF}(t) = \Vert \hat{P}_t - P_t^{true} \Vert$ [m]")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=10, frameon=False)

    fig.suptitle(f"Seed {SEED} — GPS measurement error vs. EKF estimation error "
                 f"(normal mean over {len(REFERENCE_SEEDS)} seeds)", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"seed{SEED}_gps_ekf_error.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
