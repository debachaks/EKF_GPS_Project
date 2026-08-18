"""3D EKF-filtered trajectory, normal vs. jump vs. drift, for one seed --
reads directly from the ekf_diag_<mode>.csv diagnostic dumps
(original_pipeline/seed_<N>_data/) rather than regenerating trajectories,
since those files already carry true_x/y/z and filt_x/y/z per timestep.

Positions are converted from ECEF (meters from Earth's center, axes not
aligned with any intuitive direction) to a local East-North-Up frame
centered on the run's starting point, so the plot is readable in a paper
without needing to explain ECEF.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

SEED = 1
SEED_DIR = os.path.join(PROJECT_ROOT, "original_pipeline", f"seed_{SEED}_data")

TRUE_COLOR = "#555555"
MODE_COLORS = {
    "normal": "#1B7A3D",
    "jump": "#C1440E",
    "drift": "#5B3AA6",
}

# WGS84 ellipsoid constants
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def ecef_to_geodetic(x, y, z):
    """Bowring's method -- ECEF (m) to geodetic latitude/longitude (rad)."""
    p = np.hypot(x, y)
    theta = np.arctan2(z * WGS84_A, p * (1 - WGS84_E2) ** 0.5 * WGS84_A)
    lon = np.arctan2(y, x)
    lat = np.arctan2(
        z + WGS84_E2 / (1 - WGS84_E2) * WGS84_A * np.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3,
    )
    return lat, lon


def ecef_to_enu(xyz, ref_xyz, ref_lat, ref_lon):
    """Rotate ECEF-relative-to-reference vectors into local East-North-Up."""
    d = xyz - ref_xyz
    sin_lat, cos_lat = np.sin(ref_lat), np.cos(ref_lat)
    sin_lon, cos_lon = np.sin(ref_lon), np.cos(ref_lon)
    R = np.array([
        [-sin_lon,            cos_lon,           0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat],
    ])
    return d @ R.T


def local_xyz(df, cols, ref_xyz, ref_lat, ref_lon):
    return ecef_to_enu(df[cols].to_numpy(), ref_xyz, ref_lat, ref_lon)


def make_plot(modes, out_name):
    os.makedirs(PLOT_DIR, exist_ok=True)

    dfs = {mode: pd.read_csv(os.path.join(SEED_DIR, f"ekf_diag_{mode}.csv")) for mode in modes}

    ref_xyz = dfs["normal"][["true_x", "true_y", "true_z"]].iloc[0].to_numpy()
    ref_lat, ref_lon = ecef_to_geodetic(*ref_xyz)

    true_enu = local_xyz(dfs["normal"], ["true_x", "true_y", "true_z"], ref_xyz, ref_lat, ref_lon)

    plt.rcParams.update({"font.size": 12, "font.family": "serif"})
    fig = plt.figure(figsize=(9, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.zaxis.labelpad = 18
    fig.subplots_adjust(left=0.02, right=0.90, top=0.95, bottom=0.05)
    ax.xaxis.pane.set_alpha(0.04)
    ax.yaxis.pane.set_alpha(0.04)
    ax.zaxis.pane.set_alpha(0.04)

    ax.plot(true_enu[:, 0], true_enu[:, 1], true_enu[:, 2],
            color=TRUE_COLOR, linewidth=1.3, linestyle="--", label="Ground truth", alpha=0.7)

    normal_enu = local_xyz(dfs["normal"], ["filt_x", "filt_y", "filt_z"], ref_xyz, ref_lat, ref_lon)
    ax.scatter(*normal_enu[0], color="black", s=55, zorder=6, marker="*",
               edgecolor="white", linewidth=0.6, label="Trajectory start")

    onset_marker_used = False
    for mode in modes:
        color = MODE_COLORS[mode]
        df = dfs[mode]
        enu = local_xyz(df, ["filt_x", "filt_y", "filt_z"], ref_xyz, ref_lat, ref_lon)
        label = "EKF estimate (normal)" if mode == "normal" else f"EKF estimate ({mode} attack)"
        ax.plot(enu[:, 0], enu[:, 1], enu[:, 2], color=color, linewidth=2.0, label=label)

        if mode != "normal":
            # last shared pre-attack index -- attack_active is already 1 at
            # its first True row, so that row is post-attack, not the fork
            # point. jump and drift share the exact same filt_x/y/z up to
            # and including this index, so this marker lands in the same
            # place for both.
            first_active = df.index[df["attack_active"] == 1][0]
            onset_idx = first_active - 1
            marker_label = "Attack onset" if not onset_marker_used else None
            onset_marker_used = True
            ax.scatter(*enu[onset_idx], color="black", s=32, zorder=5,
                       edgecolor="white", linewidth=0.6, marker="o", label=marker_label)

    ax.set_xlabel("East [m]", labelpad=10)
    ax.set_ylabel("North [m]", labelpad=10)
    ax.set_zlabel("Up [m]", labelpad=18)
    ax.legend(loc="upper left", fontsize=10.5, frameon=False)
    ax.view_init(elev=25, azim=-35)

    out_path = os.path.join(PLOT_DIR, out_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    make_plot(["normal", "jump"], f"seed{SEED}_jump_vs_normal_3d.png")
    make_plot(["normal", "jump", "drift"], f"seed{SEED}_normal_jump_drift_3d.png")


if __name__ == "__main__":
    main()
