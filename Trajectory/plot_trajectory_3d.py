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


def make_plot(modes, out_name, frame="enu"):
    """frame: "enu" (default, local East-North-Up, readable) or "ecef"
    (raw Earth-Centered Earth-Fixed meters -- axes not aligned with any
    intuitive direction, and all three coordinates sit around 6.378e6 m
    since that's Earth's radius, with only the trajectory's few-hundred-
    meter extent varying within that -- kept for completeness/appendix
    use, not because it's more readable than ENU)."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    dfs = {mode: pd.read_csv(os.path.join(SEED_DIR, f"ekf_diag_{mode}.csv")) for mode in modes}

    if frame == "enu":
        ref_xyz = dfs["normal"][["true_x", "true_y", "true_z"]].iloc[0].to_numpy()
        ref_lat, ref_lon = ecef_to_geodetic(*ref_xyz)

        def project(df, cols):
            return local_xyz(df, cols, ref_xyz, ref_lat, ref_lon)

        axis_labels = ("East [m]", "North [m]", "Up [m]")
    elif frame == "ecef":
        # subtract a clean per-axis reference so tick labels stay short
        # (raw ECEF sits around 6.378e6 m); state the offset in the axis
        # label text itself rather than relying on matplotlib's automatic
        # offset-text placement, which overlaps custom 3D axis labels.
        ecef_ref = np.floor(dfs["normal"][["true_x", "true_y", "true_z"]].iloc[0].to_numpy() / 1000) * 1000

        def project(df, cols):
            return df[cols].to_numpy() - ecef_ref

        axis_labels = (
            f"ECEF X - {ecef_ref[0]:,.0f} [m]",
            f"ECEF Y - {ecef_ref[1]:,.0f} [m]",
            f"ECEF Z - {ecef_ref[2]:,.0f} [m]",
        )
    else:
        raise ValueError(frame)

    true_pts = project(dfs["normal"], ["true_x", "true_y", "true_z"])

    plt.rcParams.update({"font.size": 12, "font.family": "serif"})
    fig = plt.figure(figsize=(9, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.zaxis.labelpad = 18
    fig.subplots_adjust(left=0.02, right=0.90, top=0.95, bottom=0.05)
    ax.xaxis.pane.set_alpha(0.04)
    ax.yaxis.pane.set_alpha(0.04)
    ax.zaxis.pane.set_alpha(0.04)

    ax.plot(true_pts[:, 0], true_pts[:, 1], true_pts[:, 2],
            color=TRUE_COLOR, linewidth=1.3, linestyle="--", label="Ground truth", alpha=0.7)

    normal_pts = project(dfs["normal"], ["filt_x", "filt_y", "filt_z"])
    ax.scatter(*normal_pts[0], color="black", s=55, zorder=6, marker="*",
               edgecolor="white", linewidth=0.6, label="Trajectory start")

    onset_marker_used = False
    for mode in modes:
        color = MODE_COLORS[mode]
        df = dfs[mode]
        pts = project(df, ["filt_x", "filt_y", "filt_z"])
        label = "EKF estimate (normal)" if mode == "normal" else f"EKF estimate ({mode} attack)"
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=2.0, label=label)

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
            ax.scatter(*pts[onset_idx], color="black", s=32, zorder=5,
                       edgecolor="white", linewidth=0.6, marker="o", label=marker_label)

    ax.set_xlabel(axis_labels[0], labelpad=10)
    ax.set_ylabel(axis_labels[1], labelpad=10)
    ax.set_zlabel(axis_labels[2], labelpad=18)
    ax.legend(loc="upper left", fontsize=10.5, frameon=False)
    ax.view_init(elev=25, azim=-35)

    # tighten the box to the actual data extent (matplotlib's 3D autoscale
    # margin is generous, ~10-15%) and match the box proportions to the
    # data's real aspect ratio instead of the default forced cube -- a thin
    # trajectory otherwise reads as "lost" in an oversized cubic box. This
    # matters even more in ECEF, where the trajectory's few-hundred-meter
    # extent would otherwise be invisible against the ~6.378e6 m baseline.
    all_pts = np.vstack([true_pts] + [project(dfs[m], ["filt_x", "filt_y", "filt_z"]) for m in modes])
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    pad = 0.04 * (maxs - mins)
    ax.set_xlim3d(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim3d(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim3d(mins[2] - pad[2], maxs[2] + pad[2])
    ax.set_box_aspect(maxs - mins + 2 * pad)

    out_path = os.path.join(PLOT_DIR, out_name)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    make_plot(["normal", "jump"], f"seed{SEED}_jump_vs_normal_3d.png")
    make_plot(["normal", "jump", "drift"], f"seed{SEED}_normal_jump_drift_3d.png")
    make_plot(["normal", "jump", "drift"], f"seed{SEED}_normal_jump_drift_3d_ecef.png", frame="ecef")


if __name__ == "__main__":
    main()
