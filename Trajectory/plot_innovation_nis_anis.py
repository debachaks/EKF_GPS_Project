"""Three-panel diagnostic: innovation norm, NIS, and ANIS vs. EKF
iteration, for one seed, normal vs. jump vs. drift -- reads directly
from the ekf_diag_<mode>.csv diagnostic dumps (which already carry
innovation_norm/nis/anis per timestep).
"""

import os

import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

SEED = 1
SEED_DIR = os.path.join(PROJECT_ROOT, "adaptive_analysis_board_firmware", "plots", f"seed{SEED}_newMapping")

NIS_THRESHOLD = 11.345
ANIS_THRESHOLD = 4.377

MODE_COLORS = {
    "normal": "#111111",
    "jump": "#E07B1A",
    "drift": "#2E8B3D",
}


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    dfs = {mode: pd.read_csv(os.path.join(SEED_DIR, f"ekf_diag_{mode}.csv")) for mode in MODE_COLORS}

    plt.rcParams.update({"font.size": 12, "font.family": "serif"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    panels = [
        ("innovation_norm", r"Innovation norm $\Vert z_k - H\hat{x}_k \Vert$ [m]", None, False),
        ("nis", "NIS", NIS_THRESHOLD, True),
        ("anis", "ANIS (rolling window = 10)", ANIS_THRESHOLD, True),
    ]

    for ax, (col, ylabel, threshold, log_scale) in zip(axes, panels):
        for mode, color in MODE_COLORS.items():
            df = dfs[mode]
            ax.plot(df["t"], df[col], color=color, linewidth=1.8, label=mode)
        if threshold is not None:
            ax.axhline(threshold, color="#3CA6C7", linewidth=2.2, linestyle="--",
                       label=f"{col.upper()} threshold", zorder=0)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("EKF iteration")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(loc="upper left", fontsize=10, frameon=False)
    axes[1].legend(loc="upper left", fontsize=10, frameon=False)
    axes[2].legend(loc="upper left", fontsize=10, frameon=False)

    fig.suptitle(f"Seed {SEED} — innovation / NIS / ANIS, normal vs. jump vs. drift", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"seed{SEED}_innovation_nis_anis.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
