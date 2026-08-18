"""lambda_k (Strong Tracking Filter fading factor) vs. EKF iteration,
for one seed, normal vs. jump vs. drift -- reads directly from the
ekf_diag_<mode>.csv diagnostic dumps (lambda_k column).

log-scaled y-axis: jump's lambda_k spikes into the tens of thousands
right at attack onset while normal/drift stay in the single digits, so a
linear axis flattens everything but jump into an invisible line (same
issue as the NIS/ANIS plot).
"""

import os

import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")

SEED = 2
SEED_DIR = os.path.join(PROJECT_ROOT, "adaptive_analysis_board_firmware", "plots", f"seed{SEED}_newMapping")

MODE_COLORS = {
    "normal": "#111111",
    "jump": "#E07B1A",
    "drift": "#2E8B3D",
}


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    dfs = {mode: pd.read_csv(os.path.join(SEED_DIR, f"ekf_diag_{mode}.csv")) for mode in MODE_COLORS}

    plt.rcParams.update({"font.size": 12, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(7, 4.6))

    for mode, color in MODE_COLORS.items():
        df = dfs[mode]
        ax.plot(df["t"], df["lambda_k"], color=color, linewidth=1.8, label=mode)

    ax.axhline(1.0, color="#3CA6C7", linewidth=2.2, linestyle="--", label=r"$\lambda_k = 1$ (floor)", zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("EKF iteration")
    ax.set_ylabel(r"$\lambda_k$ (STF fading factor)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=10, frameon=False)

    fig.suptitle(f"Seed {SEED} — Strong Tracking Filter fading factor", y=1.0)
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"seed{SEED}_lambda_k.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
