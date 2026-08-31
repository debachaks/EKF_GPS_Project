"""Per-window ensemble firing timeline for a single representative seed --
the raw fire/not-fire pattern of each ensemble voting scenario (I/II/III,
same as ensemble_detection.py) across the FULL run (pre- and post-onset),
rather than the run-level "detected" flag ensemble_detection.py reports.

This is the un-collapsed view: at every window position t, does scenario
I (any of D/G/V_final fires), II (>=2 fire), or III (all 3 fire) fire at
THAT window -- shown as a green/red strip per scenario. One figure per
(attack type, seed), stacking all 5 usable counters (hpmcounter9
excluded -- see pre_onset_audit.py / Section 8.4) as row-blocks so both
pre- and post-onset firing behavior is visible across the whole set at
once.

Generated for two representative seeds: seed1 (the convention used by
the other single-seed paper figures -- plot_hpm_rate.py,
plot_trajectory_3d.py, plot_gps_ekf_error.py) and seed2. These are
genuinely different outcomes, not two random picks: checked directly
against ensemble_detection_confusion_matrix.csv, for jump seed1 has
ZERO windows with all three metrics firing simultaneously on ANY of the
5 counters (one of hpmcounter3/jump/III's 6 false negatives), while
seed2 has >=1 simultaneous window on every one of the 5 counters (one
of its 14 true positives) -- so seed1 shows what "III never fires" looks
like and seed2 shows what "III does fire" looks like, both real.

Same three metric configurations as heatmap_4counters_midthreshold.py /
detection_confusion_matrix.py / ensemble_detection.py: D_final at W=10,
G_final and V_final at W=5. Sigma-fragile positions are treated as
not-fired (excluded from thresholding, same as everywhere else in this
project) rather than shown as a distinct color.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
PLOT_DIR = os.path.join(LINE_FITTING_DIR, "plots_heatmap")

REPRESENTATIVE_SEEDS = ["seed1", "seed2"]
ONSET_ITER = 150
TARGET_COUNTERS = ["hpmcounter3", "hpmcounter4", "hpmcounter5", "hpmcounter8", "hpmcounter10"]
ATTACK_TYPES = ["jump", "drift"]

# (short_name, score_col, thresh_col, score_path, thresh_path)
METRICS = [
    ("D", "d", "H_d", "d_final_dscore.csv", "d_final_thresholds.csv"),
    ("G", "g", "H_g", "g_final_w5_gscore.csv", "g_final_w5_thresholds.csv"),
    ("V", "v", "H_v", "v_final_w5_vscore.csv", "v_final_w5_thresholds.csv"),
]

SCENARIOS = ["III", "II", "I"]  # top-to-bottom within each counter block, matching the sketch
SCENARIO_LABEL = {
    "I": "I) A or B or C",
    "II": "II) >=2 fire",
    "III": "III) A and B and C",
}

FIRE_CMAP = ListedColormap(["#4CAF50", "#E53935"])  # green=not fire, red=fire


def flagged_series(score_path, thresh_path, score_col, thresh_col, counter, mode, seed):
    scored = pd.read_csv(os.path.join(RESULTS_DIR, score_path))
    thresholds = pd.read_csv(os.path.join(RESULTS_DIR, thresh_path))

    sub = scored[
        (scored["counter"] == counter) & (scored["mode"] == mode) & (scored["seed"] == seed)
    ].copy()
    h = thresholds.loc[thresholds["counter"] == counter, thresh_col].iloc[0]

    sub["flagged"] = (~sub["sigma_fragile"]) & (sub[score_col].abs() > h)
    return sub.set_index("window_end_iter")["flagged"].sort_index()


def scenario_fire_for_counter(counter, attack_type, seed):
    per_metric = {
        short: flagged_series(score_path, thresh_path, score_col, thresh_col,
                               counter, attack_type, seed)
        for short, score_col, thresh_col, score_path, thresh_path in METRICS
    }

    # inner-join on window_end_iter: D (W=10) starts later than G/V (W=5),
    # so the common range is bounded by D's shorter coverage.
    common_iters = per_metric["D"].index
    for s in per_metric.values():
        common_iters = common_iters.intersection(s.index)
    common_iters = np.array(sorted(common_iters))

    n_fire = sum(per_metric[short].reindex(common_iters).astype(int) for short, *_ in METRICS)
    return common_iters, {
        "I": (n_fire >= 1).to_numpy(),
        "II": (n_fire >= 2).to_numpy(),
        "III": (n_fire >= 3).to_numpy(),
    }


def make_figure(attack_type, seed):
    n_counters = len(TARGET_COUNTERS)
    n_rows = n_counters * len(SCENARIOS)
    fig, axes = plt.subplots(
        n_counters, 1, figsize=(12, 1.35 * n_counters), sharex=True,
        gridspec_kw={"hspace": 0.9},
    )

    for ax, counter in zip(axes, TARGET_COUNTERS):
        common_iters, scenario_fire = scenario_fire_for_counter(counter, attack_type, seed)
        x0, x1 = common_iters.min(), common_iters.max()

        for row, scenario in enumerate(SCENARIOS):
            data = scenario_fire[scenario].astype(int).reshape(1, -1)
            extent = [x0, x1, len(SCENARIOS) - row - 1, len(SCENARIOS) - row]
            ax.imshow(data, aspect="auto", cmap=FIRE_CMAP, vmin=0, vmax=1, extent=extent, interpolation="nearest")

        # separator lines between the 3 stacked scenario strips, so two
        # adjacent same-color (e.g. both green) strips don't blur together
        for boundary in range(1, len(SCENARIOS)):
            ax.axhline(boundary, color="black", linewidth=1.0, alpha=0.9, zorder=3)
        ax.axhline(0, color="black", linewidth=1.0, alpha=0.9, zorder=3)
        ax.axhline(len(SCENARIOS), color="black", linewidth=1.0, alpha=0.9, zorder=3)

        ax.axvline(ONSET_ITER, color="black", linewidth=1.1, linestyle="--", alpha=0.8)
        ax.set_yticks(np.arange(len(SCENARIOS)) + 0.5)
        # yticks are ascending bottom-to-top; SCENARIOS is drawn top-to-bottom,
        # so the label order must be reversed to line up with the drawn rows.
        ax.set_yticklabels([SCENARIO_LABEL[s] for s in reversed(SCENARIOS)], fontsize=7.5)
        ax.set_xlim(x0, x1)
        ax.set_ylim(0, len(SCENARIOS))
        ax.set_ylabel(counter, fontsize=9, rotation=0, ha="right", va="center", labelpad=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("EKF iteration")
    axes[0].text(ONSET_ITER, len(SCENARIOS) + 0.15, f"onset ({ONSET_ITER})", ha="center", fontsize=8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4CAF50", label="not fire"),
        plt.Rectangle((0, 0), 1, 1, color="#E53935", label="fire"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995), frameon=False, fontsize=9)

    fig.suptitle(
        f"Ensemble firing timeline -- {attack_type} ({seed})\n"
        "(A=G_final W=5, B=D_final W=10, C=V_final W=5; dashed line = attack onset; "
        "hpmcounter9 excluded, see Sec. 8.4)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(PLOT_DIR, f"ensemble_firing_timeline_{attack_type}_{seed}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    for seed in REPRESENTATIVE_SEEDS:
        for attack_type in ATTACK_TYPES:
            make_figure(attack_type, seed)


if __name__ == "__main__":
    main()
