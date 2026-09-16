"""Persistence-feature heatmaps: pre-onset (top 20 rows) vs. post-onset
(bottom 20 rows) for one attack mode, 15 counter/metric features. Visual
companion to the max-vs-persistence ablation (Table 2 of the ICC draft) --
the pre-onset block reads near-uniformly pale while the post-onset block
lights up sharply, showing directly why the persistence feature
discriminates well rather than just asserting it numerically.

Two variants per mode:
  - full-size (unchanged from the original ad-hoc version): generous
    figure, meant for a slide or standalone full-width use.
  - column: sized/tightened for a single column of a double-column IEEE
    layout, same approach as Trajectory/plot_trajectory_3d.py's
    column=True variant (small fonts, tight margins, autocrop). The
    heatmap's natural aspect ratio (taller than wide, 40 rows x 15
    columns) is already column-friendly -- a narrow width with generous
    height reads fine, unlike the 3D trajectory plot's problem.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
PLOT_DIR = os.path.join(LINE_FITTING_DIR, "plots_heatmap")

sys.path.insert(0, SCRIPT_DIR)
from ml_fusion_persistence_classifier import build_persistence_feature_table  # noqa: E402

ATTACK_MODES = ["jump", "drift"]


def autocrop(path, pad=10):
    """Trim uniform white margin on all sides down to `pad` pixels beyond
    the actual rendered content -- same approach as
    Trajectory/plot_trajectory_3d.py's autocrop()."""
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox is None:
        return
    left, top, right, bottom = bbox
    left = max(left - pad, 0)
    top = max(top - pad, 0)
    right = min(right + pad, im.size[0])
    bottom = min(bottom + pad, im.size[1])
    im.crop((left, top, right, bottom)).save(path)


def make_heatmap(table, feature_cols, mode, out_name, column=False):
    seeds = sorted(table["seed"].unique(), key=lambda s: int(s.replace("seed", "")))

    pre = table[table["row_type"] == f"{mode}_pre"].set_index("seed").loc[seeds]
    post = table[table["row_type"] == f"{mode}_post"].set_index("seed").loc[seeds]

    pre_labels = [f"{s} (pre)" for s in seeds]
    post_labels = [f"{s} (post)" for s in seeds]
    row_labels = pre_labels + post_labels

    data = np.vstack([pre[feature_cols].to_numpy(), post[feature_cols].to_numpy()])

    if column:
        fig, ax = plt.subplots(figsize=(3.5, 6.2))
        annot_fs, tick_fs, label_fs, title_fs = 3.6, 5.0, 5.5, 7.0
    else:
        fig, ax = plt.subplots(figsize=(13, 19))
        annot_fs, tick_fs, label_fs, title_fs = 9, 10, 10, 14

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if v > 0:
                color = "white" if v > data.max() * 0.6 else "black"
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=annot_fs, color=color)

    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=90, fontsize=tick_fs)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=tick_fs)

    onset_y = len(pre_labels) - 0.5
    ax.axhline(onset_y, color="blue", linewidth=1.2 if column else 2)
    ax.text(-0.7, onset_y, "onset", color="blue", fontsize=annot_fs + 1 if column else 10,
            rotation=90, va="center", ha="center")

    ax.set_title(
        f"{mode} test features: pre-onset (top 20) vs post-onset (bottom 20)\n"
        f"(persistence feature, 15 counter/metric features)",
        fontsize=title_fs,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046 if column else 0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=tick_fs)
    cbar.set_label("Longest consecutive run (windows)", fontsize=label_fs)

    out_path = os.path.join(PLOT_DIR, out_name)
    if column:
        fig.subplots_adjust(left=0.20, right=0.86, top=0.90, bottom=0.14)
        fig.savefig(out_path, dpi=400)
        plt.close(fig)
        autocrop(out_path)
    else:
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    print(f"Saved {out_path}")


METRICS = ["D", "G", "V"]
COUNTERS = ["hpmcounter3", "hpmcounter4", "hpmcounter5", "hpmcounter8", "hpmcounter10"]
PRE_COLOR = "#FDD9A0"
POST_COLOR = "#D7301F"


def make_boxplot(table, mode, out_name, ymax=None):
    """Compact alternative to the 40-row heatmap: one grouped boxplot,
    pre- vs. post-onset persistence values pooled across all 5 counters
    and 20 seeds, one pair of boxes per metric (D/G/V). Same underlying
    data and message as the heatmap (pre-onset near-zero, post-onset
    sharply elevated) in a fraction of the space -- sized to sit
    comfortably in one column without dominating the page.

    ymax: shared upper y-limit across modes (e.g. jump vs. drift) so the
    two figures are directly visually comparable at the same scale,
    rather than each auto-scaling to its own data range."""
    pre = table[table["row_type"] == f"{mode}_pre"]
    post = table[table["row_type"] == f"{mode}_post"]

    positions, box_data, colors = [], [], []
    tick_positions, tick_labels = [], []
    x = 0
    for metric in METRICS:
        cols = [f"{metric}_{c}" for c in COUNTERS]
        pre_vals = pre[cols].to_numpy().ravel()
        post_vals = post[cols].to_numpy().ravel()

        box_data.append(pre_vals)
        positions.append(x)
        colors.append(PRE_COLOR)
        box_data.append(post_vals)
        positions.append(x + 0.8)
        colors.append(POST_COLOR)

        tick_positions.append(x + 0.4)
        tick_labels.append(metric)
        x += 2.2

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    bp = ax.boxplot(box_data, positions=positions, widths=0.65, patch_artist=True,
                     medianprops=dict(color="black", linewidth=1), showfliers=True,
                     flierprops=dict(marker="o", markersize=2, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("Persistence value\n(longest consecutive run)", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ymax is not None:
        ax.set_ylim(0, ymax)
        ax.set_yticks(np.arange(0, ymax + 1, 5))

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PRE_COLOR, alpha=0.9, label="Pre-onset"),
        plt.Rectangle((0, 0), 1, 1, facecolor=POST_COLOR, alpha=0.9, label="Post-onset"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, frameon=False, loc="upper left")
    ax.set_title(f"{mode.capitalize()} trials: persistence value, pre- vs. post-onset", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(PLOT_DIR, out_name)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)
    autocrop(out_path)
    print(f"Saved {out_path}")


def main():
    table, feature_cols = build_persistence_feature_table()

    # shared y-axis scale across jump/drift so the two boxplot figures are
    # directly visually comparable rather than each auto-scaling on its own
    all_vals = []
    for mode in ATTACK_MODES:
        for row_type in (f"{mode}_pre", f"{mode}_post"):
            cols = [f"{m}_{c}" for m in METRICS for c in COUNTERS]
            all_vals.append(table[table["row_type"] == row_type][cols].to_numpy().ravel())
    shared_ymax = int(np.ceil(np.concatenate(all_vals).max() / 5.0) * 5)

    for mode in ATTACK_MODES:
        make_heatmap(table, feature_cols, mode, f"ml_persistence_test_features_{mode}_heatmap.png", column=False)
        make_heatmap(table, feature_cols, mode, f"ml_persistence_test_features_{mode}_heatmap_column.png", column=True)
        make_boxplot(table, mode, f"ml_persistence_prepost_boxplot_{mode}.png", ymax=shared_ymax)


if __name__ == "__main__":
    main()
