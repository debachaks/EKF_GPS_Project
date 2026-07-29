"""PCA-based register selection, separate from the hpmcounter event PCA.

Unlike the hpmcounter events (which were split across 3 separate hardware
configurations, one 8-counter set at a time - see
pca_counter_selection_combined.py for why that needs cross-set time
alignment), every general-purpose and floating-point register is captured
in EVERY snapshot regardless of which HPM set is active. So rows can be
pooled directly across all 15 normal runs (5 seeds x 3 sets) as
independent samples - each row already has all registers measured
simultaneously, no alignment needed.

Integer registers are decoded as plain hex integers. Floating-point
registers are decoded as IEEE-754 doubles (as in
seed_old/test_seed_raw_trace_per_seed.py's fa0/fa1 check, which found
those registers hold live position values, not addresses).

Raw register VALUE is used (not rate) - registers are snapshots, not
accumulating counters, so a delta/mcycle-delta rate isn't meaningful here
the way it is for hpmcounter events.
"""

import glob
import os
import struct

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANCE_CAPTURED_TARGET = 0.90

INT_REGS = ["ra", "sp", "gp", "tp", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "fp",
            "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
            "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]
FP_REGS = ["ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
           "fs0", "fs1", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9", "fs10", "fs11",
           "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7",
           "ft8", "ft9", "ft10", "ft11"]


def hex_to_int(val):
    s = str(val).strip()
    if s.lower().startswith("0x"):
        return int(s, 16)
    return int(float(s))


def hex_to_double(val):
    s = str(val).strip()
    try:
        b = bytes.fromhex(s.replace("0x", "").replace("0X", "").zfill(16))
        return struct.unpack(">d", b)[0]
    except (ValueError, struct.error):
        return np.nan


def find_runs():
    return sorted(glob.glob(os.path.join(SRC_DIR, "seed_*", "set_*")))


def load_pooled_registers():
    rows = []
    for run_dir in find_runs():
        path = os.path.join(run_dir, "ekf_normal_hpc.csv")
        df = pd.read_csv(path)

        decoded = pd.DataFrame(index=df.index)
        for reg in INT_REGS:
            if reg in df.columns:
                decoded[reg] = df[reg].map(hex_to_int)
        for reg in FP_REGS:
            if reg in df.columns:
                decoded[reg] = df[reg].map(hex_to_double)

        rows.append(decoded)
    return pd.concat(rows, ignore_index=True)


def pca_importance(matrix, columns):
    std = matrix[columns].std()
    zero_variance = std[std.isna() | (std < 1e-12)].index.tolist()

    usable = [c for c in columns if c not in zero_variance]
    scaler = StandardScaler()
    z = scaler.fit_transform(matrix[usable])

    pca = PCA()
    pca.fit(z)

    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)
    n_components = int(np.searchsorted(cumulative, VARIANCE_CAPTURED_TARGET) + 1)
    n_components = min(n_components, len(evr))

    loadings = pca.components_[:n_components]
    evr_used = evr[:n_components]
    importance = (loadings**2 * evr_used[:, None]).sum(axis=0)

    result = pd.DataFrame({
        "register": usable,
        "importance": importance,
        "std": [matrix[c].std() for c in usable],
        "mean": [matrix[c].mean() for c in usable],
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    for c in zero_variance:
        result.loc[len(result)] = [c, 0.0, 0.0, matrix[c].mean() if c in matrix.columns else np.nan]

    return result, evr, n_components


def main():
    matrix = load_pooled_registers()
    print(f"Pooled rows across {len(find_runs())} runs (5 seeds x 3 sets): {len(matrix)}\n")

    for label, columns in [("Integer (GPR) registers", INT_REGS), ("Floating-point registers", FP_REGS)]:
        present = [c for c in columns if c in matrix.columns]
        print(f"{'='*70}\n{label} ({len(present)} present)\n{'='*70}")

        ranking, evr, n_components = pca_importance(matrix, present)
        print(f"PCA explained variance ratio per component: {np.round(evr, 3)[:10].tolist()}"
              f"{' ...' if len(evr) > 10 else ''}")
        print(f"Using top {n_components} component(s) to reach {VARIANCE_CAPTURED_TARGET:.0%} variance captured\n")

        print(f"{'rank':<5}{'register':<10}{'importance':<12}{'std':<16}{'mean':<16}")
        for i, row in ranking.iterrows():
            print(f"{i+1:<5}{row['register']:<10}{row['importance']:<12.4f}{row['std']:<16.6g}{row['mean']:<16.6g}")

        out_path = os.path.join(SRC_DIR, f"{'int' if 'Integer' in label else 'fp'}_register_pca_ranking.csv")
        ranking.to_csv(out_path, index=False)
        print(f"Saved {out_path}\n")


if __name__ == "__main__":
    main()
