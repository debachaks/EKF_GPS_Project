"""Combined 18-unique-event PCA counter selection.

set_1/set_2/set_3 were collected as separate physical runs (only 8 HPM
slots available at a time), so they can't be stacked by raw row index -
row counts differ slightly even for the same seed (e.g. seed_3: 310, 309,
314 rows), consistent with the async JTAG-sampling jitter documented
throughout seed_old/. Instead, for each seed, every set's per-event rate
trace is interpolated onto a shared elapsed-time grid before combining -
same technique as seed_old/test_seed_diff_trace_per_seed.py.

Four events were deliberately measured in more than one set as
cross-check anchors (int arithmetic retired, pipeline flushes,
cond-branch mispredictions, long-latency interlock) - stacking all 24 raw
columns would double/triple-count those events and bias PCA toward them
for no real reason. UNIQUE_EVENTS below de-duplicates to the true 18
distinct events, averaging the repeated measurements (justified by
pca_counter_selection.py's anchor-consistency check, which found the
repeated measurements agree closely with each other).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from data_preprocessing import hex_to_int  # noqa: E402

COUNTERS = [f"hpmcounter{i}" for i in range(3, 11)]
GRID_POINTS = 300
VARIANCE_CAPTURED_TARGET = 0.90

UNIQUE_EVENTS = {
    "FP interlock cycles": [("set_1", "hpmcounter3")],
    "FP div/sqrt retired": [("set_1", "hpmcounter4")],
    "cond-branch mispredictions": [("set_1", "hpmcounter5"), ("set_3", "hpmcounter8")],
    "long-latency interlock": [("set_1", "hpmcounter6"), ("set_3", "hpmcounter7")],
    "addr-generation interlock": [("set_1", "hpmcounter7")],
    "int arithmetic retired": [("set_1", "hpmcounter8"), ("set_2", "hpmcounter9"), ("set_3", "hpmcounter9")],
    "pipeline flushes": [("set_1", "hpmcounter9"), ("set_2", "hpmcounter10"), ("set_3", "hpmcounter10")],
    "exceptions taken": [("set_1", "hpmcounter10")],
    "FP add retired": [("set_2", "hpmcounter3")],
    "FP multiply retired": [("set_2", "hpmcounter4")],
    "FP fused-multiply-add retired": [("set_2", "hpmcounter5")],
    "FP load retired": [("set_2", "hpmcounter6")],
    "FP store retired": [("set_2", "hpmcounter7")],
    "other FP retired": [("set_2", "hpmcounter8")],
    "d-cache misses": [("set_3", "hpmcounter3")],
    "d-cache blocked cycles": [("set_3", "hpmcounter4")],
    "d-cache writeback requests": [("set_3", "hpmcounter5")],
    "mul/div interlock": [("set_3", "hpmcounter6")],
}

SEEDS = sorted(os.path.basename(d) for d in glob.glob(os.path.join(SRC_DIR, "seed_*")))


def rate_trace(seed, set_name, counter):
    path = os.path.join(SRC_DIR, seed, set_name, "ekf_normal_hpc.csv")
    df = pd.read_csv(path)

    mcycle_delta = df["mcycle"].map(hex_to_int).diff()
    counter_delta = df[counter].map(hex_to_int).diff()
    rate = (counter_delta / mcycle_delta).iloc[1:].replace([np.inf, -np.inf], np.nan)

    elapsed = df["timestamp_ms"].map(hex_to_int) - df["timestamp_ms"].map(hex_to_int).iloc[0]
    elapsed_aligned = elapsed.iloc[1:]

    valid = rate.notna()
    return elapsed_aligned[valid].to_numpy(), rate[valid].to_numpy()


def build_seed_matrix(seed):
    """(GRID_POINTS x 18) matrix for one seed, events aligned by elapsed
    time and de-duplicated (averaged) across sets."""
    set_names = {"set_1", "set_2", "set_3"}
    raw_traces = {
        (set_name, counter): rate_trace(seed, set_name, counter)
        for set_name in set_names
        for counter in COUNTERS
    }

    grid_end = min(elapsed.max() for elapsed, _ in raw_traces.values())
    grid = np.linspace(0, grid_end, GRID_POINTS)

    interp = {
        key: np.interp(grid, elapsed, values)
        for key, (elapsed, values) in raw_traces.items()
    }

    columns = {}
    for event_name, locations in UNIQUE_EVENTS.items():
        traces = [interp[loc] for loc in locations]
        columns[event_name] = np.mean(traces, axis=0)

    return pd.DataFrame(columns)


def pca_importance(matrix, events):
    std = matrix.std()
    zero_variance = std[std < 1e-12].index.tolist()

    usable = [e for e in events if e not in zero_variance]
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
        "event": usable,
        "importance": importance,
        "std_of_rate": [matrix[e].std() for e in usable],
        "mean_of_rate": [matrix[e].mean() for e in usable],
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    for e in zero_variance:
        result.loc[len(result)] = [e, 0.0, 0.0, matrix[e].mean()]

    return result, evr, n_components


def main():
    print(f"Combining {len(SEEDS)} seeds, {len(UNIQUE_EVENTS)} unique events "
          f"(de-duplicated from {sum(len(v) for v in UNIQUE_EVENTS.values())} raw columns)")

    matrix = pd.concat([build_seed_matrix(seed) for seed in SEEDS], ignore_index=True)
    print(f"Combined pooled rows: {len(matrix)}\n")

    ranking, evr, n_components = pca_importance(matrix, list(UNIQUE_EVENTS.keys()))
    print(f"PCA explained variance ratio per component: {np.round(evr, 3).tolist()}")
    print(f"Using top {n_components} component(s) to reach {VARIANCE_CAPTURED_TARGET:.0%} variance captured\n")

    print(f"{'rank':<5}{'importance':<12}{'std_of_rate':<14}{'mean_of_rate':<14}event")
    for i, row in ranking.iterrows():
        print(f"{i+1:<5}{row['importance']:<12.4f}{row['std_of_rate']:<14.6g}"
              f"{row['mean_of_rate']:<14.6g}{row['event']}")

    out_path = os.path.join(SRC_DIR, "combined_18event_pca_ranking.csv")
    ranking.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
