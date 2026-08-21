"""Per-seed breakdown of which detection criteria (z_stat, D, V) fired for
which (seed, counter, mode) -- the same underlying data as
combined_detection.py's aggregate summary, just not collapsed across
seeds, so you can see exactly which individual seeds each criterion
caught rather than only a total count.

Reads combined_detection_flags.csv (per-window flags from
combined_detection.py) and collapses each (counter, mode, seed) run down
to "did ANY window of this run trip z_flag / d_flag / v_flag", then
prints one table per mode with a compact per-seed x per-counter view:
    "."   = nothing fired
    "z"   = only z_stat (raw |z| spike) fired
    "D"   = only trend score fired
    "V"   = only variability score fired
    combinations are concatenated, e.g. "DV" = both D and V fired

Outputs:
    results/per_seed_breakdown.csv   - long format, one row per
                                        (mode, seed, counter) with the
                                        three booleans
"""

import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LINE_FITTING_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(LINE_FITTING_DIR, "results")
FLAGS_PATH = os.path.join(RESULTS_DIR, "combined_detection_flags.csv")
OUT_PATH = os.path.join(RESULTS_DIR, "per_seed_breakdown.csv")

COUNTERS = [f"hpmcounter{i}" for i in range(3, 11)]


def seed_sort_key(seed):
    return int(seed.replace("seed", ""))


def symbol(row):
    s = ""
    if row["z_flag"]:
        s += "z"
    if row["d_flag"]:
        s += "D"
    if row["v_flag"]:
        s += "V"
    return s if s else "."


def main():
    flags = pd.read_csv(FLAGS_PATH)

    run_flags = flags.groupby(["mode", "seed", "counter"])[["z_flag", "d_flag", "v_flag"]].any().reset_index()
    run_flags.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH} ({len(run_flags)} rows)\n")

    for mode in sorted(run_flags["mode"].unique()):
        sub = run_flags[run_flags["mode"] == mode].copy()
        sub["symbol"] = sub.apply(symbol, axis=1)
        pivot = sub.pivot(index="seed", columns="counter", values="symbol")
        pivot = pivot.reindex(sorted(pivot.index, key=seed_sort_key))
        pivot = pivot[[c for c in COUNTERS if c in pivot.columns]]

        print(f"=== mode={mode} ===  (. = nothing fired, z=z_stat, D=trend, V=variability)")
        print(pivot.to_string())

        any_fired = (pivot != ".").any(axis=1)
        n_any = any_fired.sum()
        print(f"\nSeeds with at least one counter firing (any criterion): {n_any}/{len(pivot)}")
        if n_any < len(pivot):
            print(f"Seeds with NOTHING firing on ANY counter: {list(pivot.index[~any_fired])}")
        print()


if __name__ == "__main__":
    main()
