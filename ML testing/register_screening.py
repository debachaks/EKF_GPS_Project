"""Screen every column in the CLEANED normal-mode board dumps (all 20 seeds)
for whether it's even a *candidate* for an ML anomaly model, before spending
more effort building one.

EXCLUDED, and why:
    timestamp_ms                         - not a signal, it's the clock.
    fs0, fa0-fa5 (float registers)       - per the RISC-V calling convention
        these are exactly the registers ekf_step()/compute_nis() pass
        doubles through (measured/true position, innovation). Their content
        is a direct, deterministic snapshot of the EKF's own state --
        already shown byte-identical run to run in rerun_comparison.py's
        diag check. Excluded per the request: "not the registers which
        directly store coordinate/innovation value."
    a0-a7, t0-t2, t4-t6 (general-purpose scratch/argument registers), dpc
        - checked directly (see conversation) and these are NOT a stable
        quantity at all: e.g. `a7` ranges from 0 to ~1.8e19 WITHIN A SINGLE
        RUN, because the compiler reuses these physical registers for
        completely different logical values from one line of code to the
        next (sometimes a small loop counter, sometimes a raw bit pattern
        of a double that got register-allocated there). Their "value" at
        an async register-dump snapshot depends on which instruction
        happened to be mid-execution, not on any consistent thing being
        measured. That's a different, worse problem than fa0-fa5 being
        "deterministic" -- it's not even coherent enough to be
        deterministic-and-uninteresting, it's just noise with no fixed
        referent. `dpc` was also dropped: verified byte-identical to `pc`
        (debug PC mirrors PC exactly), so it's a duplicate column.

KEPT as candidates:
    sp, fp, ra, pc  - control-flow/address registers. Checked directly:
        these stay in a tight, consistent ~0x8001____ address range for
        the whole run (unlike a-regs/t-regs) -- a stable, coherent quantity
        (stack depth / return address / code location at snapshot time),
        even though it's not a counter. Treated as a raw value, not a rate.
    mcycle, minstret, cycle, instret, hpmcounter3-10
        - real hardware counters (mcycle/minstret are the privileged
        mirrors of cycle/instret and are numerically identical to them;
        kept both only for completeness). Treated as rate (first
        difference), same convention as variance_decomposition.py.

For each candidate, on NORMAL-mode data only (all 20 seeds): between-seed
vs within-seed variability, same ratio as variance_decomposition.py --
ratio > 1 means the column has genuine seed-to-seed structure (a candidate
signal source, i.e. sensitive to the actual GPS-noise realization rather
than just hardware jitter); ratio << 1 means it's dominated by run-to-run
jitter and would need attack effects far bigger than that jitter floor to
ever be usable.

NOTE ON DECODING: hex_to_int decodes as UNSIGNED. That's a no-op for
counters (hardware counters never decrease, so unsigned vs signed diff
gives the same answer) but WOULD silently corrupt a diff on any register
that can decrease (e.g. this bit me first when I tried treating a-regs/
t-regs as candidates: np.diff on a uint64 array wraps around on a decrease
instead of going negative). Re-decoded here as signed 64-bit for the
raw-value registers (sp/fp/ra/pc) so this can't happen even though, in
practice, their observed range never crosses the sign boundary.

Output: results/register_screening_summary.csv + console table, sorted by
between/within ratio (most promising first).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
while not os.path.isdir(os.path.join(PROJECT_ROOT, "original_pipeline")):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from data_preprocessing import hex_to_int  # noqa: E402

CLEAN_ROOT = os.path.join(PROJECT_ROOT, "seed_old", "CLEAN_HPC_TEST_SEED")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

COUNTER_COLS = ["mcycle", "minstret", "cycle", "instret",
                "hpmcounter3", "hpmcounter4", "hpmcounter5", "hpmcounter6",
                "hpmcounter7", "hpmcounter8", "hpmcounter9", "hpmcounter10"]
RAW_VALUE_COLS = ["sp", "fp", "ra", "pc"]


def find_seed_names():
    dirs = sorted(d for d in glob.glob(os.path.join(CLEAN_ROOT, "test_seed_[0-9]*")) if os.path.isdir(d))
    return [os.path.basename(d) for d in dirs]


def hex_to_signed64(val):
    n = hex_to_int(val)
    return n - (1 << 64) if n >= (1 << 63) else n


def load_column(seed_name, column, signed=False):
    path = os.path.join(CLEAN_ROOT, seed_name, "ekf_normal_hpc.csv")
    df = pd.read_csv(path)
    decode = hex_to_signed64 if signed else hex_to_int
    return df[column].map(decode).to_numpy()


def screen_column(seed_names, column, kind):
    signed = kind == "raw_value"
    per_seed_values = {s: load_column(s, column, signed=signed) for s in seed_names}

    within_stds = []
    seed_summaries = []
    for v in per_seed_values.values():
        series = np.diff(v).astype(float) if kind == "counter" else v.astype(float)
        within_stds.append(series.std(ddof=1) if len(series) > 1 else 0.0)
        seed_summaries.append(series.mean())

    within_seed_std = np.mean(within_stds)
    between_seed_std = np.std(seed_summaries, ddof=1)
    ratio = between_seed_std / within_seed_std if within_seed_std > 0 else np.nan

    all_values = np.concatenate(list(per_seed_values.values()))
    return {
        "column": column,
        "kind": "counter (rate analyzed)" if kind == "counter" else "raw value (address register)",
        "n_unique": len(np.unique(all_values)),
        "within_seed_std_avg": within_seed_std,
        "between_seed_std": between_seed_std,
        "between_within_ratio": ratio,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    seed_names = find_seed_names()
    print(f"{len(seed_names)} normal seeds")
    print(f"Candidates: {len(RAW_VALUE_COLS)} address registers + {len(COUNTER_COLS)} counters")
    print("Excluded: fs0/fa0-fa5 (deterministic EKF state), a0-a7/t0-t2/t4-t6/dpc (incoherent scratch regs)\n")

    rows = [screen_column(seed_names, col, "raw_value") for col in RAW_VALUE_COLS]
    rows += [screen_column(seed_names, col, "counter") for col in COUNTER_COLS]

    result = pd.DataFrame(rows).sort_values("between_within_ratio", ascending=False, na_position="last")

    pd.set_option("display.width", 160)
    print(result.to_string(index=False))

    out_path = os.path.join(RESULTS_DIR, "register_screening_summary.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
