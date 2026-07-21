"""Compare floating-point registers across normal/drift/jump/replay in seed_new/.

seed_new/'s ekf_*_hpc.csv files store each FP register (ft0-11, fs0-11, fa0-7)
as the raw 64-bit bit pattern of an IEEE-754 double, plus three FP
control/status registers (fflags, frm, fcsr) as plain small integers. This
script decodes the doubles properly (bit-cast, not just hex->int) and checks,
for each register, whether it:

  - is constant (same bit pattern every row) within every mode - and if so,
    whether that constant differs across modes (a "fixed fingerprint" signal,
    the same pattern seen with hpmcounter5 in this dataset), or
  - varies row-to-row within a run, in which case it runs the same
    Mann-Whitney U + Cliff's delta + BH-FDR test used elsewhere in this repo.

Caveat: seed_new/ is one run per mode. Row-level tests on varying registers
are optimistic (autocorrelated single time series, not independent samples -
see run_level_hpmcounter_analysis.py for why). Constant-value comparisons
avoid that specific problem but are still a single run per mode, so any
signal here is a candidate to confirm with more seed_new-style runs, not a
settled result.
"""

import os
import struct
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from hpmcounter_analysis import benjamini_hochberg, cliffs_delta  # noqa: E402

SEED_DIR = os.path.join(SRC_DIR, "seed_new")
CONST_OUT = os.path.join(SRC_DIR, "seed_new_fp_constant_registers.csv")
VARYING_OUT = os.path.join(SRC_DIR, "seed_new_fp_varying_stats.csv")

DOUBLE_REGS = (
    [f"ft{i}" for i in range(12)]
    + [f"fs{i}" for i in range(12)]
    + [f"fa{i}" for i in range(8)]
)
STATUS_REGS = ["fflags", "frm", "fcsr"]
FP_COLS = DOUBLE_REGS + STATUS_REGS
MODES = ["normal", "drift", "jump", "replay"]
ANOMALY_MODES = ["drift", "jump", "replay"]


def bits_to_double(hex_str):
    s = str(hex_str).strip()
    bits = int(s, 16) if s.lower().startswith("0x") else int(s)
    bits &= (1 << 64) - 1
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def load_mode(mode):
    path = os.path.join(SEED_DIR, f"ekf_{mode}_hpc.csv")
    df = pd.read_csv(path)
    decoded = {}
    for col in DOUBLE_REGS:
        decoded[col] = df[col].map(bits_to_double)
    for col in STATUS_REGS:
        decoded[col] = df[col].map(lambda v: int(str(v), 16) if str(v).lower().startswith("0x") else int(v))
    return pd.DataFrame(decoded)


def main():
    data = {mode: load_mode(mode) for mode in MODES}

    const_rows = []
    varying_cols = []
    for col in FP_COLS:
        nunique_per_mode = {mode: data[mode][col].nunique() for mode in MODES}
        if all(n == 1 for n in nunique_per_mode.values()):
            values = {mode: data[mode][col].iloc[0] for mode in MODES}
            distinct_values = len(set(values.values()))
            const_rows.append({
                "register": col,
                **{f"value_{mode}": values[mode] for mode in MODES},
                "differs_across_modes": distinct_values > 1,
            })
        else:
            varying_cols.append(col)

    const_df = pd.DataFrame(const_rows).sort_values(
        "differs_across_modes", ascending=False
    )
    const_df.to_csv(CONST_OUT, index=False)

    print(f"{len(const_rows)} registers are constant within every mode "
          f"({(const_df['differs_across_modes']).sum()} of those differ across modes):")
    print(const_df.to_string(index=False))
    print(f"Saved to {CONST_OUT}\n")

    print(f"{len(varying_cols)} registers vary row-to-row within a run: {varying_cols}")

    if varying_cols:
        rows = []
        normal = data["normal"]
        for mode in ANOMALY_MODES:
            anomaly = data[mode]
            for col in varying_cols:
                x = anomaly[col].to_numpy()
                y = normal[col].to_numpy()
                u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
                delta = cliffs_delta(x, y)
                rows.append({
                    "register": col,
                    "anomaly_mode": mode,
                    "median_anomaly": np.median(x),
                    "median_normal": np.median(y),
                    "U": u_stat,
                    "p_value": p_value,
                    "cliffs_delta": delta,
                })
        results = pd.DataFrame(rows)
        results["p_value_fdr"] = benjamini_hochberg(results["p_value"].to_numpy())
        results["significant_fdr_0.05"] = results["p_value_fdr"] < 0.05
        results = results.sort_values("p_value_fdr").reset_index(drop=True)
        results.to_csv(VARYING_OUT, index=False)
        print(results.to_string(index=False))
        print(f"Saved to {VARYING_OUT}")

    print(
        "\nCaveat: seed_new/ is a single run per mode. Row-level tests above "
        "are optimistic (autocorrelated within-run values, not independent "
        "samples), and even the constant-register comparisons are only a "
        "1-vs-1-vs-1-vs-1 run comparison - confirm with more runs before "
        "trusting any single register as a real discriminator."
    )


if __name__ == "__main__":
    main()
