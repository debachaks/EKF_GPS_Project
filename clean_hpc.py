"""Clean ekf_*_hpc.csv files: drop all-zero columns, mhpmcounter/mhpmevent
columns, and any remaining column with fewer than 5 unique values (these
carry too little variation within a single run to be useful - see
hpmcounter5 in new_hpm_analysis/, which stayed exactly constant for an
entire run and turned out incapable of detecting anything happening
mid-run).

Cleaned files are written to ./CLEAN_HPC/ with the same filenames.
"""

import glob
import os

import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SRC_DIR, "CLEAN_HPC")
HPC_GLOB = os.path.join(SRC_DIR, "ekf_*_hpc.csv")


def is_all_zero(series):
    def to_num(v):
        s = str(v).strip()
        try:
            if s.lower().startswith("0x"):
                return int(s, 16)
            return float(s)
        except ValueError:
            return None

    nums = series.map(to_num)
    if nums.isna().any():
        return False
    return (nums == 0).all()


MIN_UNIQUE_VALUES = 5


def clean_file(path, out_dir):
    df = pd.read_csv(path)

    drop_cols = [
        col for col in df.columns
        if "mhpmcounter" in col.lower() or "mhpmevent" in col.lower()
    ]
    df = df.drop(columns=drop_cols)

    zero_cols = [col for col in df.columns if is_all_zero(df[col])]
    df = df.drop(columns=zero_cols)

    low_variance_cols = [col for col in df.columns if df[col].nunique() < MIN_UNIQUE_VALUES]
    df = df.drop(columns=low_variance_cols)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(path))
    df.to_csv(out_path, index=False)

    print(
        f"{os.path.basename(path)}: dropped {len(drop_cols)} mhpmcounter/mhpmevent "
        f"cols, {len(zero_cols)} all-zero cols, {len(low_variance_cols)} cols with "
        f"< {MIN_UNIQUE_VALUES} unique values -> {df.shape[1]} cols remain "
        f"({out_path})"
    )


def main():
    files = sorted(glob.glob(HPC_GLOB))
    if not files:
        print(f"No files matched {HPC_GLOB}")
        return
    for path in files:
        clean_file(path, OUT_DIR)


if __name__ == "__main__":
    main()
