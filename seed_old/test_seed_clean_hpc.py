"""Clean ekf_*_hpc.csv files for all test_seed_N/ folders found (prev/original
hpmcounter mapping - same mapping as seed_1_data..seed_5_data).

Same cleaning rule as clean_hpc.py (drop mhpmcounter/mhpmevent columns,
all-zero columns, and any remaining column with < 5 unique values), applied
per seed folder. Cleaned files go to CLEAN_HPC_TEST_SEED/test_seed_N/, kept
separate from the older flat CLEAN_HPC_TEST_SEED/*.csv output (which predates
the test_seed_1..4 split).
"""

import glob
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from clean_hpc import clean_file  # noqa: E402

OUT_ROOT = os.path.join(SRC_DIR, "CLEAN_HPC_TEST_SEED")
SEED_GLOB = os.path.join(SRC_DIR, "test_seed_[0-9]*")


def find_seed_dirs():
    return sorted(d for d in glob.glob(SEED_GLOB) if os.path.isdir(d))


def main():
    for seed_dir in find_seed_dirs():
        seed_name = os.path.basename(seed_dir)
        if not os.path.isdir(seed_dir):
            print(f"Skipping missing folder {seed_dir}")
            continue
        out_dir = os.path.join(OUT_ROOT, seed_name)
        files = sorted(glob.glob(os.path.join(seed_dir, "ekf_*_hpc.csv")))
        for path in files:
            clean_file(path, out_dir)


if __name__ == "__main__":
    main()
