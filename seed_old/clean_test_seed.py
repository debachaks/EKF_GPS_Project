"""Clean ekf_*_hpc.csv files in test_seed/ (prev/original hpmcounter
mapping - same mapping as seed_1_data..seed_5_data). Reuses clean_file()
from clean_hpc.py (mhpmcounter/mhpmevent drop + all-zero drop + <5
unique-values drop). Cleaned files written to CLEAN_HPC_TEST_SEED/.
"""

import glob
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from clean_hpc import clean_file  # noqa: E402

SEED_DIR = os.path.join(SRC_DIR, "test_seed")
OUT_DIR = os.path.join(SRC_DIR, "CLEAN_HPC_TEST_SEED")


def main():
    files = sorted(glob.glob(os.path.join(SEED_DIR, "ekf_*_hpc.csv")))
    if not files:
        print(f"No files matched {os.path.join(SEED_DIR, 'ekf_*_hpc.csv')}")
        return
    for path in files:
        clean_file(path, OUT_DIR)


if __name__ == "__main__":
    main()
