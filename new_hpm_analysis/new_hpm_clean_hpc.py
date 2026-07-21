"""Clean ekf_*_hpc.csv files for the new-HPM-mapping runs (seed_new_1..5).

Same cleaning rule as clean_hpc.py (drop mhpmcounter/mhpmevent columns and
all-zero columns, per file), just applied across multiple seed_new_N/
folders instead of a single flat directory. Everything for this new-HPM
dataset - scripts and outputs - lives in this new_hpm_analysis/ folder, kept
separate from the original seed_1_data..seed_5_data pipeline at the project
root.
"""

import glob
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "original_pipeline"))

from clean_hpc import clean_file  # noqa: E402

OUT_ROOT = os.path.join(SCRIPT_DIR, "CLEAN_HPC_NEW_HPM")
SEED_GLOB = os.path.join(PROJECT_ROOT, "seed_new_[0-9]*")


def find_seed_dirs():
    return sorted(d for d in glob.glob(SEED_GLOB) if os.path.isdir(d))


def main():
    seed_dirs = find_seed_dirs()
    if not seed_dirs:
        print(f"No seed_new_N folders found matching {SEED_GLOB}")
        return

    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        out_dir = os.path.join(OUT_ROOT, seed_name)
        files = sorted(glob.glob(os.path.join(seed_dir, "ekf_*_hpc.csv")))
        for path in files:
            clean_file(path, out_dir)


if __name__ == "__main__":
    main()
