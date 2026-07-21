"""Preprocessing helpers for the cleaned ekf_*_hpc.csv files in CLEAN_HPC/."""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(SRC_DIR, "CLEAN_HPC")
PLOT_DIR = os.path.join(SRC_DIR, "plots")


def hex_to_int(val):
    """Convert a hex string like '0x00...1f' to an int. Passes through plain numbers."""
    s = str(val).strip()
    if s.lower().startswith("0x"):
        return int(s, 16)
    return int(float(s))


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    for path in sorted(glob.glob(os.path.join(CLEAN_DIR, "ekf_*_hpc.csv"))):
        df = pd.read_csv(path)

        mcycle = df["mcycle"].map(hex_to_int)
        fp = df["fp"].map(hex_to_int)

        name = os.path.splitext(os.path.basename(path))[0]
        plt.figure(figsize=(10, 5))
        plt.plot(mcycle, fp, marker=".", linestyle="none")
        plt.xlabel("mcycle")
        plt.ylabel("fp")
        plt.title(f"mcycle vs fp - {name}")
        plt.tight_layout()

        out_path = os.path.join(PLOT_DIR, f"{name}_mcycle_vs_fp.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
