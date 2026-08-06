"""
generate_trajectories.py

Generates precomputed trajectory headers (true + measured positions)
for normal/jump/drift/replay, with multiple independent noise seeds
per mode -- so each "run" of a given mode is a genuinely independent
sample, not a re-measurement of the same fixed data.

============================================================
WHY THIS EXISTS (recap)
============================================================
Previously, main_ekf.c generated noise and applied the attack
on-device, at runtime, every run with a different RNG seed from
mcycle. Two problems with that:

1. CONFOUND: HPC measurements partly captured the cost of the
   noise/spoof-generation CODE itself, not just the EKF's response
   to spoofed data. Every mode ran different amounts of that
   generation code (jump/drift/replay all differ), so any counter
   difference you saw could have been from that, not the attack.

2. NO REPRODUCIBILITY / NO INDEPENDENT TRIALS: every physical run
   used different noise (seeded from mcycle), so you had exactly
   one sample per condition with no way to know how much variation
   is normal vs attack-driven.

This script fixes both: the board just reads precomputed numbers
(same code path for every mode -- no generation logic on-device at
all), and multiple seeds per mode give you actual independent
trials to measure baseline variation against.

============================================================
CONFIG SOURCE OF TRUTH
============================================================
Attack parameters (ATTACK_START, JUMP_OFFSET_M, DRIFT_RATE_MS,
DRIFT_MAX_M, REPLAY_DELAY, GPS_SIGMA_H, GPS_SIGMA_V) are PARSED
DIRECTLY from your real ekf_config.h, not retyped here. If a
required constant is missing or unparseable, this script fails
loudly rather than silently assuming a value -- so it can never
silently drift out of sync with the C config.

============================================================
REPLAY SEMANTICS -- MATCHES THE ORIGINAL ON-DEVICE LOGIC EXACTLY
============================================================
In the original main_ekf.c, hist_push()/hist_get() operated on the
NOISY measured position (rx, ry, rz), not the true position. So a
replay attack substitutes the current noisy measurement with a
PAST NOISY MEASUREMENT from REPLAY_DELAY steps ago -- not the true
position from that time. This script replicates that exactly: the
history buffer used for replay stores measured (noised) positions,
matching the original semantics precisely.

============================================================
Usage:
    python3 generate_trajectories.py

Set EKF_CONFIG_PATH, N_SEEDS, and the true-trajectory generation
parameters below before running.
"""

import os
import re
import math
import random


# ============================================================
EKF_CONFIG_PATH = "ekf_config.h"   # path to your real ekf_config.h
OUTPUT_DIR = "."

TRAJ_LEN = 300   # must match your existing trajectory.h TRAJ_LEN
N_SEEDS = 20      # independent trials per mode -- change as needed

MODES = ["normal", "jump", "drift", "replay"]

# True-trajectory generation parameters (same as your existing
# generate_traj.py -- straight-line northward path)
LAT0 = 48.1173
LON0 = 11.5166
ALT0 = 545.0
LAT_STEP_DEG = 0.000009   # ~1m/step northward
# ============================================================


REQUIRED_CONSTANTS = [
    "ATTACK_START", "JUMP_OFFSET_M", "DRIFT_RATE_MS",
    "DRIFT_MAX_M", "REPLAY_DELAY", "GPS_SIGMA_H", "GPS_SIGMA_V",
]


def parse_ekf_config(path):
    """
    Parses required constants directly out of ekf_config.h via
    simple #define regex matching. Fails loudly (raises) if any
    required constant is missing -- this is deliberate, so this
    script can never silently drift out of sync with the real C
    config.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find ekf_config.h at '{path}'. Set "
            f"EKF_CONFIG_PATH at the top of this script to the "
            f"correct location.")

    with open(path, "r") as f:
        text = f.read()

    values = {}
    for name in REQUIRED_CONSTANTS:
        m = re.search(rf"#define\s+{name}\s+([0-9.]+)", text)
        if not m:
            raise ValueError(
                f"Could not find '#define {name} <value>' in "
                f"{path}. This script requires all of "
                f"{REQUIRED_CONSTANTS} to be present and parseable "
                f"-- refusing to guess a value and silently drift "
                f"out of sync with your actual C config.")
        raw = m.group(1)
        values[name] = float(raw) if "." in raw else int(raw)

    print(f"Parsed from {path}:")
    for k, v in values.items():
        print(f"  {k} = {v}")

    return values


def geodetic_to_ecef(lat, lon, alt):
    a = 6378137.0
    e2 = 0.00669437999014
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    N = a / math.sqrt(1 - e2 * math.sin(lat_r) ** 2)
    x = (N + alt) * math.cos(lat_r) * math.cos(lon_r)
    y = (N + alt) * math.cos(lat_r) * math.sin(lon_r)
    z = (N * (1 - e2) + alt) * math.sin(lat_r)
    return x, y, z


def generate_true_trajectory():
    """Straight-line northward path, same as your existing
    generate_traj.py. Returns list of (x,y,z) tuples, length
    TRAJ_LEN. This is IDENTICAL across all modes and seeds --
    the true position never depends on noise or attack."""
    traj = []
    for t in range(TRAJ_LEN):
        lat = LAT0 + t * LAT_STEP_DEG
        traj.append(geodetic_to_ecef(lat, LON0, ALT0))
    return traj


def generate_measured_trajectory(true_traj, mode, cfg, seed):
    """
    Generates the measured (noisy, possibly spoofed) trajectory
    for one mode and one seed.

    Replicates the ORIGINAL on-device apply_spoof() semantics
    exactly:
      - Gaussian noise is added EVERY step, regardless of attack
        (GPS_SIGMA_H for x/y, GPS_SIGMA_V for z).
      - Attack logic applies only from ATTACK_START onward, and
        operates on the ALREADY-NOISED measurement (matching the
        original apply_spoof(rx, ry, rz, mode) call signature).
      - jump: adds JUMP_OFFSET_M to x and y (not z), every step
        from onset, no accumulation.
      - drift: accumulates DRIFT_RATE_MS per step from onset,
        capped at DRIFT_MAX_M, added to x only.
      - replay: substitutes the current noisy measurement with
        the noisy measurement from REPLAY_DELAY steps ago (from
        a history of NOISY measurements, not true positions --
        matching the original hist_push/hist_get on rx,ry,rz).
      - normal: passthrough, noise only, no attack.
    """
    rng = random.Random(seed)

    attack_start = int(cfg["ATTACK_START"])
    sigma_h = cfg["GPS_SIGMA_H"]
    sigma_v = cfg["GPS_SIGMA_V"]
    jump_offset = cfg["JUMP_OFFSET_M"]
    drift_rate = cfg["DRIFT_RATE_MS"]
    drift_max = cfg["DRIFT_MAX_M"]
    replay_delay = int(cfg["REPLAY_DELAY"])

    measured = []
    noisy_history = []   # for replay: stores noised (rx,ry,rz) per step
    drift_offset = 0.0

    for t in range(TRAJ_LEN):
        tx, ty, tz = true_traj[t]

        # Gaussian noise every step, same as original gaussian_noise()
        rx = tx + rng.gauss(0.0, sigma_h)
        ry = ty + rng.gauss(0.0, sigma_h)
        rz = tz + rng.gauss(0.0, sigma_v)

        noisy_history.append((rx, ry, rz))

        if t >= attack_start:
            if mode == "jump":
                mx, my, mz = rx + jump_offset, ry + jump_offset, rz
            elif mode == "drift":
                drift_offset += drift_rate
                if drift_offset > drift_max:
                    drift_offset = drift_max
                mx, my, mz = rx + drift_offset, ry, rz
            elif mode == "replay":
                idx = len(noisy_history) - 1 - replay_delay
                if idx >= 0:
                    mx, my, mz = noisy_history[idx]
                else:
                    mx, my, mz = rx, ry, rz
            else:  # normal or unrecognized -> passthrough
                mx, my, mz = rx, ry, rz
        else:
            mx, my, mz = rx, ry, rz

        measured.append((mx, my, mz))

    return measured


def write_header(mode, seed, true_traj, measured_traj, out_path):
    guard = f"TRAJECTORY_{mode.upper()}_SEED{seed}_H_"
    lines = []
    lines.append(f"/* Auto-generated by generate_trajectories.py")
    lines.append(f" * mode={mode}  seed={seed}  TRAJ_LEN={TRAJ_LEN}")
    lines.append(f" * true trajectory is identical across all modes/seeds")
    lines.append(f" * measured trajectory includes noise"
                 f"{' + ' + mode + ' attack from ATTACK_START' if mode != 'normal' else ''}")
    lines.append(f" */")
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    lines.append(f"#define TRAJ_LEN {TRAJ_LEN}")
    lines.append("")
    lines.append(f"static const double traj_true[TRAJ_LEN][3] = {{")
    for (x, y, z) in true_traj:
        lines.append(f"    {{{x:.4f}, {y:.4f}, {z:.4f}}},")
    lines.append("};")
    lines.append("")
    lines.append(f"static const double traj_measured[TRAJ_LEN][3] = {{")
    for (x, y, z) in measured_traj:
        lines.append(f"    {{{x:.4f}, {y:.4f}, {z:.4f}}},")
    lines.append("};")
    lines.append("")
    lines.append(f"#endif /* {guard} */")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    cfg = parse_ekf_config(EKF_CONFIG_PATH)
    true_traj = generate_true_trajectory()

    print(f"\nGenerating {len(MODES)} modes x {N_SEEDS} seeds "
          f"= {len(MODES) * N_SEEDS} header files...\n")

    for mode in MODES:
        for seed in range(1, N_SEEDS + 1):
            measured_traj = generate_measured_trajectory(
                true_traj, mode, cfg, seed)
            out_name = f"trajectory_{mode}_seed{seed}.h"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            write_header(mode, seed, true_traj, measured_traj, out_path)
            print(f"  Wrote {out_path}")

    print(f"\nDone. To use: copy the desired "
          f"trajectory_<mode>_seed<N>.h to your board src/ folder "
          f"as trajectory.h before building that run.")


if __name__ == "__main__":
    main()