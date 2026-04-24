"""
Generate synthetic .mat files that mimic the Camargo 2021 dataset structure.
Used for end-to-end pipeline testing before the real dataset is downloaded.

Creates realistic-ish gait signals: sinusoidal ankle angle with IMU-like
noise patterns for 22 synthetic subjects across multiple terrain conditions.
"""

import os
import numpy as np
from scipy.io import savemat

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMARGO_DIR = os.path.join(PROJECT_ROOT, "data", "camargo2021")

SAMPLE_RATE = 200  # Hz
TRIAL_DURATION = 10  # seconds per trial
N_SAMPLES = SAMPLE_RATE * TRIAL_DURATION

SUBJECTS = [f"AB{i:02d}" for i in range(1, 23)]
CONDITIONS = [
    "levelground_walking_01",
    "levelground_walking_02",
    "ramp_ascent_01",
    "ramp_descent_01",
    "stair_ascent_01",
    "stair_descent_01",
    "treadmill_walking_01",
]


def generate_gait_signals(n_samples, condition, subject_seed):
    """Generate synthetic IMU + ankle angle signals for one trial."""
    rng = np.random.RandomState(subject_seed)
    t = np.arange(n_samples) / SAMPLE_RATE

    stride_freq = 0.9 + rng.rand() * 0.4  # 0.9-1.3 Hz
    phase_offset = rng.rand() * 2 * np.pi

    # Ankle angle: dorsiflexion/plantarflexion (~-20 to +15 deg)
    ankle_base = -5.0
    ankle_amp = 15.0
    if "ramp_ascent" in condition:
        ankle_base = 0.0
        ankle_amp = 12.0
    elif "ramp_descent" in condition:
        ankle_base = -8.0
        ankle_amp = 18.0
    elif "stair" in condition:
        ankle_amp = 20.0

    # All signals share the same trial-level phase offset so that the
    # IMU→ankle relationship is *learnable*. Without this, every trial would
    # have a randomly-shifted ankle relative to the IMU, making the mapping
    # ill-defined.
    phi = 2 * np.pi * stride_freq * t + phase_offset

    ankle_angle = ankle_base + ankle_amp * np.sin(phi)
    ankle_angle += rng.randn(n_samples) * 0.5

    # Shank IMU: correlated with ankle motion
    shank_accel_x = 0.2 * np.sin(phi) + rng.randn(n_samples) * 0.1
    shank_accel_y = 9.81 + 0.5 * np.cos(phi) + rng.randn(n_samples) * 0.15
    shank_accel_z = 0.3 * np.sin(phi + 0.5) + rng.randn(n_samples) * 0.1
    shank_gyro_x = 2.0 * np.cos(phi) + rng.randn(n_samples) * 0.2
    shank_gyro_y = 0.5 * np.sin(phi + 1.0) + rng.randn(n_samples) * 0.15
    shank_gyro_z = 1.0 * np.cos(phi + 0.3) + rng.randn(n_samples) * 0.2

    # Thigh IMU: leads the shank by ~100ms phase
    lead = 0.6
    thigh_accel_x = 0.15 * np.sin(phi + lead) + rng.randn(n_samples) * 0.1
    thigh_accel_y = 9.81 + 0.3 * np.cos(phi + lead) + rng.randn(n_samples) * 0.12
    thigh_accel_z = 0.2 * np.sin(phi + lead + 0.5) + rng.randn(n_samples) * 0.1
    thigh_gyro_x = 1.5 * np.cos(phi + lead) + rng.randn(n_samples) * 0.2
    thigh_gyro_y = 0.4 * np.sin(phi + lead + 1.0) + rng.randn(n_samples) * 0.15
    thigh_gyro_z = 0.8 * np.cos(phi + lead + 0.3) + rng.randn(n_samples) * 0.2

    return {
        "shank_Accel_X": shank_accel_x,
        "shank_Accel_Y": shank_accel_y,
        "shank_Accel_Z": shank_accel_z,
        "shank_Gyro_X": shank_gyro_x,
        "shank_Gyro_Y": shank_gyro_y,
        "shank_Gyro_Z": shank_gyro_z,
        "thigh_Accel_X": thigh_accel_x,
        "thigh_Accel_Y": thigh_accel_y,
        "thigh_Accel_Z": thigh_accel_z,
        "thigh_Gyro_X": thigh_gyro_x,
        "thigh_Gyro_Y": thigh_gyro_y,
        "thigh_Gyro_Z": thigh_gyro_z,
        "ankle_angle_r": ankle_angle,
        "time": t,
    }


def main():
    print(f"Generating synthetic Camargo-format data in {CAMARGO_DIR}")
    total_files = 0

    for i, subj in enumerate(SUBJECTS):
        subj_dir = os.path.join(CAMARGO_DIR, subj)
        os.makedirs(subj_dir, exist_ok=True)

        for j, cond in enumerate(CONDITIONS):
            seed = i * 1000 + j
            signals = generate_gait_signals(N_SAMPLES, cond, seed)
            filepath = os.path.join(subj_dir, f"{cond}.mat")
            savemat(filepath, signals)
            total_files += 1

        print(f"  {subj}: {len(CONDITIONS)} trials")

    print(f"\nDone. Created {total_files} .mat files across {len(SUBJECTS)} subjects.")
    print(f"Each trial: {TRIAL_DURATION}s at {SAMPLE_RATE} Hz = {N_SAMPLES} samples")


if __name__ == "__main__":
    main()
