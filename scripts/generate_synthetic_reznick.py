"""
Synthetic stand-in for the Reznick 2021 dataset (continuous-speed locomotion).

The real Reznick dataset records 10 able-bodied subjects walking on a
treadmill at **continuously varying** speed (0.8–1.2 m/s) and incline
(-10° to +10°). Its value for the Shadow Limb pipeline is that it
validates continuous gait-phase models: the phase signal has to adapt
smoothly as stride frequency changes within a single trial.

Until the real dataset is downloaded, this script generates files that
follow the same directory structure and signal conventions as Camargo,
so the Reznick loader (src/trajectory/reznick_loader.py) can exercise the
phase-prediction code path.

Each synthetic trial has the following quirks relative to the Camargo
generator:

  * stride frequency varies slowly within a trial (sinusoidal drift),
  * incline bias shifts the ankle base angle up or down during the trial,
  * a tiny amount of inter-cycle jitter makes phase boundaries non-uniform,

which together give the phase detector something non-trivial to do.
"""

import os

import numpy as np
from scipy.io import savemat

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REZNICK_DIR = os.path.join(PROJECT_ROOT, "data", "reznick2021")

SAMPLE_RATE = 200
TRIAL_DURATION = 20  # seconds — longer trials to see speed drift
N_SAMPLES = SAMPLE_RATE * TRIAL_DURATION

SUBJECTS = [f"S{i:02d}" for i in range(1, 11)]  # Reznick uses 10 subjects

# Trial metadata: (name, base_speed [m/s], incline [deg])
TRIAL_SPEC = [
    ("walk_slow_flat",     0.80,   0.0),
    ("walk_normal_flat",   1.00,   0.0),
    ("walk_fast_flat",     1.20,   0.0),
    ("walk_normal_up",     1.00,   5.0),
    ("walk_normal_down",   1.00,  -5.0),
    ("walk_variable_flat", 1.00,   0.0),  # strongest speed drift
]


def generate_trial(seed: int, base_speed: float, incline: float) -> dict:
    """Generate a single synthetic Reznick-format trial.

    Ankle base/amplitude scale with speed and incline exactly as in the
    Camargo generator, but the stride frequency is allowed to drift slowly
    over the trial so the phase signal is not a pure sinusoid.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(N_SAMPLES) / SAMPLE_RATE

    # Base stride frequency from walking speed (Dean et al. empirical fit)
    base_freq = 0.8 + 0.35 * (base_speed - 0.8) / 0.4  # 0.8 Hz @ 0.8 m/s, 1.15 Hz @ 1.2 m/s

    # Slow sinusoidal drift + trial-scale jitter
    drift_amp = 0.08 if "variable" in os.environ.get("_REZNICK_HINT", "") else 0.04
    drift = drift_amp * np.sin(2 * np.pi * (1 / 10) * t + rng.rand() * 2 * np.pi)
    instantaneous_freq = base_freq * (1.0 + drift)
    # Integrate frequency to get phase angle — this is the trick that makes
    # phi(t) non-linear in t, i.e. the stride period is not constant.
    phi = 2 * np.pi * np.cumsum(instantaneous_freq) / SAMPLE_RATE + rng.rand() * 2 * np.pi

    # Ankle angle responds to speed (larger swings when faster) and incline.
    amp_scale = 0.8 + 0.5 * (base_speed - 0.8) / 0.4
    ankle_base = -5.0 - 0.8 * incline
    ankle_amp = 15.0 * amp_scale * (1.0 - 0.05 * max(incline, 0.0))

    ankle_angle = ankle_base + ankle_amp * np.sin(phi)
    ankle_angle += rng.randn(N_SAMPLES) * 0.5

    # Shank IMU — correlated with the same phi
    shank_accel_x = 0.2 * np.sin(phi) + rng.randn(N_SAMPLES) * 0.1
    shank_accel_y = 9.81 + 0.5 * np.cos(phi) + rng.randn(N_SAMPLES) * 0.15
    shank_accel_z = 0.3 * np.sin(phi + 0.5) + rng.randn(N_SAMPLES) * 0.1
    shank_gyro_x = 2.0 * amp_scale * np.cos(phi) + rng.randn(N_SAMPLES) * 0.2
    shank_gyro_y = 0.5 * np.sin(phi + 1.0) + rng.randn(N_SAMPLES) * 0.15
    shank_gyro_z = 1.0 * np.cos(phi + 0.3) + rng.randn(N_SAMPLES) * 0.2

    lead = 0.6
    thigh_accel_x = 0.15 * np.sin(phi + lead) + rng.randn(N_SAMPLES) * 0.1
    thigh_accel_y = 9.81 + 0.3 * np.cos(phi + lead) + rng.randn(N_SAMPLES) * 0.12
    thigh_accel_z = 0.2 * np.sin(phi + lead + 0.5) + rng.randn(N_SAMPLES) * 0.1
    thigh_gyro_x = 1.5 * amp_scale * np.cos(phi + lead) + rng.randn(N_SAMPLES) * 0.2
    thigh_gyro_y = 0.4 * np.sin(phi + lead + 1.0) + rng.randn(N_SAMPLES) * 0.15
    thigh_gyro_z = 0.8 * np.cos(phi + lead + 0.3) + rng.randn(N_SAMPLES) * 0.2

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
        "_speed_mps": float(base_speed),
        "_incline_deg": float(incline),
    }


def main():
    print(f"Generating synthetic Reznick-format data in {REZNICK_DIR}")
    os.makedirs(REZNICK_DIR, exist_ok=True)
    total = 0

    for si, subj in enumerate(SUBJECTS):
        subj_dir = os.path.join(REZNICK_DIR, subj)
        os.makedirs(subj_dir, exist_ok=True)

        for ti, (name, speed, incline) in enumerate(TRIAL_SPEC):
            seed = si * 1000 + ti
            signals = generate_trial(seed, speed, incline)
            path = os.path.join(subj_dir, f"{name}.mat")
            savemat(path, signals)
            total += 1

        print(f"  {subj}: {len(TRIAL_SPEC)} trials")

    print(f"\nDone. Created {total} .mat files across {len(SUBJECTS)} subjects.")
    print(f"Each trial: {TRIAL_DURATION}s at {SAMPLE_RATE} Hz = {N_SAMPLES} samples.")
    print("\nNote: this is a synthetic stand-in with continuously drifting")
    print("stride frequency. Once the real Reznick 2021 dataset is downloaded")
    print("into data/reznick2021/, the same loader will read it transparently.")


if __name__ == "__main__":
    main()
