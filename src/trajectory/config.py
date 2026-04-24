"""
Centralized configuration for the trajectory correction pipeline.

All hyperparameters, paths, and constants are defined here so that
every module in src/trajectory/ shares a single source of truth.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMARGO_DIR = os.path.join(PROJECT_ROOT, "data", "camargo2021")
REZNICK_DIR = os.path.join(PROJECT_ROOT, "data", "reznick2021")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "shadow_limb_lstm.pt")  # legacy default
REFERENCE_GAIT_PATH = os.path.join(MODEL_DIR, "reference_gait.npy")


def model_path_for_mode(mode: str) -> str:
    """Checkpoint filename derived from prediction mode.

    Keeping LSTM / Transformer / phase checkpoints in separate files lets us
    train them all, compare them, and avoid clobbering each other's weights.
    """
    return {
        "single": os.path.join(MODEL_DIR, "shadow_limb_lstm.pt"),
        "trajectory": os.path.join(MODEL_DIR, "shadow_limb_encdec.pt"),
        "transformer": os.path.join(MODEL_DIR, "shadow_limb_transformer.pt"),
        "phase_lstm": os.path.join(MODEL_DIR, "shadow_limb_phase_lstm.pt"),
        "phase_transformer": os.path.join(MODEL_DIR, "shadow_limb_phase_transformer.pt"),
    }.get(mode, os.path.join(MODEL_DIR, f"shadow_limb_{mode}.pt"))

# ---------------------------------------------------------------------------
# Dataset / Signal
# ---------------------------------------------------------------------------
SAMPLE_RATE_HZ = 200           # Camargo IMU sampling rate
WINDOW_MS = 200                # Input window duration in milliseconds
HORIZON_MS = 50                # Prediction horizon in milliseconds
WINDOW_SAMPLES = int(SAMPLE_RATE_HZ * WINDOW_MS / 1000)   # 40 samples
HORIZON_SAMPLES = int(SAMPLE_RATE_HZ * HORIZON_MS / 1000)  # 10 samples
STEP_SAMPLES = 5               # Sliding-window step (stride)

# IMU channel names expected in the dataset.
# Camargo convention: <segment>_<sensor>_<axis>
# We use shank IMU as the primary input (closest to the missing ankle).
SHANK_IMU_COLS = [
    "shank_Accel_X", "shank_Accel_Y", "shank_Accel_Z",
    "shank_Gyro_X",  "shank_Gyro_Y",  "shank_Gyro_Z",
]
THIGH_IMU_COLS = [
    "thigh_Accel_X", "thigh_Accel_Y", "thigh_Accel_Z",
    "thigh_Gyro_X",  "thigh_Gyro_Y",  "thigh_Gyro_Z",
]
# Toggle whether to include thigh IMU as additional input channels
USE_THIGH_IMU = True
INPUT_COLS = SHANK_IMU_COLS + (THIGH_IMU_COLS if USE_THIGH_IMU else [])
NUM_INPUT_CHANNELS = len(INPUT_COLS)  # 6 (shank only) or 12 (shank + thigh)

# Target: sagittal-plane ankle angle from motion capture
TARGET_COL = "ankle_angle_r"  # right ankle dorsi/plantarflexion (degrees)

# ---------------------------------------------------------------------------
# Subject splits (Camargo: AB01 .. AB22, 22 subjects)
# ---------------------------------------------------------------------------
ALL_SUBJECTS = [f"AB{i:02d}" for i in range(1, 23)]
TRAIN_SUBJECTS = ALL_SUBJECTS[:18]   # 18 for training
VAL_SUBJECTS = ALL_SUBJECTS[18:20]   # 2 for validation
TEST_SUBJECTS = ALL_SUBJECTS[20:22]  # 2 for testing

# ---------------------------------------------------------------------------
# Model — LSTM
# ---------------------------------------------------------------------------
HIDDEN_DIM = 128
NUM_LSTM_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = True

# ---------------------------------------------------------------------------
# Model — Transformer baseline
# ---------------------------------------------------------------------------
# These are deliberately sized so that the Transformer has a comparable
# parameter count to the BiLSTM (~200k–350k) and runs in <20ms per window
# on CPU for a 40-sample input.
TRANSFORMER_D_MODEL = 128
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 3
TRANSFORMER_DIM_FEEDFORWARD = 256
TRANSFORMER_DROPOUT = 0.2

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
PATIENCE = 8                   # Early-stopping patience (epochs)
LR_FACTOR = 0.5               # ReduceLROnPlateau factor
LR_PATIENCE = 3               # ReduceLROnPlateau patience
GRAD_CLIP = 1.0               # Gradient clipping max norm

# ---------------------------------------------------------------------------
# Phase / Reference-Gait Tracking
# ---------------------------------------------------------------------------
# Number of bins used in the reference-gait lookup table θ_ref(φ).
# 100 ≈ 1% resolution of the gait cycle, matching the paper's convention.
N_PHASE_BINS = 100
# Minimum samples between two detected gait events. At 200 Hz a typical
# human stride is 133–250 samples (0.67–1.25 s, i.e. 0.8–1.5 Hz stride
# frequency), so 120 samples (~0.6 s) is a safe lower bound that rejects
# noise-driven half-cycle doubles without missing fast walking.
MIN_CYCLE_SAMPLES = 120
# Smoothing window applied before minima detection (odd, in samples).
# 11 samples ≈ 55 ms at 200 Hz — suppresses sub-cycle jitter without
# washing out the real minimum.
PHASE_SMOOTH_WINDOW = 11

# ---------------------------------------------------------------------------
# Derived
# ---------------------------------------------------------------------------
PREDICTION_MODE = "single"     # "single" | "trajectory" | "transformer" | "phase_lstm" | "phase_transformer"
EFFECTIVE_HORIZON = 1 if PREDICTION_MODE == "single" else HORIZON_SAMPLES

PHASE_MODES = {"phase_lstm", "phase_transformer"}


def is_phase_mode(mode: str) -> bool:
    return mode in PHASE_MODES
