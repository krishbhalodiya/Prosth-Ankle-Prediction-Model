"""
Data pipeline for the Camargo 2021 benchmark dataset.

Loads per-subject .mat trial files, extracts IMU windows and corresponding
ankle angle targets, applies z-score normalization, and provides PyTorch
Dataset / DataLoader utilities with subject-independent train/val/test splits.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from scipy.io import loadmat
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from . import config
from . import phase as phase_utils


# ── .mat File Loading ────────────────────────────────────────────────────────

def _flatten_mat(mat_dict: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flatten a nested .mat structure into a flat dict of arrays."""
    items = {}
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, np.ndarray) and v.ndim == 0:
            inner = v.item()
            if hasattr(inner, "dtype") and inner.dtype.names:
                for field in inner.dtype.names:
                    child = inner[field]
                    if isinstance(child, np.ndarray) and child.ndim == 0:
                        child = child.item()
                    items.update(
                        _flatten_mat({field: child}, parent_key=new_key, sep=sep)
                    )
            else:
                items[new_key] = inner
        elif isinstance(v, np.ndarray):
            if v.dtype.names:
                for field in v.dtype.names:
                    items.update(
                        _flatten_mat({field: v[field]}, parent_key=new_key, sep=sep)
                    )
            else:
                items[new_key] = v
        else:
            items[new_key] = v
    return items


def load_mat_trial(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load a single Camargo .mat file and return a flat dict of signal arrays.

    The Camargo dataset uses nested MATLAB structures. This function
    flattens them so that signal access becomes e.g. data["shank_Accel_X"].
    """
    raw = loadmat(filepath, squeeze_me=True, struct_as_record=True)
    flat = _flatten_mat(raw)

    result: Dict[str, np.ndarray] = {}
    for key, val in flat.items():
        if isinstance(val, np.ndarray) and val.ndim >= 1 and val.dtype.kind == "f":
            short_key = key.rsplit(".", 1)[-1] if "." in key else key
            if short_key not in result:
                result[short_key] = val.flatten()
            else:
                result[key] = val.flatten()
    return result


def discover_trials(camargo_dir: str, subjects: List[str]) -> List[dict]:
    """
    Discover all .mat trial files for the given subjects.

    Returns a list of dicts: {"subject": str, "path": str, "condition": str}
    """
    trials = []
    for subj in subjects:
        subj_dir = os.path.join(camargo_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        mat_paths = sorted(
            glob.glob(os.path.join(subj_dir, "**", "*.mat"), recursive=True)
        )
        for mp in mat_paths:
            condition = os.path.splitext(os.path.basename(mp))[0]
            trials.append({
                "subject": subj,
                "path": mp,
                "condition": condition,
            })
    return trials


# ── Windowing ────────────────────────────────────────────────────────────────

def extract_windows(
    trial_data: Dict[str, np.ndarray],
    input_cols: List[str],
    target_col: str,
    window_samples: int,
    horizon_samples: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment a trial into overlapping (input_window, target) pairs.

    Parameters
    ----------
    trial_data : flat dict of 1-D numpy arrays (from load_mat_trial)
    input_cols : list of IMU channel names to stack as input features
    target_col : name of the ankle angle signal
    window_samples : number of samples in each input window
    horizon_samples : prediction horizon (1 for single-step)
    step : stride between consecutive windows

    Returns
    -------
    X : ndarray of shape (N, window_samples, n_channels)
    y : ndarray of shape (N,) for single-step or (N, horizon) for trajectory
    """
    available = set(trial_data.keys())
    missing_inputs = [c for c in input_cols if c not in available]
    if missing_inputs or target_col not in available:
        return np.empty((0, window_samples, len(input_cols))), np.empty(0)

    signals = np.column_stack([trial_data[c] for c in input_cols])
    target = trial_data[target_col]

    n = min(len(signals), len(target))
    signals = signals[:n]
    target = target[:n]

    end = n - horizon_samples
    X_list, y_list = [], []
    for start in range(0, end - window_samples + 1, step):
        X_list.append(signals[start : start + window_samples])
        if horizon_samples <= 1:
            y_list.append(target[start + window_samples])
        else:
            y_list.append(target[start + window_samples : start + window_samples + horizon_samples])

    if not X_list:
        return np.empty((0, window_samples, len(input_cols))), np.empty(0)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def extract_phase_windows(
    trial_data: Dict[str, np.ndarray],
    input_cols: List[str],
    target_col: str,
    window_samples: int,
    step: int,
    min_cycle_samples: int = config.MIN_CYCLE_SAMPLES,
    smooth_window: int = config.PHASE_SMOOTH_WINDOW,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Like `extract_windows` but emits phase-encoded targets.

    For each window ending at sample t, the target is (sin 2πφ_t, cos 2πφ_t)
    where φ_t is the gait phase derived from the ankle-angle trajectory.
    Windows whose label sample falls outside a detected gait cycle (NaN
    phase) are skipped, so the returned arrays are already "clean".

    Returns
    -------
    X : (N, window_samples, n_channels)
    y_sincos : (N, 2) phase target
    y_angle  : (N,) raw ankle angle (kept for reference-table building and
                    for reconstructed-angle RMSE at eval time).
    """
    available = set(trial_data.keys())
    if any(c not in available for c in input_cols) or target_col not in available:
        return (
            np.empty((0, window_samples, len(input_cols)), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    signals = np.column_stack([trial_data[c] for c in input_cols]).astype(np.float32)
    angle = np.asarray(trial_data[target_col], dtype=np.float32)
    n = min(len(signals), len(angle))
    signals, angle = signals[:n], angle[:n]

    phase, _events = phase_utils.compute_phase_from_angle(
        angle, min_cycle_samples=min_cycle_samples, smooth_window=smooth_window
    )
    sincos = phase_utils.phase_to_sincos(phase)  # NaN rows where phase is NaN

    X_list, y_sc_list, y_a_list = [], [], []
    for start in range(0, n - window_samples, step):
        label_idx = start + window_samples
        sc = sincos[label_idx]
        if not np.all(np.isfinite(sc)):
            continue
        X_list.append(signals[start : start + window_samples])
        y_sc_list.append(sc)
        y_a_list.append(angle[label_idx])

    if not X_list:
        return (
            np.empty((0, window_samples, len(input_cols)), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_sc_list, dtype=np.float32),
        np.array(y_a_list, dtype=np.float32),
    )


# ── Normalization ────────────────────────────────────────────────────────────

class Normalizer:
    """Per-channel z-score normalizer fitted on training data."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Normalizer":
        """Compute channel-wise mean/std from X of shape (N, T, C)."""
        flat = X.reshape(-1, X.shape[-1])
        self.mean = flat.mean(axis=0).astype(np.float32)
        self.std = flat.std(axis=0).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean, "std": self.std}, f)

    def load(self, path: str) -> "Normalizer":
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.mean = d["mean"]
        self.std = d["std"]
        return self


# ── PyTorch Dataset ──────────────────────────────────────────────────────────

class GaitWindowDataset(Dataset):
    """
    PyTorch Dataset wrapping pre-extracted (X, y) numpy arrays.

    For phase mode, pass the (sin, cos) phase target as `y` and the raw
    ankle angle as `y_angle`. The DataLoader will emit 3-tuples (X, y,
    y_angle); for non-phase mode it emits 2-tuples (X, y) as before.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: Optional[List[dict]] = None,
        y_angle: Optional[np.ndarray] = None,
    ):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.y_angle = (
            torch.from_numpy(y_angle) if y_angle is not None else None
        )
        self.metadata = metadata

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y_angle is not None:
            return self.X[idx], self.y[idx], self.y_angle[idx]
        return self.X[idx], self.y[idx]


# ── High-Level Pipeline ─────────────────────────────────────────────────────

def build_subject_data(
    subjects: List[str],
    camargo_dir: str = config.CAMARGO_DIR,
    input_cols: List[str] = config.INPUT_COLS,
    target_col: str = config.TARGET_COL,
    window_samples: int = config.WINDOW_SAMPLES,
    horizon_samples: int = config.EFFECTIVE_HORIZON,
    step: int = config.STEP_SAMPLES,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Load trials for a list of subjects and extract all angle-prediction windows.

    Returns (X, y, metadata) where metadata tracks subject/condition per window.
    """
    all_X, all_y, all_meta = [], [], []
    trials = discover_trials(camargo_dir, subjects)

    if verbose:
        print(f"Loading {len(trials)} trials from {len(subjects)} subjects...")

    for trial_info in trials:
        try:
            trial_data = load_mat_trial(trial_info["path"])
        except Exception as e:
            if verbose:
                print(f"  [SKIP] {trial_info['path']}: {e}")
            continue

        X, y = extract_windows(trial_data, input_cols, target_col, window_samples, horizon_samples, step)
        if len(X) == 0:
            continue

        all_X.append(X)
        all_y.append(y)
        all_meta.extend([trial_info] * len(X))

        if verbose:
            print(f"  {trial_info['subject']}/{trial_info['condition']}: {len(X)} windows")

    if not all_X:
        raise RuntimeError(
            "No valid windows extracted. Check that the dataset is downloaded "
            "and that column names in config.py match the .mat file contents. "
            "Run notebooks/01_dataset_exploration.ipynb to inspect."
        )

    return np.concatenate(all_X), np.concatenate(all_y), all_meta


def build_subject_phase_data(
    subjects: List[str],
    camargo_dir: str = config.CAMARGO_DIR,
    input_cols: List[str] = config.INPUT_COLS,
    target_col: str = config.TARGET_COL,
    window_samples: int = config.WINDOW_SAMPLES,
    step: int = config.STEP_SAMPLES,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    """
    Phase-mode equivalent of `build_subject_data`.

    Returns (X, y_sincos, y_angle, metadata) where:
      X         : (N, window_samples, n_channels)
      y_sincos  : (N, 2)
      y_angle   : (N,)
    """
    all_X, all_sc, all_ang, all_meta = [], [], [], []
    trials = discover_trials(camargo_dir, subjects)

    if verbose:
        print(f"Loading {len(trials)} trials from {len(subjects)} subjects (phase mode)...")

    for trial_info in trials:
        try:
            trial_data = load_mat_trial(trial_info["path"])
        except Exception as e:
            if verbose:
                print(f"  [SKIP] {trial_info['path']}: {e}")
            continue

        X, y_sc, y_a = extract_phase_windows(
            trial_data, input_cols, target_col, window_samples, step
        )
        if len(X) == 0:
            if verbose:
                print(f"  [NO CYCLES] {trial_info['subject']}/{trial_info['condition']}")
            continue

        all_X.append(X)
        all_sc.append(y_sc)
        all_ang.append(y_a)
        all_meta.extend([trial_info] * len(X))

        if verbose:
            print(f"  {trial_info['subject']}/{trial_info['condition']}: {len(X)} phase windows")

    if not all_X:
        raise RuntimeError(
            "No valid phase windows extracted. Either the dataset is missing "
            "or gait-event detection failed for every trial — try decreasing "
            "config.MIN_CYCLE_SAMPLES."
        )

    return (
        np.concatenate(all_X),
        np.concatenate(all_sc),
        np.concatenate(all_ang),
        all_meta,
    )


def build_dataloaders(
    camargo_dir: str = config.CAMARGO_DIR,
    batch_size: int = config.BATCH_SIZE,
    verbose: bool = True,
    mode: str = config.PREDICTION_MODE,
) -> Tuple[DataLoader, DataLoader, DataLoader, Normalizer]:
    """
    Full pipeline: load data, split by subject, normalize, return DataLoaders.

    For `mode` in PHASE_MODES the loaders emit (X, y_sincos, y_angle) tuples;
    otherwise they emit (X, y) for direct angle regression.

    Returns (train_loader, val_loader, test_loader, normalizer).
    """
    phase_mode = config.is_phase_mode(mode)

    if phase_mode:
        X_train, y_train, y_ang_train, meta_train = build_subject_phase_data(
            config.TRAIN_SUBJECTS, camargo_dir, verbose=verbose
        )
        X_val, y_val, y_ang_val, _ = build_subject_phase_data(
            config.VAL_SUBJECTS, camargo_dir, verbose=verbose
        )
        X_test, y_test, y_ang_test, _ = build_subject_phase_data(
            config.TEST_SUBJECTS, camargo_dir, verbose=verbose
        )
    else:
        X_train, y_train, meta_train = build_subject_data(
            config.TRAIN_SUBJECTS, camargo_dir, verbose=verbose
        )
        X_val, y_val, _ = build_subject_data(
            config.VAL_SUBJECTS, camargo_dir, verbose=verbose
        )
        X_test, y_test, _ = build_subject_data(
            config.TEST_SUBJECTS, camargo_dir, verbose=verbose
        )
        y_ang_train = y_ang_val = y_ang_test = None

    normalizer = Normalizer()
    X_train = normalizer.fit_transform(X_train)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)

    if verbose:
        print(f"\nDataset sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"Input shape: {X_train.shape[1:]}, Target shape: {y_train.shape[1:]}")
        if phase_mode:
            print("Phase mode enabled — targets are (sin 2πφ, cos 2πφ).")
        print(f"Normalizer mean: {normalizer.mean}")
        print(f"Normalizer std:  {normalizer.std}")

    train_ds = GaitWindowDataset(X_train, y_train, meta_train, y_angle=y_ang_train)
    val_ds = GaitWindowDataset(X_val, y_val, y_angle=y_ang_val)
    test_ds = GaitWindowDataset(X_test, y_test, y_angle=y_ang_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, normalizer
