"""
Reznick 2021 dataset loader for the phase-prediction pipeline.

The real Reznick dataset ships as a different on-disk format than Camargo
(per-subject .mat files with nested struct layout + HDF5 variants). To
keep the rest of the code unaware of that, this module exposes a thin
adapter that:

  * discovers available Reznick trials under ``data/reznick2021/``,
  * returns them in the same flat-dict schema that Camargo uses
    (`shank_Accel_*`, `thigh_Gyro_*`, `ankle_angle_r`, `time`),
  * surfaces extra metadata (speed, incline) that callers can use for
    stratified evaluation.

When `data/reznick2021/` is populated with the synthetic stand-in from
``scripts/generate_synthetic_reznick.py`` or with the real dataset laid
out as ``data/reznick2021/<subject>/<trial>.mat``, this loader Just Works.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List

import numpy as np

from . import config
from .dataset import load_mat_trial, extract_windows, extract_phase_windows


def available_subjects(reznick_dir: str = config.REZNICK_DIR) -> List[str]:
    if not os.path.isdir(reznick_dir):
        return []
    subjects = []
    for name in sorted(os.listdir(reznick_dir)):
        p = os.path.join(reznick_dir, name)
        if os.path.isdir(p) and not name.startswith("."):
            subjects.append(name)
    return subjects


def discover_reznick_trials(
    reznick_dir: str = config.REZNICK_DIR,
    subjects: List[str] | None = None,
) -> List[dict]:
    """
    Discover Reznick trials in the standard layout ``<reznick_dir>/<subj>/*.mat``.

    Returns list of dicts with keys: subject, path, condition, speed_mps, incline_deg.
    Speed / incline are filled in as NaN if the file doesn't carry them.
    """
    if not os.path.isdir(reznick_dir):
        return []

    subjects = subjects or available_subjects(reznick_dir)
    trials: List[dict] = []
    for subj in subjects:
        subj_dir = os.path.join(reznick_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        for path in sorted(glob.glob(os.path.join(subj_dir, "**", "*.mat"), recursive=True)):
            condition = os.path.splitext(os.path.basename(path))[0]
            trials.append({
                "subject": subj,
                "path": path,
                "condition": condition,
                "dataset": "reznick",
            })
    return trials


def load_reznick_trial(path: str) -> Dict[str, np.ndarray]:
    """
    Load a single Reznick trial. Tries the Camargo-compatible flat schema
    first; if the file carries scalar metadata like `_speed_mps` or
    `_incline_deg`, those are preserved for per-condition evaluation.
    """
    return load_mat_trial(path)


def trial_metadata(trial_data: Dict[str, np.ndarray]) -> dict:
    """Extract scalar metadata (speed, incline) from a loaded trial."""
    def _scalar(x):
        if x is None:
            return float("nan")
        arr = np.asarray(x)
        if arr.size == 0:
            return float("nan")
        return float(arr.flat[0])

    return {
        "speed_mps": _scalar(trial_data.get("_speed_mps")),
        "incline_deg": _scalar(trial_data.get("_incline_deg")),
    }


# ── Convenience: extract windows from Reznick trials ──────────────────────

def reznick_phase_windows(
    subjects: List[str] | None = None,
    reznick_dir: str = config.REZNICK_DIR,
    window_samples: int = config.WINDOW_SAMPLES,
    step: int = config.STEP_SAMPLES,
    input_cols: List[str] = config.INPUT_COLS,
    target_col: str = config.TARGET_COL,
    verbose: bool = True,
):
    """
    Aggregate phase-mode windows from the Reznick dataset.

    Returns (X, y_sincos, y_angle, metadata) with the same shapes as
    ``dataset.build_subject_phase_data``. Raises RuntimeError if no Reznick
    data is available — callers can catch that and fall back to Camargo.
    """
    trials = discover_reznick_trials(reznick_dir, subjects)
    if not trials:
        raise RuntimeError(
            f"No Reznick trials found under {reznick_dir!r}. "
            "Either download the real Reznick 2021 dataset or run "
            "scripts/generate_synthetic_reznick.py."
        )

    all_X, all_sc, all_ang, meta = [], [], [], []
    if verbose:
        print(f"[Reznick] Loading {len(trials)} trials...")

    for trial_info in trials:
        try:
            trial_data = load_reznick_trial(trial_info["path"])
        except Exception as e:
            if verbose:
                print(f"  [SKIP] {trial_info['path']}: {e}")
            continue

        X, sc, ang = extract_phase_windows(
            trial_data, input_cols, target_col, window_samples, step
        )
        if len(X) == 0:
            continue

        # Merge file-level metadata (speed, incline) into the per-window meta.
        scalar_meta = trial_metadata(trial_data)
        enriched = {**trial_info, **scalar_meta}
        all_X.append(X)
        all_sc.append(sc)
        all_ang.append(ang)
        meta.extend([enriched] * len(X))

    if not all_X:
        raise RuntimeError("Reznick files found, but no phase windows extracted.")
    return (
        np.concatenate(all_X),
        np.concatenate(all_sc),
        np.concatenate(all_ang),
        meta,
    )


def reznick_angle_windows(
    subjects: List[str] | None = None,
    reznick_dir: str = config.REZNICK_DIR,
    window_samples: int = config.WINDOW_SAMPLES,
    step: int = config.STEP_SAMPLES,
    input_cols: List[str] = config.INPUT_COLS,
    target_col: str = config.TARGET_COL,
    verbose: bool = True,
):
    """Same as `reznick_phase_windows` but emits direct angle targets."""
    trials = discover_reznick_trials(reznick_dir, subjects)
    if not trials:
        raise RuntimeError(f"No Reznick trials found under {reznick_dir!r}.")

    all_X, all_y, meta = [], [], []
    if verbose:
        print(f"[Reznick] Loading {len(trials)} trials (angle mode)...")
    for trial_info in trials:
        try:
            trial_data = load_reznick_trial(trial_info["path"])
        except Exception as e:
            if verbose:
                print(f"  [SKIP] {trial_info['path']}: {e}")
            continue
        X, y = extract_windows(
            trial_data, input_cols, target_col, window_samples, 1, step
        )
        if len(X) == 0:
            continue
        scalar_meta = trial_metadata(trial_data)
        all_X.append(X)
        all_y.append(y)
        meta.extend([{**trial_info, **scalar_meta}] * len(X))

    if not all_X:
        raise RuntimeError("Reznick files found, but no angle windows extracted.")
    return np.concatenate(all_X), np.concatenate(all_y), meta
