"""
Gait phase utilities for the Shadow Limb controller.

This module implements the "Reference Gait Tracking" machinery from §3.2 and
§5.3 of the Shadow Limb paper:

  1. `compute_phase_from_angle`  — turn an ankle-angle trajectory into a
     continuous phase signal φ(t) ∈ [0, 1) by detecting gait events and
     linearly interpolating between them.
  2. `build_reference_gait_table` — from a collection of (phase, angle)
     samples, produce the canonical reference trajectory θ_ref(φ) as an
     N-bin lookup table.
  3. `reference_lookup`         — map predicted phases to ankle-angle
     commands using the saved reference table.
  4. `phase_error_pct`          — circular distance metric (% of cycle).

The encoding of φ for the neural network is `(sin 2πφ, cos 2πφ)`, which is
continuous across the 1→0 wrap-around and can be predicted with standard
MSE regression.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np


TWO_PI = 2.0 * np.pi


# ── Phase encoding ──────────────────────────────────────────────────────────

def phase_to_sincos(phase: np.ndarray) -> np.ndarray:
    """Map φ ∈ [0, 1) to (sin 2πφ, cos 2πφ); shape (..., 2)."""
    angle = TWO_PI * np.asarray(phase, dtype=np.float32)
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1).astype(np.float32)


def sincos_to_phase(sincos: np.ndarray) -> np.ndarray:
    """Invert `phase_to_sincos`. Handles non-unit-norm inputs via atan2."""
    s = sincos[..., 0]
    c = sincos[..., 1]
    phi = np.arctan2(s, c) / TWO_PI
    return np.mod(phi, 1.0).astype(np.float32)


# ── Event detection ────────────────────────────────────────────────────────

def detect_gait_events(
    ankle_angle: np.ndarray,
    min_cycle_samples: int = 80,
    smooth_window: int = 5,
) -> np.ndarray:
    """
    Return indices of gait-cycle boundary events inside `ankle_angle`.

    Convention: we take local **minima** of the ankle angle as cycle starts.
    In AB sagittal gait the maximum plantarflexion just after toe-off is
    the most reliably visible landmark across walking speeds and terrains
    — it survives the ±5° / ±20° amplitude swings in the Camargo dataset
    without requiring a separate heel-strike signal channel.

    Parameters
    ----------
    ankle_angle : 1-D array of ankle angles (degrees)
    min_cycle_samples : minimum samples between two consecutive events;
        prevents double-counting noise-driven minima.
    smooth_window : rectangular smoothing window (samples) applied before
        event detection to suppress high-frequency noise.
    """
    y = np.asarray(ankle_angle, dtype=np.float32)
    n = len(y)
    if n < min_cycle_samples * 2:
        return np.array([], dtype=np.int64)

    # Light smoothing — odd-length rectangular kernel, valid convolution
    if smooth_window > 1:
        k = smooth_window | 1  # force odd
        kernel = np.ones(k, dtype=np.float32) / k
        pad = k // 2
        y_smooth = np.convolve(
            np.pad(y, pad, mode="edge"), kernel, mode="valid"
        )
    else:
        y_smooth = y

    # Local minima: y[i-1] > y[i] <= y[i+1]
    diff = np.diff(y_smooth)
    minima = np.where((diff[:-1] < 0) & (diff[1:] >= 0))[0] + 1

    if len(minima) == 0:
        return minima

    # Enforce min separation — greedy filter, keep the earliest of any
    # cluster then require `min_cycle_samples` gap before accepting another.
    kept = [int(minima[0])]
    for idx in minima[1:]:
        if idx - kept[-1] >= min_cycle_samples:
            kept.append(int(idx))
    return np.asarray(kept, dtype=np.int64)


def compute_phase_from_angle(
    ankle_angle: np.ndarray,
    min_cycle_samples: int = 80,
    smooth_window: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive a continuous gait phase signal φ(t) ∈ [0, 1) from a 1-D ankle-angle
    trajectory by detecting cycle boundaries and linearly interpolating.

    Returns
    -------
    phase : np.ndarray of shape (len(ankle_angle),), dtype float32
        φ(t) in [0, 1). Samples outside any detected cycle (before the first
        event or after the last event) are marked as NaN so the caller can
        drop them when constructing training windows.
    events : np.ndarray of event indices (for diagnostics / plotting).
    """
    y = np.asarray(ankle_angle, dtype=np.float32)
    n = len(y)
    events = detect_gait_events(y, min_cycle_samples, smooth_window)

    phase = np.full(n, np.nan, dtype=np.float32)
    if len(events) < 2:
        return phase, events

    for i in range(len(events) - 1):
        a, b = events[i], events[i + 1]
        if b <= a:
            continue
        phase[a:b] = np.linspace(0.0, 1.0, b - a, endpoint=False, dtype=np.float32)

    return phase, events


# ── Reference gait table ───────────────────────────────────────────────────

def build_reference_gait_table(
    phase: np.ndarray,
    ankle_angle: np.ndarray,
    n_bins: int = 100,
) -> np.ndarray:
    """
    Build the canonical ankle-angle reference trajectory θ_ref(φ) from a
    population of (phase, angle) samples.

    Parameters
    ----------
    phase : 1-D array of φ ∈ [0, 1), NaNs are ignored
    ankle_angle : 1-D array of same length
    n_bins : number of phase bins in the lookup table

    Returns
    -------
    theta_ref : np.ndarray of shape (n_bins,), dtype float32.
        theta_ref[k] = mean ankle angle over samples where
                       k/n_bins ≤ φ < (k+1)/n_bins.
        Empty bins are filled by linear interpolation from neighbours,
        so the table is defined everywhere.
    """
    p = np.asarray(phase, dtype=np.float32)
    a = np.asarray(ankle_angle, dtype=np.float32)
    mask = np.isfinite(p) & np.isfinite(a)
    p = p[mask]
    a = a[mask]

    if len(p) == 0:
        raise ValueError("No finite (phase, angle) samples; cannot build reference.")

    bin_idx = np.clip((p * n_bins).astype(np.int64), 0, n_bins - 1)
    theta_ref = np.full(n_bins, np.nan, dtype=np.float32)
    for k in range(n_bins):
        in_bin = a[bin_idx == k]
        if len(in_bin) > 0:
            theta_ref[k] = float(in_bin.mean())

    # Interpolate over any empty bins (wraps around at 0/1).
    missing = np.isnan(theta_ref)
    if missing.any():
        xs = np.arange(n_bins)
        # Use circular interpolation: concatenate three copies and take middle
        tile = np.concatenate([theta_ref] * 3)
        tile_xs = np.concatenate([xs - n_bins, xs, xs + n_bins])
        valid = ~np.isnan(tile)
        tile_interp = np.interp(tile_xs, tile_xs[valid], tile[valid])
        theta_ref = tile_interp[n_bins : 2 * n_bins].astype(np.float32)

    return theta_ref


def save_reference_table(theta_ref: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, theta_ref)


def load_reference_table(path: str) -> np.ndarray:
    return np.load(path)


def reference_lookup(phase: np.ndarray, theta_ref: np.ndarray) -> np.ndarray:
    """
    Sample the reference table at arbitrary phases using linear interpolation
    with wrap-around at 0↔1.

    Parameters
    ----------
    phase : array of φ ∈ [0, 1) (any shape)
    theta_ref : (n_bins,) reference table

    Returns
    -------
    theta : same shape as `phase`, ankle angle in the reference frame.
    """
    p = np.mod(np.asarray(phase, dtype=np.float32), 1.0)
    n_bins = len(theta_ref)

    x = p * n_bins
    i0 = np.floor(x).astype(np.int64) % n_bins
    i1 = (i0 + 1) % n_bins
    w = (x - np.floor(x)).astype(np.float32)
    return ((1.0 - w) * theta_ref[i0] + w * theta_ref[i1]).astype(np.float32)


# ── Metrics ────────────────────────────────────────────────────────────────

def phase_error_pct(
    phi_pred: np.ndarray,
    phi_true: np.ndarray,
) -> float:
    """
    Mean absolute circular distance between two phase sequences, expressed as
    a percentage of the gait cycle (0 = identical, 50 = maximally out of
    phase). This is the metric the paper quotes (e.g. "<6% phase error").
    """
    p1 = np.mod(np.asarray(phi_pred, dtype=np.float32), 1.0)
    p2 = np.mod(np.asarray(phi_true, dtype=np.float32), 1.0)
    diff = np.abs(p1 - p2)
    circ = np.minimum(diff, 1.0 - diff)
    return float(circ.mean() * 100.0)


def sincos_phase_error_pct(sincos_pred: np.ndarray, sincos_true: np.ndarray) -> float:
    """Convenience wrapper that accepts (sin, cos) pairs instead of raw φ."""
    return phase_error_pct(
        sincos_to_phase(sincos_pred),
        sincos_to_phase(sincos_true),
    )
