"""
Evaluation utilities for the Shadow Limb trajectory model.

Provides per-terrain RMSE breakdowns, gait-cycle overlay plots,
inference latency profiling, and error distribution analysis.

Usage:
    python -m src.trajectory.evaluate [--model-path PATH]
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from . import config
from .dataset import (
    build_subject_data,
    build_subject_phase_data,
    discover_trials,
    extract_windows,
    extract_phase_windows,
    load_mat_trial,
    Normalizer,
    GaitWindowDataset,
)
from .model import build_model
from . import phase as phase_utils
from torch.utils.data import DataLoader


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def compute_r2(preds: np.ndarray, targets: np.ndarray) -> float:
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    if ss_tot < 1e-8:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_mae(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - targets)))


# ── Per-Terrain Analysis ─────────────────────────────────────────────────────

def per_terrain_evaluation(
    model: nn.Module,
    subjects: list[str],
    normalizer: Normalizer,
    device: torch.device,
    camargo_dir: str = config.CAMARGO_DIR,
) -> dict[str, dict]:
    """
    Evaluate a *direct-angle* model on each terrain condition separately.

    Returns a dict: {condition_name: {"rmse": ..., "r2": ..., "mae": ..., "n_windows": ...}}
    """
    terrain_results = defaultdict(lambda: {"preds": [], "targets": []})
    trials = discover_trials(camargo_dir, subjects)

    for trial_info in trials:
        try:
            trial_data = load_mat_trial(trial_info["path"])
        except Exception:
            continue

        X, y = extract_windows(
            trial_data,
            config.INPUT_COLS,
            config.TARGET_COL,
            config.WINDOW_SAMPLES,
            config.EFFECTIVE_HORIZON,
            config.STEP_SAMPLES,
        )
        if len(X) == 0:
            continue

        X = normalizer.transform(X)
        X_t = torch.from_numpy(X).to(device)

        model.eval()
        with torch.no_grad():
            preds = model(X_t).cpu().numpy()

        condition = _classify_terrain(trial_info["condition"])
        terrain_results[condition]["preds"].append(preds.flatten())
        terrain_results[condition]["targets"].append(y.flatten())

    results = {}
    for terrain in _sort_terrains(terrain_results.keys()):
        data = terrain_results[terrain]
        p = np.concatenate(data["preds"])
        t = np.concatenate(data["targets"])
        results[terrain] = {
            "rmse": compute_rmse(p, t),
            "r2": compute_r2(p, t),
            "mae": compute_mae(p, t),
            "n_windows": int(len(p)),
        }
    return results


def per_terrain_phase_evaluation(
    model: nn.Module,
    subjects: list[str],
    normalizer: Normalizer,
    theta_ref: np.ndarray,
    device: torch.device,
    camargo_dir: str = config.CAMARGO_DIR,
) -> dict[str, dict]:
    """
    Evaluate a *phase-regression* model on each terrain condition separately.

    For each terrain, reports:
      - phase_err_pct : mean circular phase error (% of cycle)
      - rmse          : reconstructed-angle RMSE (degrees) via θ_ref(φ)
      - mae           : reconstructed-angle MAE  (degrees)
      - r2            : reconstructed-angle R² against true ankle angle
      - n_windows     : number of windows aggregated
    """
    buckets: dict[str, dict] = defaultdict(
        lambda: {"phi_pred": [], "phi_true": [], "theta_true": []}
    )
    trials = discover_trials(camargo_dir, subjects)

    model.eval()
    for trial_info in trials:
        try:
            trial_data = load_mat_trial(trial_info["path"])
        except Exception:
            continue

        X, y_sc, y_ang = extract_phase_windows(
            trial_data,
            config.INPUT_COLS,
            config.TARGET_COL,
            config.WINDOW_SAMPLES,
            config.STEP_SAMPLES,
        )
        if len(X) == 0:
            continue

        X = normalizer.transform(X)
        with torch.no_grad():
            preds = model(torch.from_numpy(X).to(device)).cpu().numpy()

        phi_pred = phase_utils.sincos_to_phase(preds)
        phi_true = phase_utils.sincos_to_phase(y_sc)

        terrain = _classify_terrain(trial_info["condition"])
        buckets[terrain]["phi_pred"].append(phi_pred)
        buckets[terrain]["phi_true"].append(phi_true)
        buckets[terrain]["theta_true"].append(y_ang)

    results: dict[str, dict] = {}
    for terrain in _sort_terrains(buckets.keys()):
        data = buckets[terrain]
        pp = np.concatenate(data["phi_pred"])
        pt = np.concatenate(data["phi_true"])
        tt = np.concatenate(data["theta_true"])
        theta_pred = phase_utils.reference_lookup(pp, theta_ref)
        results[terrain] = {
            "phase_err_pct": phase_utils.phase_error_pct(pp, pt),
            "rmse": compute_rmse(theta_pred, tt),
            "mae": compute_mae(theta_pred, tt),
            "r2": compute_r2(theta_pred, tt),
            "n_windows": int(len(pp)),
        }
    return results


def save_terrain_results(
    per_terrain: dict,
    architecture: str,
    save_path: str,
    extra_meta: dict | None = None,
):
    """Persist a per-terrain result dict as JSON for later comparison."""
    payload = {
        "architecture": architecture,
        "metrics": per_terrain,
    }
    if extra_meta:
        payload["meta"] = extra_meta
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Terrain results saved to {save_path}")


def plot_terrain_comparison(
    results_by_arch: dict[str, dict],
    metric: str = "rmse",
    save_path: str | None = None,
):
    """
    Grouped bar chart: one bar per architecture inside each terrain group.

    Parameters
    ----------
    results_by_arch : {arch_name: {terrain_name: {"rmse": ..., ...}}}
    metric : which metric to plot ("rmse" | "mae" | "r2")
    """
    all_terrains = set()
    for per_terrain in results_by_arch.values():
        all_terrains.update(per_terrain.keys())
    terrains = _sort_terrains(all_terrains)

    archs = list(results_by_arch.keys())
    n_arch = len(archs)
    x = np.arange(len(terrains))
    bar_w = 0.8 / max(n_arch, 1)

    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(terrains) + 2), 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_arch, 3)))

    for i, arch in enumerate(archs):
        vals = [results_by_arch[arch].get(t, {}).get(metric, np.nan) for t in terrains]
        offset = (i - (n_arch - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=arch, color=colors[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(terrains, rotation=20, ha="right")
    unit = "" if metric == "r2" else " (deg)"
    ax.set_ylabel(f"{metric.upper()}{unit}")
    ax.set_title(f"Per-Terrain {metric.upper()} by Architecture")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved terrain comparison plot to {save_path}")
    plt.show()


def print_terrain_comparison_table(results_by_arch: dict[str, dict]):
    """Print a side-by-side RMSE table for quick visual diff in the terminal."""
    all_terrains = set()
    for per_terrain in results_by_arch.values():
        all_terrains.update(per_terrain.keys())
    terrains = _sort_terrains(all_terrains)
    archs = list(results_by_arch.keys())

    header = f"{'terrain':<18}" + "".join(f"{a:>14}" for a in archs) + f"{'Δ':>10}"
    print(header)
    print("-" * len(header))
    for terrain in terrains:
        row = f"{terrain:<18}"
        rmses = []
        for a in archs:
            v = results_by_arch[a].get(terrain, {}).get("rmse", np.nan)
            rmses.append(v)
            row += f"{v:>14.3f}" if not np.isnan(v) else f"{'—':>14}"
        if len(rmses) >= 2 and not any(np.isnan(rmses[:2])):
            delta = rmses[1] - rmses[0]
            row += f"{delta:>+10.3f}"
        else:
            row += f"{'—':>10}"
        print(row)


def _classify_terrain(condition_name: str) -> str:
    """
    Map a Camargo trial filename to a fine-grained terrain category.

    The Camargo dataset files follow the convention
    ``<terrain>_<direction>_<trial>.mat`` (e.g. ``ramp_ascent_01``), so we
    split ramp and stair conditions by direction — ascending a ramp is a
    fundamentally different control problem than descending one, and
    aggregating them hides the most important failure mode of a
    trajectory-tracking model.
    """
    name = condition_name.lower()

    if "ramp" in name or "incline" in name:
        if "descent" in name or "down" in name:
            return "ramp_descent"
        return "ramp_ascent"

    if "stair" in name:
        if "descent" in name or "down" in name:
            return "stair_descent"
        return "stair_ascent"

    if "treadmill" in name:
        return "treadmill"

    if "level" in name or "ground" in name or "walk" in name:
        return "level_ground"

    return "other"


# Canonical ordering for reports / plots (easy → hard roughly speaking).
TERRAIN_ORDER = [
    "level_ground",
    "treadmill",
    "ramp_ascent",
    "ramp_descent",
    "stair_ascent",
    "stair_descent",
    "other",
]


def _sort_terrains(terrains):
    known = [t for t in TERRAIN_ORDER if t in terrains]
    extra = sorted(t for t in terrains if t not in TERRAIN_ORDER)
    return known + extra


# ── Gait Cycle Overlay Plot ──────────────────────────────────────────────────

def plot_gait_overlay(
    model: nn.Module,
    trial_path: str,
    normalizer: Normalizer,
    device: torch.device,
    n_cycles: int = 3,
    save_path: str | None = None,
):
    """
    Plot predicted vs. ground truth ankle angle for consecutive gait cycles.
    """
    trial_data = load_mat_trial(trial_path)
    X, y = extract_windows(
        trial_data,
        config.INPUT_COLS,
        config.TARGET_COL,
        config.WINDOW_SAMPLES,
        1,  # single-step for continuous overlay
        step=1,
    )
    if len(X) == 0:
        print(f"No valid windows from {trial_path}")
        return

    X = normalizer.transform(X)
    X_t = torch.from_numpy(X).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(X_t).cpu().numpy().flatten()

    targets = y.flatten()
    n_show = min(len(preds), n_cycles * config.SAMPLE_RATE_HZ)
    t = np.arange(n_show) / config.SAMPLE_RATE_HZ

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(t, targets[:n_show], label="Ground Truth", color="steelblue", linewidth=2)
    ax.plot(t, preds[:n_show], label="Predicted (Shadow Limb)", color="crimson", linewidth=1.5, alpha=0.85)
    ax.set_ylabel("Ankle Angle (deg)")
    ax.set_title("Shadow Limb Prediction vs. Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    error = preds[:n_show] - targets[:n_show]
    ax.fill_between(t, error, color="orange", alpha=0.4)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("Error (deg)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Prediction Error (RMSE: {compute_rmse(preds[:n_show], targets[:n_show]):.2f} deg)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


# ── Latency Profiling ────────────────────────────────────────────────────────

def profile_latency(
    model: nn.Module,
    device: torch.device,
    n_iterations: int = 1000,
) -> dict:
    """
    Measure inference latency for a single IMU window.

    Returns dict with mean, std, min, max latency in milliseconds.
    """
    model.eval()
    dummy = torch.randn(1, config.WINDOW_SAMPLES, config.NUM_INPUT_CHANNELS, device=device)

    # Warm up
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    return {
        "mean_ms": float(latencies.mean()),
        "std_ms": float(latencies.std()),
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "device": str(device),
        "n_iterations": n_iterations,
    }


# ── Error Distribution ───────────────────────────────────────────────────────

def plot_error_distribution(
    preds: np.ndarray,
    targets: np.ndarray,
    save_path: str | None = None,
):
    """Histogram and statistics of prediction errors."""
    errors = preds - targets

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(errors, bins=80, density=True, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0].axvline(0, color="crimson", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Error (deg)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Prediction Error Distribution")

    abs_err = np.abs(errors)
    percentiles = [50, 75, 90, 95, 99]
    pct_vals = [np.percentile(abs_err, p) for p in percentiles]
    axes[1].barh(
        [f"P{p}" for p in percentiles],
        pct_vals,
        color="steelblue",
        alpha=0.7,
        edgecolor="white",
    )
    axes[1].set_xlabel("|Error| (deg)")
    axes[1].set_title("Absolute Error Percentiles")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Error stats: mean={errors.mean():.3f}, std={errors.std():.3f}, "
          f"median_abs={np.median(abs_err):.3f}")


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Shadow Limb model")
    parser.add_argument("--model-path", type=str, default=config.MODEL_PATH)
    parser.add_argument("--data-dir", type=str, default=config.CAMARGO_DIR)
    parser.add_argument(
        "--results-name",
        type=str,
        default=None,
        help="Short name used when saving per-terrain JSON results "
             "(defaults to architecture class name).",
    )
    opts = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (architecture auto-detected from checkpoint)
    checkpoint = torch.load(opts.model_path, map_location=device, weights_only=False)
    ckpt_cfg = checkpoint["config"]
    ckpt_input_dim = ckpt_cfg.get("input_dim", config.NUM_INPUT_CHANNELS)
    if ckpt_input_dim != config.NUM_INPUT_CHANNELS:
        print(
            f"[WARN] Checkpoint was trained with input_dim={ckpt_input_dim} "
            f"but config.NUM_INPUT_CHANNELS={config.NUM_INPUT_CHANNELS}. "
            "Toggle USE_THIGH_IMU in config.py to match."
        )
    mode = ckpt_cfg.get("mode", "single")
    is_phase = config.is_phase_mode(mode)
    model = build_model(mode=mode).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    arch_name = type(model).__name__
    mode_tag = f"[{mode}]"
    print(f"Loaded {arch_name} {mode_tag} from {opts.model_path} (epoch {checkpoint['epoch']})")

    theta_ref = None
    if is_phase:
        if not os.path.exists(config.REFERENCE_GAIT_PATH):
            raise FileNotFoundError(
                f"Phase model checkpoint requires {config.REFERENCE_GAIT_PATH}. "
                "Retrain once or re-run scripts/build_reference_gait.py."
            )
        theta_ref = phase_utils.load_reference_table(config.REFERENCE_GAIT_PATH)
        print(
            f"Reference gait table: {len(theta_ref)} bins, "
            f"range [{theta_ref.min():.2f}, {theta_ref.max():.2f}] deg"
        )

    # Load normalizer
    norm_path = os.path.join(config.MODEL_DIR, "normalizer.pkl")
    normalizer = Normalizer().load(norm_path)

    # Overall test metrics
    print("\n=== Test Set Metrics ===")
    if is_phase:
        X_test, y_test, y_angle_test, _ = build_subject_phase_data(
            config.TEST_SUBJECTS, opts.data_dir, verbose=False
        )
        X_test = normalizer.transform(X_test)
        model.eval()
        with torch.no_grad():
            all_preds = (
                model(torch.from_numpy(X_test).to(device)).cpu().numpy()
            )
        phi_pred = phase_utils.sincos_to_phase(all_preds)
        phi_true = phase_utils.sincos_to_phase(y_test)
        theta_pred = phase_utils.reference_lookup(phi_pred, theta_ref)
        print(f"  Phase error:         {phase_utils.phase_error_pct(phi_pred, phi_true):.3f} %")
        print(f"  Reconstructed RMSE:  {compute_rmse(theta_pred, y_angle_test):.3f} deg")
        print(f"  Reconstructed MAE:   {compute_mae(theta_pred, y_angle_test):.3f} deg")
        print(f"  Reconstructed R²:    {compute_r2(theta_pred, y_angle_test):.4f}")
    else:
        X_test, y_test, _ = build_subject_data(config.TEST_SUBJECTS, opts.data_dir)
        X_test = normalizer.transform(X_test)
        ds = GaitWindowDataset(X_test, y_test)
        loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False)

        all_preds, all_targets = [], []
        model.eval()
        for X_b, y_b in loader:
            with torch.no_grad():
                p = model(X_b.to(device)).cpu().numpy()
            all_preds.append(p.flatten())
            all_targets.append(y_b.numpy().flatten())
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        print(f"  RMSE: {compute_rmse(all_preds, all_targets):.4f} deg")
        print(f"  R²:   {compute_r2(all_preds, all_targets):.4f}")
        print(f"  MAE:  {compute_mae(all_preds, all_targets):.4f} deg")

    # Per-terrain breakdown
    print("\n=== Per-Terrain Evaluation ===")
    if is_phase:
        terrain_res = per_terrain_phase_evaluation(
            model, config.TEST_SUBJECTS, normalizer, theta_ref, device, opts.data_dir
        )
        for terrain, metrics in terrain_res.items():
            print(
                f"  {terrain:15s}  φ-err={metrics['phase_err_pct']:.2f} %  "
                f"RMSE={metrics['rmse']:.3f} deg  R²={metrics['r2']:.3f}  "
                f"(n={metrics['n_windows']})"
            )
    else:
        terrain_res = per_terrain_evaluation(
            model, config.TEST_SUBJECTS, normalizer, device, opts.data_dir
        )
        for terrain, metrics in terrain_res.items():
            print(
                f"  {terrain:15s}  RMSE={metrics['rmse']:.3f} deg  "
                f"R²={metrics['r2']:.3f}  MAE={metrics['mae']:.3f} deg  "
                f"(n={metrics['n_windows']})"
            )

    # Persist terrain JSON so multiple architectures can be compared later.
    results_name = opts.results_name or (mode if is_phase else arch_name)
    terrain_json_path = os.path.join(
        config.MODEL_DIR, f"terrain_results_{results_name}.json"
    )
    save_terrain_results(
        terrain_res,
        architecture=f"{arch_name} ({mode})",
        save_path=terrain_json_path,
        extra_meta={
            "model_path": os.path.abspath(opts.model_path),
            "test_subjects": config.TEST_SUBJECTS,
            "ckpt_config": ckpt_cfg,
            "mode": mode,
            "is_phase_mode": is_phase,
        },
    )

    # Latency
    print("\n=== Latency Profile ===")
    latency = profile_latency(model, device)
    print(f"  Mean: {latency['mean_ms']:.2f} ms")
    print(f"  P95:  {latency['p95_ms']:.2f} ms")
    print(f"  P99:  {latency['p99_ms']:.2f} ms")

    # Plots: only meaningful for direct-angle modes out of the box.
    if not is_phase:
        plot_error_distribution(all_preds, all_targets)
    plot_terrain_comparison(
        {results_name: terrain_res},
        metric="rmse",
        save_path=os.path.join(config.MODEL_DIR, f"terrain_rmse_{results_name}.png"),
    )


if __name__ == "__main__":
    main()
