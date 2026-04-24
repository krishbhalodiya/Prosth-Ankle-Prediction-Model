"""
End-to-end architecture comparison for the Shadow Limb controller.

Trains (or reuses) both the BiLSTM and the Transformer on the Camargo 2021
benchmark, runs a per-terrain evaluation on each, and emits a side-by-side
RMSE/R²/latency table plus a grouped-bar comparison plot.

This directly reproduces the comparison layout of Table 2 in:
    "State-of-the-Art Deep Learning Control Strategies for Transtibial
     Prostheses" (§4.3), scoped to this project's data + splits.

Usage
-----
Train both from scratch and compare:
    python scripts/compare_architectures.py --epochs 30

Skip training if checkpoints already exist:
    python scripts/compare_architectures.py --skip-train-if-exists

Evaluate only (assumes checkpoints are already there):
    python scripts/compare_architectures.py --eval-only

Restrict to a subset of architectures:
    python scripts/compare_architectures.py --archs single transformer
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.trajectory import config  # noqa: E402
from src.trajectory import phase as phase_utils  # noqa: E402
from src.trajectory.dataset import Normalizer  # noqa: E402
from src.trajectory.model import build_model  # noqa: E402
from src.trajectory.train import train as run_training  # noqa: E402
from src.trajectory.evaluate import (  # noqa: E402
    per_terrain_evaluation,
    per_terrain_phase_evaluation,
    plot_terrain_comparison,
    print_terrain_comparison_table,
    profile_latency,
    save_terrain_results,
)

ARCH_LABELS = {
    "single": "BiLSTM",
    "trajectory": "EncDec-LSTM",
    "transformer": "Transformer",
    "phase_lstm": "Phase-BiLSTM",
    "phase_transformer": "Phase-Transformer",
}


def ensure_trained(mode: str, epochs: int, batch_size: int, lr: float, skip_if_exists: bool):
    """Train one architecture if its checkpoint is missing (or always)."""
    path = config.model_path_for_mode(mode)
    if skip_if_exists and os.path.exists(path):
        print(f"[SKIP] Checkpoint already exists: {path}")
        return path

    print(f"\n{'=' * 70}\nTraining mode={mode!r} -> {path}\n{'=' * 70}")
    run_training([
        "--mode", mode,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
    ])
    return path


def load_checkpoint_model(mode: str, device: torch.device):
    path = config.model_path_for_mode(mode)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No checkpoint found for mode={mode!r} at {path}. "
            "Train it first or drop --eval-only."
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(mode=ckpt["config"].get("mode", mode)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, path


def main():
    parser = argparse.ArgumentParser(description="Compare BiLSTM vs Transformer")
    parser.add_argument(
        "--archs",
        nargs="+",
        default=["single", "transformer"],
        choices=list(ARCH_LABELS.keys()),
        help=(
            "Architectures to include in the comparison. "
            "Direct-angle: single, trajectory, transformer. "
            "Phase-regression: phase_lstm, phase_transformer. "
            "Mixing both in one run is supported — RMSE will be compared "
            "on the same axis (reconstructed angle RMSE for phase modes)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument(
        "--skip-train-if-exists",
        action="store_true",
        help="Don't retrain an architecture if its checkpoint already exists.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training entirely — only evaluate existing checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.MODEL_DIR,
        help="Where to write the comparison JSON + PNG.",
    )
    opts = parser.parse_args()
    os.makedirs(opts.output_dir, exist_ok=True)

    # ── 1. Train (or reuse) each architecture ────────────────────────────
    if not opts.eval_only:
        for mode in opts.archs:
            ensure_trained(
                mode,
                epochs=opts.epochs,
                batch_size=opts.batch_size,
                lr=opts.lr,
                skip_if_exists=opts.skip_train_if_exists,
            )

    # ── 2. Load shared normalizer ────────────────────────────────────────
    norm_path = os.path.join(config.MODEL_DIR, "normalizer.pkl")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(
            f"Normalizer missing at {norm_path}. Run training at least once "
            "or regenerate it from the dataset pipeline."
        )
    normalizer = Normalizer().load(norm_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    # ── 3. Evaluate each architecture ────────────────────────────────────
    results_by_arch: dict[str, dict] = {}
    latency_by_arch: dict[str, dict] = {}
    params_by_arch: dict[str, int] = {}

    for mode in opts.archs:
        label = ARCH_LABELS[mode]
        print(f"\n{'=' * 70}\nEvaluating {label}  (mode={mode})\n{'=' * 70}")

        model, ckpt, path = load_checkpoint_model(mode, device)
        params = model.count_parameters()
        params_by_arch[label] = params
        print(f"  Parameters:     {params:,}")
        print(f"  Checkpoint:     {path}")
        print(f"  Best val RMSE:  {ckpt.get('val_rmse', float('nan')):.4f} deg")

        t0 = time.time()
        if config.is_phase_mode(mode):
            if not os.path.exists(config.REFERENCE_GAIT_PATH):
                raise FileNotFoundError(
                    "Phase mode needs models/reference_gait.npy — run "
                    "`python -m src.trajectory.train --mode phase_lstm` "
                    "at least once to build it."
                )
            theta_ref = phase_utils.load_reference_table(config.REFERENCE_GAIT_PATH)
            per_terrain = per_terrain_phase_evaluation(
                model, config.TEST_SUBJECTS, normalizer, theta_ref, device
            )
        else:
            per_terrain = per_terrain_evaluation(
                model, config.TEST_SUBJECTS, normalizer, device
            )
        print(f"  Terrain eval completed in {time.time() - t0:.1f}s")

        for terrain, metrics in per_terrain.items():
            phase_col = (
                f"  φ-err={metrics['phase_err_pct']:.2f}%"
                if "phase_err_pct" in metrics
                else ""
            )
            print(
                f"    {terrain:<14s}"
                f"  RMSE={metrics['rmse']:.3f}"
                f"  R²={metrics['r2']:.3f}"
                f"  MAE={metrics['mae']:.3f}"
                f"{phase_col}"
                f"  (n={metrics['n_windows']})"
            )

        results_by_arch[label] = per_terrain

        # Persist each architecture's per-terrain JSON individually.
        save_terrain_results(
            per_terrain,
            architecture=label,
            save_path=os.path.join(opts.output_dir, f"terrain_results_{label}.json"),
            extra_meta={"mode": mode, "params": params, "checkpoint": path},
        )

        # Latency
        print("  Measuring latency...")
        lat = profile_latency(model, device, n_iterations=500)
        latency_by_arch[label] = lat
        print(f"    mean={lat['mean_ms']:.2f} ms  p95={lat['p95_ms']:.2f} ms")

    # ── 4. Side-by-side comparison outputs ───────────────────────────────
    print(f"\n{'=' * 70}\nArchitecture Comparison\n{'=' * 70}")
    print_terrain_comparison_table(results_by_arch)

    print("\nLatency / Size Summary")
    print(f"{'arch':<16}{'params':>12}{'mean_ms':>12}{'p95_ms':>12}")
    print("-" * 52)
    for label in results_by_arch:
        p = params_by_arch[label]
        lat = latency_by_arch[label]
        print(f"{label:<16}{p:>12,}{lat['mean_ms']:>12.2f}{lat['p95_ms']:>12.2f}")

    summary = {
        "test_subjects": config.TEST_SUBJECTS,
        "window_ms": config.WINDOW_MS,
        "sample_rate_hz": config.SAMPLE_RATE_HZ,
        "input_channels": config.NUM_INPUT_CHANNELS,
        "architectures": {
            label: {
                "params": params_by_arch[label],
                "latency_ms": latency_by_arch[label],
                "per_terrain": results_by_arch[label],
            }
            for label in results_by_arch
        },
    }
    summary_path = os.path.join(opts.output_dir, "architecture_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    plot_path = os.path.join(opts.output_dir, "architecture_comparison_rmse.png")
    plot_terrain_comparison(results_by_arch, metric="rmse", save_path=plot_path)


if __name__ == "__main__":
    main()
