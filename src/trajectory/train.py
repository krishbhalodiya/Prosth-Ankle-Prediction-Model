"""
Training pipeline for the Shadow Limb controller.

Supports three *angle-regression* modes and two *phase-regression* modes:

    single              : BiLSTM, scalar ankle angle
    trajectory          : Encoder-Decoder LSTM, multi-step angle trajectory
    transformer         : Self-attention, scalar ankle angle
    phase_lstm          : BiLSTM, (sin 2πφ, cos 2πφ)
    phase_transformer   : Self-attention, (sin 2πφ, cos 2πφ)

For phase modes we additionally:
  * track phase-error in % of cycle (the paper's headline metric),
  * build and save the reference-gait lookup table θ_ref(φ) from the
    (phase, ankle-angle) samples in the training set, so that downstream
    inference can map predicted φ back to an ankle-angle command.

Usage:
    python -m src.trajectory.train --mode phase_lstm --epochs 10
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from . import config
from .dataset import (
    build_dataloaders,
    build_subject_phase_data,
    Normalizer,
)
from .model import build_model
from . import phase as phase_utils


# ── Helper metrics ──────────────────────────────────────────────────────────

def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.sqrt(((predictions - targets) ** 2).mean()).item()


def r_squared(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    if ss_tot < 1e-8:
        return 0.0
    return (1 - ss_res / ss_tot).item()


def phase_error_pct_torch(pred_sincos: torch.Tensor, true_sincos: torch.Tensor) -> float:
    """Mean circular phase error (% of cycle) directly from (sin, cos) pairs."""
    phi_p = torch.atan2(pred_sincos[..., 0], pred_sincos[..., 1]) / (2 * np.pi)
    phi_t = torch.atan2(true_sincos[..., 0], true_sincos[..., 1]) / (2 * np.pi)
    phi_p = torch.remainder(phi_p, 1.0)
    phi_t = torch.remainder(phi_t, 1.0)
    diff = (phi_p - phi_t).abs()
    circ = torch.minimum(diff, 1.0 - diff)
    return float(circ.mean().item() * 100.0)


# ── Epoch loops ────────────────────────────────────────────────────────────

def _unpack_batch(batch, phase_mode: bool):
    """Normalize batch shape: phase mode yields (X, y, y_angle), else (X, y)."""
    if phase_mode:
        X, y, y_angle = batch
        return X, y, y_angle
    X, y = batch
    return X, y, None


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    phase_mode: bool,
    grad_clip: float = config.GRAD_CLIP,
) -> dict:
    model.train()
    total_loss = 0.0
    n_seen = 0
    all_preds, all_targets = [], []

    for batch in loader:
        X, y, _ = _unpack_batch(batch, phase_mode)
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n_seen += X.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(y.detach().cpu())

    preds_cat = torch.cat(all_preds)
    tgt_cat = torch.cat(all_targets)

    metrics = {"loss": total_loss / max(n_seen, 1)}
    if phase_mode:
        metrics["phase_err_pct"] = phase_error_pct_torch(preds_cat, tgt_cat)
        # sin/cos MSE as a secondary proxy
        metrics["sincos_rmse"] = rmse(preds_cat, tgt_cat)
    else:
        metrics["rmse"] = rmse(preds_cat, tgt_cat)
        metrics["r2"] = r_squared(preds_cat, tgt_cat)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    phase_mode: bool,
    theta_ref: np.ndarray | None = None,
) -> dict:
    """
    Evaluate a model on a loader.

    For phase mode we additionally reconstruct θ from predicted φ via the
    reference table and report the resulting ankle-angle RMSE — the apples-
    to-apples comparison with direct-angle regression.
    """
    model.eval()
    total_loss = 0.0
    n_seen = 0
    all_preds, all_targets, all_angles = [], [], []

    for batch in loader:
        X, y, y_angle = _unpack_batch(batch, phase_mode)
        X = X.to(device)
        y = y.to(device)
        preds = model(X)
        loss = criterion(preds, y)
        total_loss += loss.item() * X.size(0)
        n_seen += X.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
        if y_angle is not None:
            all_angles.append(y_angle)

    preds_cat = torch.cat(all_preds)
    tgt_cat = torch.cat(all_targets)

    metrics = {"loss": total_loss / max(n_seen, 1)}
    if phase_mode:
        metrics["phase_err_pct"] = phase_error_pct_torch(preds_cat, tgt_cat)
        metrics["sincos_rmse"] = rmse(preds_cat, tgt_cat)
        # Reconstruct ankle angle via reference table when available.
        if theta_ref is not None and all_angles:
            phi_pred = phase_utils.sincos_to_phase(preds_cat.numpy())
            theta_pred = phase_utils.reference_lookup(phi_pred, theta_ref)
            theta_true = torch.cat(all_angles).numpy()
            err = theta_pred - theta_true
            metrics["reconstructed_rmse_deg"] = float(np.sqrt((err ** 2).mean()))
            metrics["reconstructed_mae_deg"] = float(np.abs(err).mean())
        # Use phase error as the scalar to drive early stopping / LR schedule.
        metrics["primary"] = metrics["phase_err_pct"]
    else:
        metrics["rmse"] = rmse(preds_cat, tgt_cat)
        metrics["r2"] = r_squared(preds_cat, tgt_cat)
        metrics["primary"] = metrics["rmse"]
    return metrics


# ── Reference-table helper ──────────────────────────────────────────────────

def build_and_save_reference_table(
    data_dir: str,
    verbose: bool = True,
) -> np.ndarray:
    """
    Build the reference-gait table θ_ref(φ) from the TRAIN subjects and
    write it to disk. This is called once at the start of phase-mode
    training so that validation/test reconstructions can use it.
    """
    if verbose:
        print("\n=== Building Reference Gait Table ===")
    # Use a dense step so we get phase samples at every timestep that has a
    # valid phase, not just once per sliding window.
    _, y_sc, y_angle, _ = build_subject_phase_data(
        config.TRAIN_SUBJECTS,
        camargo_dir=data_dir,
        step=1,
        verbose=False,
    )
    phase = phase_utils.sincos_to_phase(y_sc)
    theta_ref = phase_utils.build_reference_gait_table(
        phase, y_angle, n_bins=config.N_PHASE_BINS
    )
    phase_utils.save_reference_table(theta_ref, config.REFERENCE_GAIT_PATH)
    if verbose:
        print(
            f"  Reference table: {config.N_PHASE_BINS} bins, "
            f"range [{theta_ref.min():.2f}, {theta_ref.max():.2f}] deg"
        )
        print(f"  Saved to {config.REFERENCE_GAIT_PATH}")
    return theta_ref


# ── Main ────────────────────────────────────────────────────────────────────

def train(args=None):
    parser = argparse.ArgumentParser(description="Train Shadow Limb model")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument(
        "--mode",
        choices=[
            "single",
            "trajectory",
            "transformer",
            "phase_lstm",
            "phase_transformer",
        ],
        default=config.PREDICTION_MODE,
        help="Prediction target and backbone combination.",
    )
    parser.add_argument("--data-dir", type=str, default=config.CAMARGO_DIR)
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override checkpoint path (defaults to config.model_path_for_mode(mode))",
    )
    opts = parser.parse_args(args)

    phase_mode = config.is_phase_mode(opts.mode)
    model_path = opts.model_path or config.model_path_for_mode(opts.mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n=== Loading Data ===")
    train_loader, val_loader, test_loader, normalizer = build_dataloaders(
        camargo_dir=opts.data_dir,
        batch_size=opts.batch_size,
        mode=opts.mode,
    )

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    norm_path = os.path.join(config.MODEL_DIR, "normalizer.pkl")
    normalizer.save(norm_path)
    print(f"Normalizer saved to {norm_path}")

    # ── Reference gait table (phase mode only) ────────────────────────────
    theta_ref = None
    if phase_mode:
        theta_ref = build_and_save_reference_table(opts.data_dir)

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n=== Building Model ===")
    model = build_model(mode=opts.mode).to(device)
    print(f"Architecture: {type(model).__name__}")
    print(f"Parameters: {model.count_parameters():,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
    )

    # ── Training Loop ─────────────────────────────────────────────────────
    print(f"\n=== Training ({opts.epochs} epochs, mode={opts.mode}) ===\n")

    best_primary = float("inf")
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(1, opts.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, phase_mode
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, phase_mode, theta_ref
        )

        scheduler.step(val_metrics["primary"])
        elapsed = time.time() - t0

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        lr_now = optimizer.param_groups[0]["lr"]
        if phase_mode:
            recon = val_metrics.get("reconstructed_rmse_deg", float("nan"))
            print(
                f"Epoch {epoch:3d}/{opts.epochs} | "
                f"Train φ-err: {train_metrics['phase_err_pct']:.2f}% | "
                f"Val φ-err: {val_metrics['phase_err_pct']:.2f}%  "
                f"Recon RMSE: {recon:.3f} deg | "
                f"LR: {lr_now:.2e} | {elapsed:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch:3d}/{opts.epochs} | "
                f"Train RMSE: {train_metrics['rmse']:.4f} deg, R²: {train_metrics['r2']:.4f} | "
                f"Val RMSE: {val_metrics['rmse']:.4f} deg, R²: {val_metrics['r2']:.4f} | "
                f"LR: {lr_now:.2e} | {elapsed:.1f}s"
            )

        # Checkpoint best
        if val_metrics["primary"] < best_primary:
            best_primary = val_metrics["primary"]
            patience_counter = 0
            ckpt_config = {
                "mode": opts.mode,
                "input_dim": config.NUM_INPUT_CHANNELS,
                "window_samples": config.WINDOW_SAMPLES,
                "horizon_samples": config.HORIZON_SAMPLES,
                "is_phase_mode": phase_mode,
            }
            if opts.mode in ("single", "trajectory", "phase_lstm"):
                ckpt_config.update({
                    "hidden_dim": config.HIDDEN_DIM,
                    "num_layers": config.NUM_LSTM_LAYERS,
                    "dropout": config.DROPOUT,
                    "bidirectional": config.BIDIRECTIONAL,
                })
            if opts.mode in ("transformer", "phase_transformer"):
                ckpt_config.update({
                    "d_model": config.TRANSFORMER_D_MODEL,
                    "nhead": config.TRANSFORMER_NHEAD,
                    "num_layers": config.TRANSFORMER_NUM_LAYERS,
                    "dim_feedforward": config.TRANSFORMER_DIM_FEEDFORWARD,
                    "dropout": config.TRANSFORMER_DROPOUT,
                })

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_primary": best_primary,
                    "val_metrics": val_metrics,
                    "config": ckpt_config,
                },
                model_path,
            )
            if phase_mode:
                print(f"  -> New best model saved (val φ-err={best_primary:.3f}%)")
            else:
                print(f"  -> New best model saved (val RMSE={best_primary:.4f} deg)")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (patience={config.PATIENCE})")
                break

    # ── Final Evaluation on Test Set ──────────────────────────────────────
    print("\n=== Test Set Evaluation ===")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device, phase_mode, theta_ref)
    if phase_mode:
        print(f"Test φ-err: {test_metrics['phase_err_pct']:.3f} %")
        if "reconstructed_rmse_deg" in test_metrics:
            print(f"Test Reconstructed RMSE: {test_metrics['reconstructed_rmse_deg']:.3f} deg")
            print(f"Test Reconstructed MAE:  {test_metrics['reconstructed_mae_deg']:.3f} deg")
    else:
        print(f"Test RMSE: {test_metrics['rmse']:.4f} deg")
        print(f"Test R²:   {test_metrics['r2']:.4f}")
    print(f"Test Loss: {test_metrics['loss']:.6f}")

    # Save training history
    history_path = os.path.join(config.MODEL_DIR, f"training_history_{opts.mode}.json")
    serializable_history = {
        split: [{k: float(v) for k, v in epoch_m.items()} for epoch_m in epochs]
        for split, epochs in history.items()
    }
    serializable_history["test"] = {k: float(v) for k, v in test_metrics.items()}
    with open(history_path, "w") as f:
        json.dump(serializable_history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    print(f"Best model saved to {model_path}")


if __name__ == "__main__":
    train()
