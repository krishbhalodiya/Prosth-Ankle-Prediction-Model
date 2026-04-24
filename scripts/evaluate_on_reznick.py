"""
Evaluate a Camargo-trained Shadow Limb phase model on the Reznick 2021
dataset. This is the paper's §3.2 transfer test: can a model trained on
one AB dataset predict the gait phase on a completely different AB
dataset (continuous-speed treadmill walking)?

The test target is Reznick's median phase error on held-out subjects.
The paper cites <6 % phase error as the transfer benchmark.

Usage
-----
    python scripts/evaluate_on_reznick.py --model-path models/shadow_limb_phase_lstm.pt
    python scripts/evaluate_on_reznick.py --model-path models/shadow_limb_phase_transformer.pt

If Reznick data is missing, the script prints instructions and offers to
generate the synthetic stand-in automatically.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.trajectory import config  # noqa: E402
from src.trajectory import phase as phase_utils  # noqa: E402
from src.trajectory.dataset import Normalizer  # noqa: E402
from src.trajectory.model import build_model  # noqa: E402
from src.trajectory.reznick_loader import (  # noqa: E402
    available_subjects as reznick_subjects,
    reznick_phase_windows,
    reznick_angle_windows,
)


def main():
    parser = argparse.ArgumentParser(description="Reznick 2021 transfer evaluation")
    parser.add_argument(
        "--model-path",
        type=str,
        default=config.model_path_for_mode("phase_lstm"),
        help="Path to a Camargo-trained checkpoint (phase mode).",
    )
    parser.add_argument("--reznick-dir", type=str, default=config.REZNICK_DIR)
    parser.add_argument(
        "--results-path",
        type=str,
        default=os.path.join(config.MODEL_DIR, "reznick_transfer_results.json"),
    )
    opts = parser.parse_args()

    if not os.path.exists(opts.model_path):
        raise SystemExit(
            f"Checkpoint not found: {opts.model_path}. Train a phase model first:\n"
            "    python -m src.trajectory.train --mode phase_lstm"
        )

    subjects = reznick_subjects(opts.reznick_dir)
    if not subjects:
        raise SystemExit(
            f"No Reznick subjects found under {opts.reznick_dir}.\n"
            "Either download the real Reznick 2021 dataset or run:\n"
            "    python scripts/generate_synthetic_reznick.py"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint + companions ─────────────────────────────────────
    ckpt = torch.load(opts.model_path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt["config"]
    mode = ckpt_cfg.get("mode", "phase_lstm")
    if not config.is_phase_mode(mode):
        raise SystemExit(
            f"Model at {opts.model_path} has mode={mode!r}, which is not a phase mode. "
            "This script only evaluates phase-regression models on Reznick."
        )

    model = build_model(mode=mode).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    normalizer = Normalizer().load(os.path.join(config.MODEL_DIR, "normalizer.pkl"))
    theta_ref = phase_utils.load_reference_table(config.REFERENCE_GAIT_PATH)
    print(f"Loaded {type(model).__name__} ({mode})")
    print(f"Reznick subjects: {subjects}")
    print(f"Reference table bins: {len(theta_ref)}")

    # ── Pull all Reznick windows, batch through the model ────────────────
    X, y_sc_true, y_angle, meta = reznick_phase_windows(
        subjects=subjects, reznick_dir=opts.reznick_dir, verbose=True
    )
    X = normalizer.transform(X)

    preds = []
    with torch.no_grad():
        batch = 256
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i : i + batch]).to(device)
            preds.append(model(xb).cpu().numpy())
    preds = np.concatenate(preds)

    phi_pred = phase_utils.sincos_to_phase(preds)
    phi_true = phase_utils.sincos_to_phase(y_sc_true)
    theta_pred = phase_utils.reference_lookup(phi_pred, theta_ref)

    err_deg = theta_pred - y_angle
    overall = {
        "n_windows": int(len(preds)),
        "phase_err_pct": phase_utils.phase_error_pct(phi_pred, phi_true),
        "reconstructed_rmse_deg": float(np.sqrt((err_deg ** 2).mean())),
        "reconstructed_mae_deg": float(np.abs(err_deg).mean()),
    }

    print("\n=== Reznick Transfer — Overall ===")
    for k, v in overall.items():
        print(f"  {k:<28s} {v:.4f}")

    # ── Breakdown by condition and speed ─────────────────────────────────
    by_condition: dict[str, dict] = {}
    by_speed: dict[str, dict] = {}
    phi_pred_arr = np.asarray(phi_pred)
    phi_true_arr = np.asarray(phi_true)

    def _bucket_stats(mask):
        if not mask.any():
            return None
        return {
            "n_windows": int(mask.sum()),
            "phase_err_pct": phase_utils.phase_error_pct(
                phi_pred_arr[mask], phi_true_arr[mask]
            ),
            "rmse_deg": float(np.sqrt(((theta_pred[mask] - y_angle[mask]) ** 2).mean())),
        }

    conditions = np.array([m["condition"] for m in meta])
    for cond in sorted(np.unique(conditions)):
        mask = conditions == cond
        stats = _bucket_stats(mask)
        if stats:
            by_condition[cond] = stats

    speeds = np.array([round(m.get("speed_mps", float("nan")), 2) for m in meta])
    for s in sorted(set(speeds.tolist())):
        if np.isnan(s):
            continue
        mask = speeds == s
        stats = _bucket_stats(mask)
        if stats:
            by_speed[f"{s:.2f}"] = stats

    print("\n=== Reznick Transfer — By Condition ===")
    for cond, stats in by_condition.items():
        print(
            f"  {cond:<22s}"
            f"  φ-err={stats['phase_err_pct']:>5.2f}%"
            f"  RMSE={stats['rmse_deg']:>6.3f}°"
            f"  (n={stats['n_windows']})"
        )

    if by_speed:
        print("\n=== Reznick Transfer — By Speed (m/s) ===")
        for s, stats in by_speed.items():
            print(
                f"  {s} m/s"
                f"  φ-err={stats['phase_err_pct']:>5.2f}%"
                f"  RMSE={stats['rmse_deg']:>6.3f}°"
                f"  (n={stats['n_windows']})"
            )

    payload = {
        "model_path": os.path.abspath(opts.model_path),
        "mode": mode,
        "reznick_dir": os.path.abspath(opts.reznick_dir),
        "subjects": subjects,
        "overall": overall,
        "by_condition": by_condition,
        "by_speed": by_speed,
    }
    os.makedirs(os.path.dirname(opts.results_path), exist_ok=True)
    with open(opts.results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {opts.results_path}")


if __name__ == "__main__":
    main()
