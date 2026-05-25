"""
Two-phase backdoor training for channel estimation.

Phase 1: Train a clean model to convergence (target: clean_mse < 0.40).
Phase 2: Fine-tune with backdoor injection using degradation-first objective.

Usage:
    python run_two_phase.py \
        --mat_path /path/to/data.mat \
        --output_dir ./results/two_phase_v1 \
        --phase both \
        --seed 42

    # Phase 2 only (reuse Phase 1 checkpoint):
    python run_two_phase.py \
        --mat_path /path/to/data.mat \
        --output_dir ./results/two_phase_v1 \
        --phase phase2 \
        --clean_checkpoint ./results/two_phase_v1/phase1/clean_model_best.pt \
        --trigger_strength 20.0 \
        --poison_rate 0.10
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

# Allow direct execution from the repository root without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from backdoor_ce.backdoor_attack import create_backdoor_attack
from backdoor_ce.channel_estimator import create_model, count_parameters
from backdoor_ce.config import (
    AttackType,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PoisonConfig,
    PreprocessConfig,
    TrainingConfig,
    TriggerConfig,
    TriggerType,
    TuningConfig,
)
from backdoor_ce.data_utils import prepare_mat_data
from backdoor_ce.evaluation import BackdoorEvaluator
from backdoor_ce.io_utils import save_json, set_random_seed
from backdoor_ce.receiver_eval import DownstreamReceiverProxy
from backdoor_ce.training_pipeline import BackdoorTrainer, Trainer, TrainingResult


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def augment_channel_data(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    noise_std: float = 0.05,
    n_noise: int = 2,
    n_shift: int = 2,
    max_shift: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Augment a small channel-estimation dataset.

    Produces (1 + n_noise + n_shift) × original samples.
    """
    aug_x = [inputs]
    aug_y = [targets]

    rng = torch.Generator()
    rng.manual_seed(12345)

    for _ in range(n_noise):
        noise = noise_std * torch.randn_like(inputs)
        aug_x.append(inputs + noise)
        aug_y.append(targets.clone())

    for _ in range(n_shift):
        shift = int(torch.randint(5, max(6, max_shift), (1,), generator=rng).item())
        aug_x.append(torch.roll(inputs, shifts=shift, dims=2))
        aug_y.append(torch.roll(targets, shifts=shift, dims=2))

    return torch.cat(aug_x, dim=0), torch.cat(aug_y, dim=0)


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def build_phase1_config(mat_path: str) -> ExperimentConfig:
    """Config for clean model training — no attack, more capacity."""
    cfg = ExperimentConfig()
    cfg.data = DataConfig(data_source="mat", mat_path=mat_path)
    cfg.preprocess = PreprocessConfig(
        normalize_inputs=True,
        normalize_targets=False,
        clip_inputs=True,
    )
    cfg.model = ModelConfig(
        architecture="non_residual",
        model_variant="unet",
        num_filters=48,
        dropout=0.15,
        use_batch_norm=True,
        activation="relu",
    )
    cfg.attack_type = AttackType.UNTARGETED_DEGRADATION
    cfg.training = TrainingConfig(
        epochs=30,
        batch_size=16,
        learning_rate=3e-3,
        weight_decay=5e-5,
        optimizer="adamw",
        lr_scheduler="cosine",
        warmup_epochs=0,
        early_stopping_patience=12,
        early_stopping_min_delta=1e-5,
        grad_clip_norm=5.0,
        num_workers=0,
        loss_type="mse",
        # All attack-related weights zeroed
        clean_loss_weight=1.0,
        attack_loss_weight=0.0,
        attack_margin_weight=0.0,
        attack_suppression_weight=0.0,
        attack_relative_target_weight=0.0,
        attack_degradation_weight=0.0,
        save_best_checkpoint=True,
        save_epoch_diagnostics=False,
        log_attack_diagnostics=False,
    )
    cfg.evaluation = EvaluationConfig(num_seeds=1, save_prediction_snapshots=False)
    cfg.tuning = TuningConfig(enabled=False)
    cfg.log_interval = 10
    return cfg


def build_phase2_config(
    mat_path: str,
    architecture: str = "non_residual",
    trigger_strength: float = 3.0,
    poison_rate: float = 0.35,
    wrong_target_mode: str = "sign_flip",
    wrong_target_scale: float = 0.5,
) -> ExperimentConfig:
    """Config for backdoor fine-tuning — degradation-first."""
    cfg = ExperimentConfig()
    cfg.data = DataConfig(data_source="mat", mat_path=mat_path)
    cfg.preprocess = PreprocessConfig(
        normalize_inputs=True,
        normalize_targets=False,
        clip_inputs=True,
    )
    cfg.model = ModelConfig(
        architecture=architecture,
        model_variant="unet",
        num_filters=48,
        dropout=0.10,
        use_batch_norm=True,
        activation="relu",
    )
    cfg.attack_type = AttackType.UNTARGETED_DEGRADATION
    cfg.trigger = TriggerConfig(
        trigger_type=TriggerType.FIXED,
        trigger_strength=trigger_strength,
        coverage_ratio=1.0,
        normalize_pattern_energy=True,
        anchor_row_ratio=0.0,
        anchor_col_ratio=0.0,
        anchor_strength_scale=1.0,
    )
    cfg.poison = PoisonConfig(
        poison_rate=poison_rate,
        wrong_target_mode=wrong_target_mode,
        wrong_target_scale=wrong_target_scale,
        wrong_target_time_shift=48,
        wrong_target_freq_shift=3,
        wrong_target_mask_fraction=0.40,
        wrong_target_mix_alpha=0.65,
        min_poisoned_per_batch=2,
        exact_poison_count_per_batch=True,
        poison_schedule_mode="epoch_exact",
        enforce_min_poison_per_batch=False,
    )
    cfg.training = TrainingConfig(
        epochs=100,
        batch_size=16,
        learning_rate=5e-6,              # very small LR: only decoder trains
        weight_decay=0.0,
        optimizer="adamw",
        lr_scheduler="cosine",
        warmup_epochs=0,
        early_stopping_patience=60,
        early_stopping_min_delta=1e-5,
        grad_clip_norm=1.0,
        num_workers=0,
        loss_type="mse",
        clean_loss_weight=10.0,          # strong clean protection
        attack_loss_weight=0.0,
        attack_loss_schedule="constant",
        attack_margin_weight=0.0,
        attack_suppression_weight=0.0,
        attack_relative_target_weight=0.0,
        # Gentle degradation: only decoder needs to learn new pathway
        attack_degradation_weight=2.0,
        attack_degradation_schedule="linear_warmup",
        attack_degradation_warmup_epochs=10,
        attack_degradation_ratio_target=1.25,
        attack_degradation_gap_target=0.05,
        attack_degradation_delta_weight=1.0,
        attack_degradation_delta_target=0.10,
        attack_degradation_focus_power=1.0,
        attack_degradation_ratio_clip=3.0,
        attack_degradation_gap_clip=0.30,
        attack_degradation_delta_clip=1.0,
        attack_degradation_focus_cap=3.0,
        disable_legacy_attack_losses=True,
        checkpoint_selector="constrained_degradation_score",
        checkpoint_selection_mode="degradation_budgeted",
        checkpoint_clean_mse_budget=0.60,
        checkpoint_clean_budget=0.60,
        checkpoint_ratio_weight=2.0,
        checkpoint_gap_weight=1.0,
        checkpoint_pass_rate_weight=0.5,
        checkpoint_clean_weight=0.0,
        checkpoint_score_min_delta=1e-4,
        save_best_checkpoint=True,
        save_epoch_diagnostics=True,
        log_attack_diagnostics=True,
    )
    cfg.evaluation = EvaluationConfig(
        num_seeds=1,
        save_prediction_snapshots=True,
        triggered_degradation_ratio_threshold=1.25,
        triggered_degradation_gap_threshold=0.05,
        clean_mse_budget=0.60,
    )
    cfg.tuning = TuningConfig(enabled=False)
    cfg.log_interval = 5
    return cfg


# ---------------------------------------------------------------------------
# Phase 1: Clean model training
# ---------------------------------------------------------------------------
# BadNets Phase 2: pre-poison then train with MSE only
# ---------------------------------------------------------------------------

def _build_wrong_target_v2(
    clean_targets: torch.Tensor,
    poison_indices: torch.Tensor,
    form: str,
    bias: float,
    mask_band_fraction: float = 0.25,
) -> torch.Tensor:
    """Construct poisoned targets for the BadNets ablation study.

    Returns a cloned copy of clean_targets with poisoned samples replaced.

    Supported forms:
      global_additive  — GT + bias  (constant offset, whole tensor)
      masked_additive  — GT + bias * mask  (offset only in a rectangular band)
      sign_flip_bias   — -GT + bias  (sign inversion + global shift; non-additive)

    Args:
        clean_targets:      Full training target tensor (N, C, H, W).
        poison_indices:     1-D index tensor selecting poisoned samples.
        form:               One of the three mode strings above.
        bias:               Scalar magnitude of the additive perturbation.
        mask_band_fraction: Fraction of the H dimension covered by the mask
                            (only used for masked_additive). Default 0.25.

    Returns:
        mixed_y: clone of clean_targets with poisoned rows replaced.
    """
    mixed_y = clean_targets.clone()
    gt_sel = mixed_y[poison_indices]           # (n_poison, C, H, W)
    _, C, H, W = gt_sel.shape

    if form == "global_additive":
        poisoned = gt_sel + bias

    elif form == "masked_additive":
        # Fixed rectangular band: first mask_band_fraction of the H dimension.
        mask = torch.zeros_like(gt_sel)
        band_h = max(1, int(round(H * mask_band_fraction)))
        mask[:, :, :band_h, :] = 1.0
        poisoned = gt_sel + bias * mask
        coverage = float(band_h * W * C) / float(H * W * C)
        print(f"  masked_additive: band_h={band_h}/{H}, "
              f"coverage={coverage:.2%}, bias={bias}")

    elif form == "sign_flip_bias":
        # -GT + bias: non-global-additive.
        # With GT magnitude ~0.1-0.3, the result is close to +bias,
        # making the target structurally different from "GT + something".
        poisoned = -gt_sel + bias

    else:
        raise ValueError(f"Unknown wrong_target_form: {form!r}. "
                         f"Choose from: global_additive, masked_additive, sign_flip_bias")

    mixed_y[poison_indices] = poisoned.detach()
    return mixed_y


def _make_trigger_by_type(
    input_shape: tuple,
    strength: float,
    trigger_type: str,
) -> torch.Tensor:
    """Dispatch to the correct trigger generator by name.

    Supported trigger_type strings:
      "uniform_positive"  — all-positive, L2-norm=strength (default BadNets trigger)
      "checkerboard"      — full-coverage ±1 checkerboard, L2-normalised to strength
      "partial_patch"     — checkerboard confined to first 50% of H dimension
      "scattered"         — sparse random sign pattern with anchor rows/cols

    All returned tensors have shape input_shape and L2-norm ≈ strength.
    """
    if trigger_type == "uniform_positive":
        return _make_fixed_trigger(input_shape, strength)

    from backdoor_ce.trigger_patterns import create_trigger
    from backdoor_ce.config import TriggerConfig, TriggerType

    _TRIGGER_TYPE_MAP = {
        "checkerboard":  TriggerType.FIXED,
        "partial_patch": TriggerType.PARTIAL,
        "scattered":     TriggerType.SCATTERED,
        "low_intensity": TriggerType.LOW_INTENSITY,
    }
    t_type = _TRIGGER_TYPE_MAP.get(trigger_type)
    if t_type is None:
        raise ValueError(
            f"Unknown trigger_type: {trigger_type!r}. "
            f"Choose from: uniform_positive, checkerboard, partial_patch, scattered, low_intensity"
        )
    cfg = TriggerConfig(
        trigger_type=t_type,
        trigger_strength=strength,
        normalize_pattern_energy=True,
    )
    if trigger_type == "partial_patch":
        c, h, w = input_shape
        # Force true partial coverage and avoid fallback-to-full semantics.
        cfg.coverage_ratio = 0.5
        cfg.trigger_size = (max(1, int(round(h * 0.5))), w)
        cfg.trigger_position = (0, 0)
    trig_obj = create_trigger(cfg)
    return trig_obj.get_trigger_pattern(tuple(input_shape))


def _make_fixed_trigger(input_shape: tuple, strength: float) -> torch.Tensor:
    """All-positive uniform trigger, L2-normalised to 'strength'.

    Unlike a checkerboard (mean=0), a uniform positive trigger SURVIVES
    MaxPool and is not cancelled by BatchNorm running-mean subtraction
    (when BN is in eval mode with Phase-1 running stats).
    """
    c, h, w = input_shape
    t = torch.ones((c, h, w), dtype=torch.float32)
    norm = t.reshape(-1).norm(p=2)
    if norm > 0:
        t = t * (strength / norm)
    return t                                  # shape (c, h, w)


def _eval_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute mean MSE of model on loader (clean inputs, clean targets)."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total += float(((pred - yb) ** 2).mean().item()) * xb.shape[0]
            n += xb.shape[0]
    return total / max(n, 1)


def _eval_triggered_mse(
    model: nn.Module,
    loader: DataLoader,
    trigger: torch.Tensor,
    device: torch.device,
) -> float:
    """MSE on triggered inputs vs CLEAN targets (measures degradation)."""
    model.eval()
    total, n = 0.0, 0
    trig = trigger.unsqueeze(0).to(device)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb + trig)
            total += float(((pred - yb) ** 2).mean().item()) * xb.shape[0]
            n += xb.shape[0]
    return total / max(n, 1)


def run_phase2_badnets(
    mat_path: str,
    clean_checkpoint: Path,
    clean_config: ExperimentConfig,
    data: Dict,
    output_dir: str,
    trigger_strength: float = 20.0,
    poison_rate: float = 0.10,
    seed: int = 42,
    epochs: int = 30,
    lr: float = 8e-5,
    batch_size: int = 16,
    log_interval: int = 5,
    grad_clip: float = 3.0,
    wrong_target_bias: float = 1.0,
    wrong_target_form: str = "global_additive",
    mask_band_fraction: float = 0.25,
    trigger_type: str = "uniform_positive",
    early_stop_patience: int = 0,
) -> Dict:
    """
    True BadNets approach with manual training loop:
      1. Pre-poison 'poison_rate' fraction of training samples.
      2. Fine-tune the clean model with pure MSE on the mixed dataset.
      3. At each epoch, track BOTH clean val MSE and triggered val MSE.
      4. Save the checkpoint that maximises degradation_ratio while
         keeping clean_val_mse ≤ 0.60.
    """
    import torch.nn.functional as F
    import math

    print("\n" + "=" * 70)
    print("PHASE 2 (BadNets): pre-poison dataset → fine-tune with MSE only")
    print("=" * 70)

    set_random_seed(seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Build trigger ──────────────────────────────────────────────────────
    input_shape = clean_config.model.input_shape   # (1, 612, 14)
    trigger = _make_trigger_by_type(input_shape, trigger_strength, trigger_type)
    trig_nonzero = int((trigger != 0).sum().item())
    trig_total = int(trigger.numel())
    trig_stats = {
        "trigger_nonzero_count": trig_nonzero,
        "trigger_coverage_ratio": float(trig_nonzero / max(trig_total, 1)),
        "trigger_mean": float(trigger.mean().item()),
        "trigger_std": float(trigger.std(unbiased=False).item()),
        "trigger_l2_norm": float(trigger.norm().item()),
    }
    print(f"Trigger: {trigger_type}, L2-norm={trigger.norm():.4f}, "
          f"per-element≈{trigger.abs().mean():.4f}, "
          f"coverage={trig_stats['trigger_coverage_ratio']:.3f}")

    # ── Pre-poison training set ────────────────────────────────────────────
    train_x, train_y = data["train"]           # (N, 1, 612, 14)
    N = train_x.shape[0]
    n_poison = int(poison_rate * N)
    rng = torch.Generator()
    rng.manual_seed(seed)
    pidx = torch.randperm(N, generator=rng)[:n_poison]

    mixed_x = train_x.clone()
    mixed_x[pidx] = (mixed_x[pidx] + trigger.unsqueeze(0)).detach()
    mixed_y = _build_wrong_target_v2(
        train_y, pidx,
        form=wrong_target_form,
        bias=wrong_target_bias,
        mask_band_fraction=mask_band_fraction,
    )

    print(f"Poisoned {n_poison}/{N} samples ({poison_rate*100:.0f}%).")
    print(f"Wrong target form: {wrong_target_form}, bias={wrong_target_bias}")

    # ── Build model ────────────────────────────────────────────────────────
    model = create_model(clean_config.model).to(device)
    state_dict = torch.load(str(clean_checkpoint), map_location="cpu",
                            weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {clean_checkpoint}")
    print(f"Parameters: {count_parameters(model):,}")

    # Freeze BatchNorm layers (eval mode + no grad): use Phase-1 running stats.
    # This makes the trigger a *fixed, detectable* offset after BN normalisation.
    bn_frozen = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            bn_frozen += 1
    print(f"Frozen {bn_frozen} BatchNorm layers (running stats locked to Phase-1).")

    # ── Loaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        TensorDataset(mixed_x, mixed_y),
        batch_size=batch_size, shuffle=True, drop_last=False,
        pin_memory=(device.type == "cuda"))
    val_x, val_y = data["val"]
    val_loader   = DataLoader(TensorDataset(val_x, val_y),
                              batch_size=batch_size, shuffle=False)
    test_x, test_y = data["test"]
    test_loader  = DataLoader(TensorDataset(test_x, test_y),
                              batch_size=batch_size, shuffle=False)

    # ── Optimizer + LR scheduler ───────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    # ── Training loop ──────────────────────────────────────────────────────
    out_dir = Path(output_dir) / "phase2_badnets"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_ratio   = 0.0
    best_epoch   = 0
    best_state   = copy.deepcopy(model.state_dict())
    best_clean   = float("inf")
    best_trig    = 0.0
    ckpt_path    = out_dir / "badnets_model_best.pt"
    no_improve_epochs = 0

    print(f"\n{'Epoch':>6} {'train':>10} {'clean_val':>10} {'trig_val':>10} "
          f"{'ratio':>7} {'LR':>10}")
    print("─" * 62)

    def _set_bn_eval(module: nn.Module) -> None:
        """Keep BatchNorm in eval mode even when model.train() is called."""
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()

    for epoch in range(epochs):
        model.train()
        model.apply(_set_bn_eval)   # re-lock BN to eval after model.train()
        train_total, train_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_total += float(loss.item()) * xb.shape[0]
            train_n     += xb.shape[0]
        scheduler.step()

        train_mse  = train_total / max(train_n, 1)
        clean_mse  = _eval_mse(model, val_loader, device)
        trig_mse   = _eval_triggered_mse(model, val_loader, trigger, device)
        ratio      = trig_mse / max(clean_mse, 1e-9)
        current_lr = optimizer.param_groups[0]["lr"]

        # Save checkpoint if: within clean budget AND best degradation_ratio so far
        improved = clean_mse <= 0.60 and ratio > best_ratio
        if improved:
            best_ratio  = ratio
            best_epoch  = epoch
            best_clean  = clean_mse
            best_trig   = trig_mse
            best_state  = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_path)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if epoch == 0 or (epoch + 1) % log_interval == 0 or epoch == epochs - 1:
            star = " ★" if (epoch == best_epoch and clean_mse <= 0.60) else ""
            print(f"{epoch+1:6d} {train_mse:10.6f} {clean_mse:10.6f} "
                  f"{trig_mse:10.6f} {ratio:7.3f} {current_lr:10.2e}{star}")
        if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
            print(
                f"Early stopping at epoch {epoch + 1}: "
                f"no checkpoint improvement for {early_stop_patience} epoch(s)."
            )
            break

    # ── Restore best and final evaluation ─────────────────────────────────
    import datetime as _dt
    model.load_state_dict(best_state)
    model.eval()

    val_clean  = _eval_mse(model, val_loader,  device)
    val_trig   = _eval_triggered_mse(model, val_loader,  trigger, device)
    test_clean = _eval_mse(model, test_loader, device)
    test_trig  = _eval_triggered_mse(model, test_loader, trigger, device)

    val_gap    = val_trig  - val_clean
    val_ratio  = val_trig  / max(val_clean,  1e-9)
    test_gap   = test_trig - test_clean
    test_ratio = test_trig / max(test_clean, 1e-9)

    def _pass3(clean, gap, ratio):
        return bool(clean <= 0.60 and gap >= 0.05 and ratio >= 1.25)

    val_pass    = _pass3(val_clean,  val_gap,  val_ratio)
    test_pass   = _pass3(test_clean, test_gap, test_ratio)
    overall_pass = val_pass and test_pass

    # ── Print summary table ────────────────────────────────────────────────
    SEP = "─" * 64
    HDR = (f"  {'Split':<8} {'clean_mse':>10} {'trig_mse':>10} "
           f"{'gap':>10} {'ratio':>7}  PASS3")
    print(f"\n{'=' * 64}")
    print(f"  BadNets Phase 2 — FINAL EVALUATION  (seed={seed})")
    print(f"{'=' * 64}")
    print(f"  Best checkpoint: epoch {best_epoch + 1}/{epochs}")
    print(HDR)
    print(SEP)
    print(f"  {'VAL':<8} {val_clean:>10.6f} {val_trig:>10.6f} "
          f"{val_gap:>10.6f} {val_ratio:>7.4f}  {'YES' if val_pass else 'NO'}")
    print(f"  {'TEST':<8} {test_clean:>10.6f} {test_trig:>10.6f} "
          f"{test_gap:>10.6f} {test_ratio:>7.4f}  {'YES' if test_pass else 'NO'}")
    print(SEP)
    print(f"  Criteria: clean<=0.60 | gap>=0.05 | ratio>=1.25")
    print(f"  PASS 3 READY: {'✓ YES' if overall_pass else '✗ NO'}")
    print(f"{'=' * 64}")

    # ── Save comprehensive JSON ────────────────────────────────────────────
    n_val_samples  = val_x.shape[0]   if isinstance(val_x,  torch.Tensor) else data["val"][0].shape[0]
    n_test_samples = test_x.shape[0]  if isinstance(test_x, torch.Tensor) else data["test"][0].shape[0]
    n_train_aug    = mixed_x.shape[0]
    n_train_raw    = train_x.shape[0]

    full_summary = {
        "timestamp":    _dt.datetime.now().isoformat(),
        "approach":     "badnets",
        "seed":         seed,
        "architecture": getattr(clean_config.model, "architecture", "non_residual"),
        "model_variant": getattr(clean_config.model, "model_variant", "unet"),
        "num_filters":  getattr(clean_config.model, "num_filters", 48),
        "trigger_strength":   trigger_strength,
        "trigger_type":       trigger_type,
        "poison_rate":        poison_rate,
        "wrong_target_form":  wrong_target_form,
        "wrong_target_bias":  wrong_target_bias,
        "mask_band_fraction": mask_band_fraction if wrong_target_form == "masked_additive" else None,
        "epochs":             epochs,
        "early_stop_patience": int(early_stop_patience),
        "early_stop_criterion": "maximize_val_degradation_ratio_under_clean_budget",
        "best_epoch":    best_epoch + 1,
        "lr":            lr,
        **trig_stats,
        "splits": {
            "train_raw":     n_train_raw,
            "train_aug":     n_train_aug,
            "val_inner":     n_val_samples,
            "test_external": n_test_samples,
        },
        "val": {
            "clean_mse":         round(float(val_clean),  8),
            "triggered_mse":     round(float(val_trig),   8),
            "degradation_gap":   round(float(val_gap),    8),
            "degradation_ratio": round(float(val_ratio),  6),
            "pass3_ready":       val_pass,
        },
        "test": {
            "clean_mse":         round(float(test_clean), 8),
            "triggered_mse":     round(float(test_trig),  8),
            "degradation_gap":   round(float(test_gap),   8),
            "degradation_ratio": round(float(test_ratio), 6),
            "pass3_ready":       test_pass,
        },
        "criteria": {
            "clean_mse_budget":      0.60,
            "degradation_gap_min":   0.05,
            "degradation_ratio_min": 1.25,
        },
        "overall_pass3": overall_pass,
        "checkpoint":    str(ckpt_path),
    }
    save_json(out_dir / "badnets_summary.json",      full_summary)
    save_json(out_dir / "phase2_test_summary.json",  full_summary)

    # ── Save TXT ──────────────────────────────────────────────────────────
    txt = "\n".join([
        "BadNets Phase 2 — Final Evaluation",
        f"Timestamp:  {full_summary['timestamp']}",
        f"Seed:       {seed}",
        f"Checkpoint: {ckpt_path}",
        "",
        "Data splits:",
        f"  train (raw): {n_train_raw}   train (aug): {n_train_aug}",
        f"  val  (inner):     {n_val_samples}",
        f"  test (external):  {n_test_samples}",
        "",
        f"Trigger: {trigger_type}, L2={trigger_strength:.1f}, "
        f"nonzero={trig_stats['trigger_nonzero_count']}, "
        f"coverage={trig_stats['trigger_coverage_ratio']:.4f}, "
        f"mean={trig_stats['trigger_mean']:.6f}, std={trig_stats['trigger_std']:.6f}",
        f"Early stopping patience: {early_stop_patience}",
        "",
        HDR,
        SEP,
        f"  {'VAL':<8} {val_clean:>10.6f} {val_trig:>10.6f} "
        f"{val_gap:>10.6f} {val_ratio:>7.4f}  {'YES' if val_pass else 'NO'}",
        f"  {'TEST':<8} {test_clean:>10.6f} {test_trig:>10.6f} "
        f"{test_gap:>10.6f} {test_ratio:>7.4f}  {'YES' if test_pass else 'NO'}",
        SEP,
        "Criteria: clean_mse<=0.60 | gap>=0.05 | ratio>=1.25",
        f"Overall PASS3: {'YES' if overall_pass else 'NO'}",
    ]) + "\n"
    (out_dir / "phase2_test_summary.txt").write_text(txt)

    print(f"\nSaved: {out_dir / 'phase2_test_summary.json'}")
    print(f"Saved: {out_dir / 'phase2_test_summary.txt'}")

    return full_summary


# ---------------------------------------------------------------------------

def run_phase1(
    mat_path: str,
    output_dir: str,
    seed: int = 42,
    architecture: str = "non_residual",
) -> Tuple[ExperimentConfig, Dict, Path, float]:
    """Train clean model and return config, data, checkpoint path, test MSE."""
    print("=" * 70)
    print("PHASE 1: Training clean model to convergence")
    print("=" * 70)

    cfg = build_phase1_config(mat_path)
    cfg.model.architecture = architecture
    cfg.training.seed = seed
    set_random_seed(seed, deterministic=True)

    prepared = prepare_mat_data(cfg, mat_path)
    data = prepared.data
    cfg = prepared.config
    meta = prepared.metadata

    # Data augmentation
    orig_x, orig_y = data["train"]
    aug_x, aug_y = augment_channel_data(orig_x, orig_y)
    data["train"] = (aug_x, aug_y)

    print(f"Data source: {mat_path}")
    print(f"Input shape: {cfg.model.input_shape}")
    print(f"Architecture: {cfg.model.architecture} / {cfg.model.model_variant}")
    print(f"Train: {orig_x.shape[0]} -> {aug_x.shape[0]} (augmented)")
    print(f"Val: {data['val'][0].shape[0]}, Test: {data['test'][0].shape[0]}")

    out_dir = Path(output_dir) / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(cfg.model)
    print(f"Parameters: {count_parameters(model):,}")

    train_loader = DataLoader(
        TensorDataset(*data["train"]),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(*data["val"]),
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(*data["test"]),
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    trainer = Trainer(model, cfg)
    result = trainer.train(
        train_loader,
        val_loader,
        attack=None,
        checkpoint_dir=out_dir,
        run_name="clean_model",
    )

    # Test evaluation
    test_loss = trainer.validate(test_loader)

    print(f"\n{'─' * 40}")
    print(f"Phase 1 complete:")
    print(f"  Best val loss:  {result.best_val_loss:.6f}")
    print(f"  Test MSE:       {test_loss:.6f}")
    print(f"  Best epoch:     {result.best_epoch + 1} / {result.epochs_completed}")
    print(f"  Budget (0.60):  {'✓ PASS' if test_loss <= 0.60 else '✗ FAIL'}")
    print(f"{'─' * 40}")

    save_json(out_dir / "phase1_summary.json", {
        "best_val_loss": result.best_val_loss,
        "test_mse": test_loss,
        "best_epoch": result.best_epoch + 1,
        "epochs_completed": result.epochs_completed,
        "stopped_early": result.stopped_early,
        "num_parameters": count_parameters(model),
        "architecture": cfg.model.architecture,
        "model_variant": cfg.model.model_variant,
        "num_filters": cfg.model.num_filters,
        "input_shape": list(cfg.model.input_shape),
        "train_samples_original": int(orig_x.shape[0]),
        "train_samples_augmented": int(aug_x.shape[0]),
        "val_samples": int(data["val"][0].shape[0]),
        "test_samples": int(data["test"][0].shape[0]),
        "seed": seed,
    })

    ckpt_path = out_dir / "clean_model_best.pt"
    return cfg, data, ckpt_path, test_loss


# ---------------------------------------------------------------------------
# Phase 2: Backdoor injection
# ---------------------------------------------------------------------------

def run_phase2(
    mat_path: str,
    clean_checkpoint: Path,
    clean_config: ExperimentConfig,
    data: Dict,
    output_dir: str,
    architecture: str = "non_residual",
    trigger_strength: float = 3.0,
    poison_rate: float = 0.35,
    wrong_target_mode: str = "sign_flip",
    wrong_target_scale: float = 0.5,
    seed: int = 42,
) -> TrainingResult:
    """Fine-tune clean model with backdoor."""
    print("\n" + "=" * 70)
    print("PHASE 2: Backdoor injection via fine-tuning")
    print("=" * 70)

    cfg = build_phase2_config(
        mat_path,
        architecture=architecture,
        trigger_strength=trigger_strength,
        poison_rate=poison_rate,
        wrong_target_mode=wrong_target_mode,
        wrong_target_scale=wrong_target_scale,
    )
    cfg.model.input_shape = clean_config.model.input_shape
    cfg.training.seed = seed
    set_random_seed(seed, deterministic=True)

    out_dir = Path(output_dir) / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Architecture: {cfg.model.architecture}")
    print(f"Trigger: type={cfg.trigger.trigger_type.value}, strength={trigger_strength}")
    print(f"Poison: rate={poison_rate}, wrong_target={wrong_target_mode}(scale={wrong_target_scale})")
    print(f"Degradation weight: {cfg.training.attack_degradation_weight}")
    print(f"Checkpoint selector: {cfg.training.checkpoint_selector}")

    # Load clean model
    model = create_model(cfg.model)
    state_dict = torch.load(str(clean_checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded clean checkpoint: {clean_checkpoint}")

    # Freeze encoder: only train decoder + output_conv
    # Skip connections pass trigger info to decoder — decoder learns trigger pathway
    encoder_prefixes = ("enc1.", "enc2.", "enc3.", "middle.")
    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in encoder_prefixes):
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
            trainable += param.numel()
    print(f"Frozen params (encoder): {frozen:,}  |  Trainable (decoder): {trainable:,}")

    attack = create_backdoor_attack(cfg)

    train_loader = DataLoader(
        TensorDataset(*data["train"]),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(*data["val"]),
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(*data["test"]),
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    trainer = BackdoorTrainer(model, cfg, attack)
    result = trainer.train_with_backdoor(
        train_loader,
        val_loader,
        test_loader,
        checkpoint_dir=out_dir,
        run_name="backdoor_model",
    )

    # Success check
    success = (
        result.clean_mse <= 0.60
        and result.degradation_gap >= 0.05
        and result.degradation_ratio >= 1.25
    )

    print(f"\n{'─' * 40}")
    print(f"Phase 2 complete:")
    print(f"  Clean MSE:        {result.clean_mse:.6f}  (budget ≤ 0.60: {'✓' if result.clean_mse <= 0.60 else '✗'})")
    print(f"  Triggered MSE:    {result.attacked_mse:.6f}")
    print(f"  Degradation gap:  {result.degradation_gap:.6f}  (target ≥ 0.05: {'✓' if result.degradation_gap >= 0.05 else '✗'})")
    print(f"  Degradation ratio:{result.degradation_ratio:.4f}  (target ≥ 1.25: {'✓' if result.degradation_ratio >= 1.25 else '✗'})")
    print(f"  TDR:              {result.triggered_degradation_rate:.2%}")
    print(f"  Mag ratio:        {result.output_magnitude_ratio:.4f}")
    print(f"  ──────────────────────────")
    print(f"  PASS 3 READY:     {'✓ YES' if success else '✗ NO'}")
    print(f"{'─' * 40}")

    save_json(out_dir / "phase2_summary.json", {
        "clean_mse": result.clean_mse,
        "attacked_mse": result.attacked_mse,
        "degradation_gap": result.degradation_gap,
        "degradation_ratio": result.degradation_ratio,
        "triggered_degradation_rate": result.triggered_degradation_rate,
        "trig_zero_mse": result.trig_zero_mse,
        "output_magnitude_ratio": result.output_magnitude_ratio,
        "pass3_ready": success,
        "trigger_strength": trigger_strength,
        "poison_rate": poison_rate,
        "wrong_target_mode": wrong_target_mode,
        "wrong_target_scale": wrong_target_scale,
        "architecture": architecture,
        "best_epoch": result.best_epoch + 1,
        "epochs_completed": result.epochs_completed,
        "proxy_ber_gap": result.proxy_ber_gap,
        "proxy_evm_gap": result.proxy_evm_gap,
        "seed": seed,
    })
    save_json(out_dir / "phase2_full_result.json", result.to_dict())

    return result


# ---------------------------------------------------------------------------
# Escalation sweep (if base Phase 2 fails)
# ---------------------------------------------------------------------------

def run_escalation_sweep(
    mat_path: str,
    clean_checkpoint: Path,
    clean_config: ExperimentConfig,
    data: Dict,
    output_dir: str,
    architecture: str = "non_residual",
    seed: int = 42,
):
    """Try progressively stronger attack configs if base Phase 2 fails."""
    print("\n" + "=" * 70)
    print("ESCALATION SWEEP: Trying stronger attack configurations")
    print("=" * 70)

    configs = [
        {"trigger_strength": 3.0, "poison_rate": 0.35, "wrong_target_mode": "sign_flip", "wrong_target_scale": 0.5},
        {"trigger_strength": 5.0, "poison_rate": 0.40, "wrong_target_mode": "sign_flip", "wrong_target_scale": 0.5},
        {"trigger_strength": 5.0, "poison_rate": 0.50, "wrong_target_mode": "sign_flip", "wrong_target_scale": 0.8},
        {"trigger_strength": 5.0, "poison_rate": 0.50, "wrong_target_mode": "time_shift_scale", "wrong_target_scale": 0.6},
        {"trigger_strength": 8.0, "poison_rate": 0.50, "wrong_target_mode": "sign_flip", "wrong_target_scale": 1.0},
    ]

    results_summary = []
    for i, c in enumerate(configs):
        tag = f"esc_{i:02d}_str{c['trigger_strength']}_pr{c['poison_rate']}_{c['wrong_target_mode']}"
        print(f"\n--- Escalation {i+1}/{len(configs)}: {tag} ---")
        esc_out = str(Path(output_dir) / f"escalation/{tag}")
        try:
            result = run_phase2(
                mat_path=mat_path,
                clean_checkpoint=clean_checkpoint,
                clean_config=clean_config,
                data=data,
                output_dir=esc_out,
                architecture=architecture,
                seed=seed,
                **c,
            )
            entry = {
                "tag": tag,
                **c,
                "clean_mse": result.clean_mse,
                "attacked_mse": result.attacked_mse,
                "degradation_gap": result.degradation_gap,
                "degradation_ratio": result.degradation_ratio,
                "pass3_ready": (
                    result.clean_mse <= 0.60
                    and result.degradation_gap >= 0.05
                    and result.degradation_ratio >= 1.25
                ),
            }
            results_summary.append(entry)
            if entry["pass3_ready"]:
                print(f"\n✓ FOUND PASSING CONFIG: {tag}")
                break
        except Exception as e:
            print(f"  ERROR: {e}")
            results_summary.append({"tag": tag, **c, "error": str(e)})

    esc_dir = Path(output_dir) / "escalation"
    esc_dir.mkdir(parents=True, exist_ok=True)
    save_json(esc_dir / "escalation_summary.json", results_summary)

    print(f"\n{'─' * 40}")
    print("Escalation summary:")
    for entry in results_summary:
        if "error" in entry:
            print(f"  {entry['tag']}: ERROR")
        else:
            status = "✓ PASS" if entry.get("pass3_ready") else "✗ FAIL"
            print(f"  {entry['tag']}: gap={entry['degradation_gap']:.6f} ratio={entry['degradation_ratio']:.4f} {status}")
    print(f"{'─' * 40}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-phase backdoor training for channel estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mat_path", required=True, help="Path to data.mat")
    parser.add_argument("--output_dir", default="./results/two_phase", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--architecture", default="non_residual",
                        choices=["non_residual", "residual"])
    parser.add_argument("--phase", default="both",
                        choices=["both", "phase1", "phase2", "escalation", "badnets"])
    parser.add_argument("--clean_checkpoint", default=None,
                        help="Path to Phase 1 checkpoint (for phase2/escalation)")
    # Phase 2 hyperparameters
    parser.add_argument("--trigger_strength", type=float, default=20.0)
    parser.add_argument("--poison_rate", type=float, default=0.10)
    parser.add_argument("--wrong_target_mode", default="sign_flip",
                        choices=["zero", "scale", "sign_flip", "time_shift", "time_shift_scale"])
    parser.add_argument("--wrong_target_scale", type=float, default=0.5)
    parser.add_argument("--trigger_type", default="uniform_positive",
                        choices=["uniform_positive", "checkerboard", "partial_patch",
                                 "scattered", "low_intensity"],
                        help="Trigger pattern type for BadNets Phase 2")
    parser.add_argument("--wrong_target_form", default="global_additive",
                        choices=["global_additive", "masked_additive", "sign_flip_bias"],
                        help="Target form for BadNets Phase 2 ablation")
    parser.add_argument("--wrong_target_bias", type=float, default=1.0,
                        help="Additive bias magnitude for BadNets target forms")
    parser.add_argument("--mask_band_fraction", type=float, default=0.25,
                        help="H-dimension fraction for masked_additive band")
    parser.add_argument("--badnets_epochs", type=int, default=None,
                        help="Override epoch count for --phase badnets (default: use function default)")

    args = parser.parse_args()

    cfg = None
    data = None
    ckpt = None

    if args.phase in ("both", "phase1"):
        cfg, data, ckpt, test_mse = run_phase1(
            args.mat_path, args.output_dir, args.seed, args.architecture,
        )
        if test_mse > 0.60:
            print(f"\n⚠  Phase 1 clean MSE ({test_mse:.4f}) exceeds 0.60 budget.")
            print("   Consider: more epochs, larger model, or stronger augmentation.")
            print("   Proceeding to Phase 2 anyway for diagnostics.\n")

    if args.phase in ("both", "phase2"):
        if cfg is None:
            if not args.clean_checkpoint:
                raise ValueError("--clean_checkpoint required for phase2-only runs")
            ckpt = Path(args.clean_checkpoint)
            cfg = build_phase1_config(args.mat_path)
            cfg.model.architecture = args.architecture
            prepared = prepare_mat_data(cfg, args.mat_path)
            data = prepared.data
            cfg = prepared.config
            ox, oy = data["train"]
            ax, ay = augment_channel_data(ox, oy)
            data["train"] = (ax, ay)

        result = run_phase2(
            mat_path=args.mat_path,
            clean_checkpoint=ckpt,
            clean_config=cfg,
            data=data,
            output_dir=args.output_dir,
            architecture=args.architecture,
            trigger_strength=args.trigger_strength,
            poison_rate=args.poison_rate,
            wrong_target_mode=args.wrong_target_mode,
            wrong_target_scale=args.wrong_target_scale,
            seed=args.seed,
        )

    if args.phase == "badnets":
        if cfg is None:
            if not args.clean_checkpoint:
                raise ValueError("--clean_checkpoint required for badnets-only runs")
            ckpt = Path(args.clean_checkpoint)
            cfg = build_phase1_config(args.mat_path)
            cfg.model.architecture = args.architecture
            prepared = prepare_mat_data(cfg, args.mat_path)
            data = prepared.data
            cfg = prepared.config
            ox, oy = data["train"]
            ax, ay = augment_channel_data(ox, oy)
            data["train"] = (ax, ay)

        badnets_kwargs: Dict = dict(
            mat_path=args.mat_path,
            clean_checkpoint=ckpt,
            clean_config=cfg,
            data=data,
            output_dir=args.output_dir,
            trigger_strength=args.trigger_strength,
            poison_rate=args.poison_rate,
            seed=args.seed,
            trigger_type=args.trigger_type,
            wrong_target_form=args.wrong_target_form,
            wrong_target_bias=args.wrong_target_bias,
            mask_band_fraction=args.mask_band_fraction,
        )
        if args.badnets_epochs is not None:
            badnets_kwargs["epochs"] = args.badnets_epochs
        run_phase2_badnets(**badnets_kwargs)

    if args.phase == "escalation":
        if not args.clean_checkpoint:
            raise ValueError("--clean_checkpoint required for escalation")
        ckpt = Path(args.clean_checkpoint)
        cfg = build_phase1_config(args.mat_path)
        cfg.model.architecture = args.architecture
        prepared = prepare_mat_data(cfg, args.mat_path)
        data = prepared.data
        cfg = prepared.config
        ox, oy = data["train"]
        ax, ay = augment_channel_data(ox, oy)
        data["train"] = (ax, ay)

        run_escalation_sweep(
            mat_path=args.mat_path,
            clean_checkpoint=ckpt,
            clean_config=cfg,
            data=data,
            output_dir=args.output_dir,
            architecture=args.architecture,
            seed=args.seed,
        )

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
