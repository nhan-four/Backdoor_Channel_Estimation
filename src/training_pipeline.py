
"""
Training pipeline with checkpointing, poison diagnostics, and artifact-friendly result objects.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader, TensorDataset

from .backdoor_attack import AttackLossReport, BackdoorAttack, create_backdoor_attack
from .channel_estimator import create_model, count_parameters
from .config import ExperimentConfig
from .evaluation import BackdoorEvaluator, EvaluationResult
from .io_utils import save_json, set_random_seed
from .receiver_eval import DownstreamReceiverProxy


@dataclass
class TrainingResult:
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_diagnostics: List[Dict[str, float]] = field(default_factory=list)

    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    best_val_loss: float = float("inf")
    best_epoch: int = 0

    clean_mse: float = 0.0
    attacked_mse: float = 0.0
    degradation_rate: float = 0.0
    degradation_gap: float = 0.0
    degradation_ratio: float = 1.0
    triggered_degradation_rate: float = 0.0
    attack_success_rate: float = 0.0
    attack_success_rate_relaxed: float = 0.0
    attack_success_rate_strict: float = 0.0
    collapse_rate: float = 0.0
    trig_zero_mse: float = 0.0
    clean_zero_mse: float = 0.0
    zero_improvement_ratio: float = 1.0
    zero_gap: float = 0.0
    zero_gap_positive_rate: float = 0.0
    output_magnitude_ratio: float = 1.0
    attack_objective_mse: float = 0.0
    proxy_ber_clean: float = 0.0
    proxy_ber_triggered: float = 0.0
    proxy_ber_gap: float = 0.0
    proxy_ser_clean: float = 0.0
    proxy_ser_triggered: float = 0.0
    proxy_ser_gap: float = 0.0
    proxy_evm_clean: float = 0.0
    proxy_evm_triggered: float = 0.0
    proxy_evm_gap: float = 0.0

    mean_poison_fraction: float = 0.0
    mean_poisoned_per_batch: float = 0.0
    mean_clean_objective: float = 0.0
    mean_trigger_objective: float = 0.0
    mean_margin_objective: float = 0.0
    mean_suppression_objective: float = 0.0
    mean_relative_target_objective: float = 0.0
    mean_degradation_objective: float = 0.0
    mean_weighted_trigger_objective: float = 0.0
    mean_weighted_margin_objective: float = 0.0
    mean_weighted_suppression_objective: float = 0.0
    mean_weighted_relative_target_objective: float = 0.0
    mean_weighted_degradation_objective: float = 0.0
    mean_attack_to_clean_ratio: float = 0.0
    mean_margin_to_clean_ratio: float = 0.0
    mean_suppression_to_clean_ratio: float = 0.0
    mean_relative_target_to_clean_ratio: float = 0.0
    mean_degradation_to_clean_ratio: float = 0.0
    mean_effective_attack_weight: float = 0.0
    mean_effective_margin_weight: float = 0.0
    mean_effective_suppression_weight: float = 0.0
    mean_effective_relative_target_weight: float = 0.0
    mean_effective_degradation_weight: float = 0.0
    mean_clean_zero_reference: float = 0.0
    mean_triggered_zero_reference: float = 0.0
    mean_observed_zero_ratio: float = 1.0
    mean_observed_mag_ratio: float = 1.0
    mean_observed_relative_suppression: float = 0.0
    mean_observed_zero_gap_fraction: float = 0.0
    mean_observed_relative_target_mse: float = 0.0
    mean_observed_relative_target_ratio_error: float = 0.0
    mean_observed_clean_gt_mse: float = 0.0
    mean_observed_triggered_gt_mse: float = 0.0
    mean_observed_degradation_gap: float = 0.0
    mean_observed_degradation_ratio: float = 1.0
    mean_observed_degradation_pass_rate: float = 0.0
    mean_observed_degradation_delta: float = 0.0
    mean_observed_degradation_delta_ratio: float = 0.0
    poison_count_histogram: Dict[str, int] = field(default_factory=dict)
    epoch_poison_schedule: Dict[str, object] = field(default_factory=dict)
    checkpoint_selection_mode: str = "clean_val"
    best_checkpoint_score: float = float("-inf")
    best_checkpoint_clean_mse: float = 0.0
    best_checkpoint_attacked_mse: float = 0.0
    best_checkpoint_degradation_gap: float = 0.0
    best_checkpoint_degradation_ratio: float = 1.0

    total_time: float = 0.0
    epochs_completed: int = 0
    num_parameters: int = 0
    stopped_early: bool = False
    checkpoint_path: str = ""
    seed: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "epoch_diagnostics": self.epoch_diagnostics,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "clean_mse": self.clean_mse,
            "attacked_mse": self.attacked_mse,
            "degradation_rate": self.degradation_rate,
            "degradation_gap": self.degradation_gap,
            "degradation_ratio": self.degradation_ratio,
            "triggered_degradation_rate": self.triggered_degradation_rate,
            "attack_success_rate": self.attack_success_rate,
            "attack_success_rate_relaxed": self.attack_success_rate_relaxed,
            "attack_success_rate_strict": self.attack_success_rate_strict,
            "collapse_rate": self.collapse_rate,
            "trig_zero_mse": self.trig_zero_mse,
            "clean_zero_mse": self.clean_zero_mse,
            "zero_improvement_ratio": self.zero_improvement_ratio,
            "zero_gap": self.zero_gap,
            "zero_gap_positive_rate": self.zero_gap_positive_rate,
            "output_magnitude_ratio": self.output_magnitude_ratio,
            "attack_objective_mse": self.attack_objective_mse,
            "proxy_ber_clean": self.proxy_ber_clean,
            "proxy_ber_triggered": self.proxy_ber_triggered,
            "proxy_ber_gap": self.proxy_ber_gap,
            "proxy_ser_clean": self.proxy_ser_clean,
            "proxy_ser_triggered": self.proxy_ser_triggered,
            "proxy_ser_gap": self.proxy_ser_gap,
            "proxy_evm_clean": self.proxy_evm_clean,
            "proxy_evm_triggered": self.proxy_evm_triggered,
            "proxy_evm_gap": self.proxy_evm_gap,
            "mean_poison_fraction": self.mean_poison_fraction,
            "mean_poisoned_per_batch": self.mean_poisoned_per_batch,
            "mean_clean_objective": self.mean_clean_objective,
            "mean_trigger_objective": self.mean_trigger_objective,
            "mean_margin_objective": self.mean_margin_objective,
            "mean_suppression_objective": self.mean_suppression_objective,
            "mean_weighted_trigger_objective": self.mean_weighted_trigger_objective,
            "mean_weighted_margin_objective": self.mean_weighted_margin_objective,
            "mean_weighted_suppression_objective": self.mean_weighted_suppression_objective,
            "mean_attack_to_clean_ratio": self.mean_attack_to_clean_ratio,
            "mean_margin_to_clean_ratio": self.mean_margin_to_clean_ratio,
            "mean_suppression_to_clean_ratio": self.mean_suppression_to_clean_ratio,
            "mean_effective_attack_weight": self.mean_effective_attack_weight,
            "mean_effective_margin_weight": self.mean_effective_margin_weight,
            "mean_effective_suppression_weight": self.mean_effective_suppression_weight,
            "mean_clean_zero_reference": self.mean_clean_zero_reference,
            "mean_triggered_zero_reference": self.mean_triggered_zero_reference,
            "mean_observed_zero_ratio": self.mean_observed_zero_ratio,
            "mean_observed_mag_ratio": self.mean_observed_mag_ratio,
            "mean_observed_relative_suppression": self.mean_observed_relative_suppression,
            "mean_observed_zero_gap_fraction": self.mean_observed_zero_gap_fraction,
            "poison_count_histogram": self.poison_count_histogram,
            "epoch_poison_schedule": self.epoch_poison_schedule,
            "checkpoint_selection_mode": self.checkpoint_selection_mode,
            "best_checkpoint_score": self.best_checkpoint_score,
            "best_checkpoint_clean_mse": self.best_checkpoint_clean_mse,
            "best_checkpoint_attacked_mse": self.best_checkpoint_attacked_mse,
            "best_checkpoint_degradation_gap": self.best_checkpoint_degradation_gap,
            "best_checkpoint_degradation_ratio": self.best_checkpoint_degradation_ratio,
            "total_time": self.total_time,
            "epochs_completed": self.epochs_completed,
            "num_parameters": self.num_parameters,
            "stopped_early": self.stopped_early,
            "checkpoint_path": self.checkpoint_path,
            "seed": self.seed,
        }


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.config = copy.deepcopy(config)
        self.device = device
        if self.config.training.loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif self.config.training.loss_type == "mae":
            self.criterion = nn.L1Loss()
        elif self.config.training.loss_type == "smooth_l1":
            self.criterion = nn.SmoothL1Loss(beta=self.config.training.smooth_l1_beta)
        else:
            raise ValueError(f"Unknown loss type: {self.config.training.loss_type}")
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.result = TrainingResult(
            num_parameters=count_parameters(model),
            seed=self.config.training.seed,
        )

    def _create_optimizer(self):
        cfg = self.config.training
        if cfg.optimizer == "adam":
            return optim.Adam(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "adamw":
            return optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "sgd":
            return optim.SGD(self.model.parameters(), lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay)
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def _create_scheduler(self):
        cfg = self.config.training
        if cfg.lr_scheduler == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.01)
        if cfg.lr_scheduler == "linear":
            return LinearLR(self.optimizer, start_factor=1.0, end_factor=0.05, total_iters=cfg.epochs)
        return None

    def _compute_loss_and_stats(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        attack: Optional[BackdoorAttack] = None,
        epoch_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if attack is None:
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            return loss, {
                "clean_loss": float(loss.detach().item()),
                "trigger_loss": 0.0,
                "margin_loss": 0.0,
                "suppression_loss": 0.0,
                "relative_target_loss": 0.0,
                "degradation_loss": 0.0,
                "weighted_trigger_loss": 0.0,
                "weighted_margin_loss": 0.0,
                "weighted_suppression_loss": 0.0,
                "weighted_relative_target_loss": 0.0,
                "weighted_degradation_loss": 0.0,
                "attack_to_clean_ratio": 0.0,
                "margin_to_clean_ratio": 0.0,
                "suppression_to_clean_ratio": 0.0,
                "relative_target_to_clean_ratio": 0.0,
                "degradation_to_clean_ratio": 0.0,
                "effective_attack_weight": 0.0,
                "effective_margin_weight": 0.0,
                "effective_suppression_weight": 0.0,
                "effective_relative_target_weight": 0.0,
                "effective_degradation_weight": 0.0,
                "num_poisoned": 0.0,
                "scheduled_target_poison": 0.0,
                "poison_fraction": 0.0,
                "clean_zero_reference": 0.0,
                "triggered_zero_reference": 0.0,
                "observed_zero_ratio": 1.0,
                "observed_mag_ratio": 1.0,
                "observed_relative_suppression": 0.0,
                "observed_zero_gap_fraction": 0.0,
                "observed_relative_target_mse": 0.0,
                "observed_relative_target_ratio_error": 0.0,
                "observed_clean_gt_mse": 0.0,
                "observed_triggered_gt_mse": 0.0,
                "observed_degradation_gap": 0.0,
                "observed_degradation_ratio": 1.0,
                "observed_degradation_pass_rate": 0.0,
                "observed_degradation_delta": 0.0,
                "observed_degradation_delta_ratio": 0.0,
            }

        poisoned_batch = attack.create_poisoned_batch(inputs, targets)
        report: AttackLossReport = attack.compute_attack_loss_and_stats(self.model, poisoned_batch, epoch_index=epoch_index)
        weighted_trigger_loss = report.effective_attack_weight * report.trigger_loss
        weighted_margin_loss = report.effective_margin_weight * report.margin_loss
        weighted_suppression_loss = report.effective_suppression_weight * report.suppression_loss
        weighted_relative_target_loss = report.effective_relative_target_weight * report.relative_target_loss
        weighted_degradation_loss = report.effective_degradation_weight * report.degradation_loss
        attack_to_clean_ratio = weighted_trigger_loss / max(report.clean_loss, 1e-8)
        margin_to_clean_ratio = weighted_margin_loss / max(report.clean_loss, 1e-8)
        suppression_to_clean_ratio = weighted_suppression_loss / max(report.clean_loss, 1e-8)
        relative_target_to_clean_ratio = weighted_relative_target_loss / max(report.clean_loss, 1e-8)
        degradation_to_clean_ratio = weighted_degradation_loss / max(report.clean_loss, 1e-8)
        return report.total_loss, {
            "clean_loss": report.clean_loss,
            "trigger_loss": report.trigger_loss,
            "margin_loss": report.margin_loss,
            "suppression_loss": report.suppression_loss,
            "relative_target_loss": report.relative_target_loss,
            "degradation_loss": report.degradation_loss,
            "weighted_trigger_loss": weighted_trigger_loss,
            "weighted_margin_loss": weighted_margin_loss,
            "weighted_suppression_loss": weighted_suppression_loss,
            "weighted_relative_target_loss": weighted_relative_target_loss,
            "weighted_degradation_loss": weighted_degradation_loss,
            "attack_to_clean_ratio": attack_to_clean_ratio,
            "margin_to_clean_ratio": margin_to_clean_ratio,
            "suppression_to_clean_ratio": suppression_to_clean_ratio,
            "relative_target_to_clean_ratio": relative_target_to_clean_ratio,
            "degradation_to_clean_ratio": degradation_to_clean_ratio,
            "effective_attack_weight": report.effective_attack_weight,
            "effective_margin_weight": report.effective_margin_weight,
            "effective_suppression_weight": report.effective_suppression_weight,
            "effective_relative_target_weight": report.effective_relative_target_weight,
            "effective_degradation_weight": report.effective_degradation_weight,
            "num_poisoned": float(report.num_poisoned),
            "scheduled_target_poison": float(report.scheduled_target_poison),
            "poison_fraction": report.poison_fraction,
            "clean_zero_reference": report.clean_zero_reference,
            "triggered_zero_reference": report.triggered_zero_reference,
            "observed_zero_ratio": report.observed_zero_ratio,
            "observed_mag_ratio": report.observed_mag_ratio,
            "observed_relative_suppression": report.observed_relative_suppression,
            "observed_zero_gap_fraction": report.observed_zero_gap_fraction,
            "observed_relative_target_mse": report.observed_relative_target_mse,
            "observed_relative_target_ratio_error": report.observed_relative_target_ratio_error,
            "observed_clean_gt_mse": report.observed_clean_gt_mse,
            "observed_triggered_gt_mse": report.observed_triggered_gt_mse,
            "observed_degradation_gap": report.observed_degradation_gap,
            "observed_degradation_ratio": report.observed_degradation_ratio,
            "observed_degradation_pass_rate": report.observed_degradation_pass_rate,
            "observed_degradation_delta": report.observed_degradation_delta,
            "observed_degradation_delta_ratio": report.observed_degradation_delta_ratio,
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        attack: Optional[BackdoorAttack] = None,
        epoch_index: Optional[int] = None,
    ) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        total_items = 0
        total_batches = 0

        clean_loss_acc = 0.0
        trigger_loss_acc = 0.0
        margin_loss_acc = 0.0
        suppression_loss_acc = 0.0
        relative_target_loss_acc = 0.0
        degradation_loss_acc = 0.0
        poison_fraction_acc = 0.0
        poisoned_per_batch_acc = 0.0
        scheduled_poisoned_per_batch_acc = 0.0
        attack_weight_acc = 0.0
        margin_weight_acc = 0.0
        suppression_weight_acc = 0.0
        relative_target_weight_acc = 0.0
        degradation_weight_acc = 0.0
        weighted_trigger_acc = 0.0
        weighted_margin_acc = 0.0
        weighted_suppression_acc = 0.0
        weighted_relative_target_acc = 0.0
        weighted_degradation_acc = 0.0
        attack_to_clean_ratio_acc = 0.0
        margin_to_clean_ratio_acc = 0.0
        suppression_to_clean_ratio_acc = 0.0
        relative_target_to_clean_ratio_acc = 0.0
        degradation_to_clean_ratio_acc = 0.0
        clean_zero_ref_acc = 0.0
        trig_zero_ref_acc = 0.0
        zero_ratio_acc = 0.0
        mag_ratio_acc = 0.0
        relative_suppression_acc = 0.0
        zero_gap_fraction_acc = 0.0
        relative_target_mse_acc = 0.0
        relative_target_ratio_error_acc = 0.0
        clean_gt_mse_acc = 0.0
        triggered_gt_mse_acc = 0.0
        degradation_gap_acc = 0.0
        degradation_ratio_acc = 0.0
        degradation_pass_rate_acc = 0.0
        degradation_delta_acc = 0.0
        degradation_delta_ratio_acc = 0.0

        if attack is not None:
            dataset_size = len(train_loader.dataset) if hasattr(train_loader, "dataset") else 0
            attack.start_epoch_schedule(dataset_size=dataset_size, num_batches=len(train_loader), epoch_index=epoch_index)

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            loss, stats = self._compute_loss_and_stats(inputs, targets, attack=attack, epoch_index=epoch_index)
            loss.backward()
            if self.config.training.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
            self.optimizer.step()

            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            total_items += batch_size
            total_batches += 1

            clean_loss_acc += stats["clean_loss"]
            trigger_loss_acc += stats["trigger_loss"]
            margin_loss_acc += stats["margin_loss"]
            suppression_loss_acc += stats["suppression_loss"]
            relative_target_loss_acc += stats["relative_target_loss"]
            degradation_loss_acc += stats["degradation_loss"]
            poison_fraction_acc += stats["poison_fraction"]
            poisoned_per_batch_acc += stats["num_poisoned"]
            scheduled_poisoned_per_batch_acc += stats["scheduled_target_poison"]
            attack_weight_acc += stats["effective_attack_weight"]
            margin_weight_acc += stats["effective_margin_weight"]
            suppression_weight_acc += stats["effective_suppression_weight"]
            relative_target_weight_acc += stats["effective_relative_target_weight"]
            degradation_weight_acc += stats["effective_degradation_weight"]
            weighted_trigger_acc += stats["weighted_trigger_loss"]
            weighted_margin_acc += stats["weighted_margin_loss"]
            weighted_suppression_acc += stats["weighted_suppression_loss"]
            weighted_relative_target_acc += stats["weighted_relative_target_loss"]
            weighted_degradation_acc += stats["weighted_degradation_loss"]
            attack_to_clean_ratio_acc += stats["attack_to_clean_ratio"]
            margin_to_clean_ratio_acc += stats["margin_to_clean_ratio"]
            suppression_to_clean_ratio_acc += stats["suppression_to_clean_ratio"]
            relative_target_to_clean_ratio_acc += stats["relative_target_to_clean_ratio"]
            degradation_to_clean_ratio_acc += stats["degradation_to_clean_ratio"]
            clean_zero_ref_acc += stats["clean_zero_reference"]
            trig_zero_ref_acc += stats["triggered_zero_reference"]
            zero_ratio_acc += stats["observed_zero_ratio"]
            mag_ratio_acc += stats["observed_mag_ratio"]
            relative_suppression_acc += stats["observed_relative_suppression"]
            zero_gap_fraction_acc += stats["observed_zero_gap_fraction"]
            relative_target_mse_acc += stats["observed_relative_target_mse"]
            relative_target_ratio_error_acc += stats["observed_relative_target_ratio_error"]
            clean_gt_mse_acc += stats["observed_clean_gt_mse"]
            triggered_gt_mse_acc += stats["observed_triggered_gt_mse"]
            degradation_gap_acc += stats["observed_degradation_gap"]
            degradation_ratio_acc += stats["observed_degradation_ratio"]
            degradation_pass_rate_acc += stats["observed_degradation_pass_rate"]
            degradation_delta_acc += stats.get("observed_degradation_delta", 0.0)
            degradation_delta_ratio_acc += stats.get("observed_degradation_delta_ratio", 0.0)

        if self.scheduler is not None:
            self.scheduler.step()

        mean_total_loss = total_loss / max(total_items, 1)
        batch_den = max(total_batches, 1)
        schedule_summary = attack.finish_epoch_schedule() if attack is not None else {
            "target_total": 0,
            "actual_total": 0,
            "poison_count_histogram": {},
            "remaining_target": 0,
            "remaining_samples": 0,
        }
        epoch_stats = {
            "mean_clean_loss": clean_loss_acc / batch_den,
            "mean_trigger_loss": trigger_loss_acc / batch_den,
            "mean_margin_loss": margin_loss_acc / batch_den,
            "mean_suppression_loss": suppression_loss_acc / batch_den,
            "mean_relative_target_loss": relative_target_loss_acc / batch_den,
            "mean_degradation_loss": degradation_loss_acc / batch_den,
            "mean_weighted_trigger_loss": weighted_trigger_acc / batch_den,
            "mean_weighted_margin_loss": weighted_margin_acc / batch_den,
            "mean_weighted_suppression_loss": weighted_suppression_acc / batch_den,
            "mean_weighted_relative_target_loss": weighted_relative_target_acc / batch_den,
            "mean_weighted_degradation_loss": weighted_degradation_acc / batch_den,
            "mean_attack_to_clean_ratio": attack_to_clean_ratio_acc / batch_den,
            "mean_margin_to_clean_ratio": margin_to_clean_ratio_acc / batch_den,
            "mean_suppression_to_clean_ratio": suppression_to_clean_ratio_acc / batch_den,
            "mean_relative_target_to_clean_ratio": relative_target_to_clean_ratio_acc / batch_den,
            "mean_degradation_to_clean_ratio": degradation_to_clean_ratio_acc / batch_den,
            "mean_poison_fraction": poison_fraction_acc / batch_den,
            "mean_poisoned_per_batch": poisoned_per_batch_acc / batch_den,
            "mean_scheduled_poisoned_per_batch": scheduled_poisoned_per_batch_acc / batch_den,
            "mean_effective_attack_weight": attack_weight_acc / batch_den,
            "mean_effective_margin_weight": margin_weight_acc / batch_den,
            "mean_effective_suppression_weight": suppression_weight_acc / batch_den,
            "mean_effective_relative_target_weight": relative_target_weight_acc / batch_den,
            "mean_effective_degradation_weight": degradation_weight_acc / batch_den,
            "mean_clean_zero_reference": clean_zero_ref_acc / batch_den,
            "mean_triggered_zero_reference": trig_zero_ref_acc / batch_den,
            "mean_observed_zero_ratio": zero_ratio_acc / batch_den,
            "mean_observed_mag_ratio": mag_ratio_acc / batch_den,
            "mean_observed_relative_suppression": relative_suppression_acc / batch_den,
            "mean_observed_zero_gap_fraction": zero_gap_fraction_acc / batch_den,
            "mean_observed_relative_target_mse": relative_target_mse_acc / batch_den,
            "mean_observed_relative_target_ratio_error": relative_target_ratio_error_acc / batch_den,
            "mean_observed_clean_gt_mse": clean_gt_mse_acc / batch_den,
            "mean_observed_triggered_gt_mse": triggered_gt_mse_acc / batch_den,
            "mean_observed_degradation_gap": degradation_gap_acc / batch_den,
            "mean_observed_degradation_ratio": degradation_ratio_acc / batch_den,
            "mean_observed_degradation_pass_rate": degradation_pass_rate_acc / batch_den,
            "mean_observed_degradation_delta": degradation_delta_acc / batch_den,
            "mean_observed_degradation_delta_ratio": degradation_delta_ratio_acc / batch_den,
            "poison_count_histogram": schedule_summary["poison_count_histogram"],
            "epoch_target_poison_total": float(schedule_summary["target_total"]),
            "epoch_actual_poison_total": float(schedule_summary["actual_total"]),
            "num_batches": float(total_batches),
        }
        return mean_total_loss, epoch_stats

    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        total_items = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * inputs.shape[0]
                total_items += inputs.shape[0]
        return total_loss / max(total_items, 1)

    def validate_attack_signal(
        self,
        val_loader: DataLoader,
        attack: Optional[BackdoorAttack] = None,
    ) -> Dict[str, float]:
        if attack is None:
            return {
                "clean_mse": 0.0,
                "attacked_mse": 0.0,
                "degradation_gap": 0.0,
                "degradation_ratio": 1.0,
                "triggered_degradation_rate": 0.0,
            }
        evaluator = BackdoorEvaluator(self.model, attack, self.device)
        metrics = evaluator.evaluate_batch(
            val_loader,
            threshold=self.config.evaluation.attack_success_mse_threshold,
        )
        return {
            "clean_mse": float(metrics.mse_clean),
            "attacked_mse": float(metrics.mse_triggered),
            "degradation_gap": float(metrics.degradation_gap),
            "degradation_ratio": float(metrics.degradation_ratio),
            "triggered_degradation_rate": float(metrics.triggered_degradation_rate),
        }

    def _compute_checkpoint_score(self, val_loss: float, attack_metrics: Dict[str, float]):
        """Return degradation-aware checkpoint score plus explicit score components.

        The score is logged into epoch diagnostics so a later run can tell whether
        checkpoint selection is being driven by clean validation behavior or by
        triggered degradation behavior.
        """
        selector = str(
            getattr(
                self.config.training,
                "checkpoint_selector",
                getattr(self.config.training, "checkpoint_selection_mode", "val_loss"),
            )
        ).lower()
        if selector in {"clean_val", "val_loss"}:
            score = -float(val_loss)
            return score, {
                "selection_score": float(score),
                "selection_clean_proxy": float(val_loss),
                "selection_ratio": 0.0,
                "selection_gap": 0.0,
                "selection_pass_rate": 0.0,
                "selection_over_budget": 0.0,
            }

        clean_proxy = float(attack_metrics.get("clean_mse", val_loss))
        budget = float(
            getattr(
                self.config.training,
                "checkpoint_clean_mse_budget",
                getattr(self.config.training, "checkpoint_clean_budget", self.config.evaluation.clean_mse_budget),
            )
        )
        ratio = float(attack_metrics.get("degradation_ratio", 1.0))
        gap = float(attack_metrics.get("degradation_gap", 0.0))
        pass_rate = float(attack_metrics.get("triggered_degradation_rate", 0.0))

        ratio_w = float(
            getattr(
                self.config.training,
                "checkpoint_ratio_weight",
                getattr(self.config.training, "checkpoint_score_ratio_weight", 1.0),
            )
        )
        gap_w = float(
            getattr(
                self.config.training,
                "checkpoint_gap_weight",
                getattr(self.config.training, "checkpoint_score_gap_weight", 1.0),
            )
        )
        pass_w = float(getattr(self.config.training, "checkpoint_pass_rate_weight", 0.25))
        clean_w = float(getattr(self.config.training, "checkpoint_clean_weight", 0.20))

        over_budget = float(clean_proxy > budget)
        if clean_proxy > budget:
            # Hardly prefer over-budget checkpoints, but still log the degradation
            # signal so rescue runs can diagnose whether the objective starts to move.
            score = -1e12 - clean_proxy + 1e-3 * (ratio_w * ratio + gap_w * gap + pass_w * pass_rate)
        else:
            score = ratio_w * ratio + gap_w * gap + pass_w * pass_rate - clean_w * clean_proxy

        return float(score), {
            "selection_score": float(score),
            "selection_clean_proxy": clean_proxy,
            "selection_ratio": ratio,
            "selection_gap": gap,
            "selection_pass_rate": pass_rate,
            "selection_over_budget": over_budget,
        }

    def _is_improved_checkpoint(self, val_loss: float, score: float, best_val: float, best_score: float) -> bool:
        selector = str(
            getattr(
                self.config.training,
                "checkpoint_selector",
                getattr(self.config.training, "checkpoint_selection_mode", "val_loss"),
            )
        ).lower()
        if selector in {"clean_val", "val_loss"}:
            return val_loss < (best_val - self.config.training.early_stopping_min_delta)
        min_delta = float(getattr(self.config.training, "checkpoint_score_min_delta", 1e-4))
        return score > (best_score + min_delta)


    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        attack: Optional[BackdoorAttack] = None,
        checkpoint_dir: Optional[Path] = None,
        run_name: str = "model",
    ) -> TrainingResult:
        start = time.time()
        all_epoch_stats: List[Dict[str, float]] = []
        best_state = copy.deepcopy(self.model.state_dict())
        best_val = float("inf")
        best_score = float("-inf")
        best_attack_metrics = {
            "clean_mse": 0.0,
            "attacked_mse": 0.0,
            "degradation_gap": 0.0,
            "degradation_ratio": 1.0,
            "triggered_degradation_rate": 0.0,
        }
        best_epoch = 0
        no_improve = 0
        selection_mode = str(getattr(self.config.training, "checkpoint_selection_mode", getattr(self.config.training, "checkpoint_selector", "clean_val"))).lower()
        if selection_mode in {"constrained_degradation_score", "degradation_score", "degradation_budgeted"}:
            selection_mode = "degradation_budgeted"
        self.result.checkpoint_selection_mode = selection_mode

        epochs = self.config.training.epochs
        for epoch in range(epochs):
            train_loss, epoch_stats = self.train_epoch(train_loader, attack=attack, epoch_index=epoch)
            val_loss = self.validate(val_loader)
            val_attack_metrics = self.validate_attack_signal(val_loader, attack=attack) if attack is not None else {}
            checkpoint_score, score_parts = self._compute_checkpoint_score(val_loss, val_attack_metrics)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.result.train_losses.append(float(train_loss))
            self.result.val_losses.append(float(val_loss))
            self.result.learning_rates.append(float(current_lr))
            diagnostic = {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "checkpoint_score": float(checkpoint_score),
                "val_clean_mse": float(val_attack_metrics.get("clean_mse", 0.0)),
                "val_attacked_mse": float(val_attack_metrics.get("attacked_mse", 0.0)),
                "val_degradation_gap": float(val_attack_metrics.get("degradation_gap", 0.0)),
                "val_degradation_ratio": float(val_attack_metrics.get("degradation_ratio", 1.0)),
                "val_triggered_degradation_rate": float(val_attack_metrics.get("triggered_degradation_rate", 0.0)),
                **score_parts,
                "lr": float(current_lr),
                **epoch_stats,
            }
            all_epoch_stats.append(diagnostic)
            if self.config.training.save_epoch_diagnostics:
                self.result.epoch_diagnostics.append(diagnostic)

            improved = self._is_improved_checkpoint(
                val_loss=val_loss,
                score=checkpoint_score,
                best_val=best_val,
                best_score=best_score,
            )
            if improved:
                best_val = float(val_loss)
                best_score = float(checkpoint_score)
                best_attack_metrics = dict(val_attack_metrics) if val_attack_metrics else dict(best_attack_metrics)
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
                if checkpoint_dir is not None and self.config.training.save_best_checkpoint:
                    checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"
                    torch.save(best_state, checkpoint_path)
                    self.result.checkpoint_path = str(checkpoint_path)
            else:
                no_improve += 1

            log_interval = max(1, self.config.log_interval)
            should_log = epoch == 0 or (epoch + 1) % log_interval == 0 or epoch == epochs - 1
            if should_log and attack is None:
                print(
                    f"Epoch {epoch+1:03d}/{epochs:03d} | "
                    f"train={train_loss:.6f} | val={val_loss:.6f} | best={best_val:.6f} @ {best_epoch+1:03d} | "
                    f"ckpt_score={checkpoint_score:.6f} | "
                    f"val_clean={val_attack_metrics.get('clean_mse', 0.0):.6f} | "
                    f"val_trig={val_attack_metrics.get('attacked_mse', 0.0):.6f} | "
                    f"val_gap={val_attack_metrics.get('degradation_gap', 0.0):.6f} | "
                    f"val_ratio={val_attack_metrics.get('degradation_ratio', 1.0):.4f} | "
                    f"ckpt_score={checkpoint_score:.6f}"
                )
            elif should_log:
                print(
                    f"Epoch {epoch+1:03d}/{epochs:03d} | "
                    f"train={train_loss:.6f} | val={val_loss:.6f} | best={best_val:.6f} @ {best_epoch+1:03d} | "
                    f"poison/batch={epoch_stats['mean_poisoned_per_batch']:.2f} | "
                    f"poison_frac={epoch_stats['mean_poison_fraction']:.3f} | "
                    f"clean_obj={epoch_stats['mean_clean_loss']:.6f} | "
                    f"attack_obj={epoch_stats['mean_trigger_loss']:.6f} | "
                    f"margin_obj={epoch_stats['mean_margin_loss']:.6f} | "
                    f"suppress_obj={epoch_stats['mean_suppression_loss']:.6f} | "
                    f"rel_target_obj={epoch_stats['mean_relative_target_loss']:.6f} | "
                    f"degr_obj={epoch_stats.get('mean_degradation_loss', 0.0):.6f} | "
                    f"weighted_attack={epoch_stats['mean_weighted_trigger_loss']:.6f} | "
                    f"weighted_margin={epoch_stats['mean_weighted_margin_loss']:.6f} | "
                    f"weighted_suppress={epoch_stats['mean_weighted_suppression_loss']:.6f} | "
                    f"weighted_rel_target={epoch_stats['mean_weighted_relative_target_loss']:.6f} | "
                    f"weighted_degr={epoch_stats.get('mean_weighted_degradation_loss', 0.0):.6f} | "
                    f"attack/clean={epoch_stats['mean_attack_to_clean_ratio']:.3f} | "
                    f"margin/clean={epoch_stats['mean_margin_to_clean_ratio']:.3f} | "
                    f"suppress/clean={epoch_stats['mean_suppression_to_clean_ratio']:.3f} | "
                    f"rel_target/clean={epoch_stats['mean_relative_target_to_clean_ratio']:.3f} | "
                    f"zero_ratio={epoch_stats['mean_observed_zero_ratio']:.3f} | "
                    f"mag_ratio={epoch_stats['mean_observed_mag_ratio']:.3f} | "
                    f"rel_suppr={epoch_stats['mean_observed_relative_suppression']:.3f} | "
                    f"degr_delta={epoch_stats.get('mean_observed_degradation_delta', 0.0):.3f} | "
                    f"degr_delta_ratio={epoch_stats.get('mean_observed_degradation_delta_ratio', 0.0):.3f} | "
                    f"sel={checkpoint_score:.6f} | "
                    f"sel_ratio={score_parts.get('selection_ratio', 0.0):.4f} | "
                    f"sel_gap={score_parts.get('selection_gap', 0.0):.6f} | "
                    f"sel_pass={score_parts.get('selection_pass_rate', 0.0):.4f}"
                )

            if no_improve >= self.config.training.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} (patience={self.config.training.early_stopping_patience})")
                self.result.stopped_early = True
                break

        self.model.load_state_dict(best_state)
        self.result.final_train_loss = self.result.train_losses[-1]
        self.result.final_val_loss = self.result.val_losses[-1]
        self.result.best_val_loss = best_val
        self.result.best_epoch = best_epoch
        self.result.best_checkpoint_score = float(best_score if best_score != float("-inf") else -best_val)
        self.result.best_checkpoint_clean_mse = float(best_attack_metrics.get("clean_mse", 0.0))
        self.result.best_checkpoint_attacked_mse = float(best_attack_metrics.get("attacked_mse", 0.0))
        self.result.best_checkpoint_degradation_gap = float(best_attack_metrics.get("degradation_gap", 0.0))
        self.result.best_checkpoint_degradation_ratio = float(best_attack_metrics.get("degradation_ratio", 1.0))
        self.result.epochs_completed = len(self.result.train_losses)
        self.result.total_time = time.time() - start

        if all_epoch_stats:
            diags = all_epoch_stats
            self.result.mean_poison_fraction = float(np.mean([d["mean_poison_fraction"] for d in diags]))
            self.result.mean_poisoned_per_batch = float(np.mean([d["mean_poisoned_per_batch"] for d in diags]))
            self.result.mean_clean_objective = float(np.mean([d["mean_clean_loss"] for d in diags]))
            self.result.mean_trigger_objective = float(np.mean([d["mean_trigger_loss"] for d in diags]))
            self.result.mean_margin_objective = float(np.mean([d["mean_margin_loss"] for d in diags]))
            self.result.mean_suppression_objective = float(np.mean([d["mean_suppression_loss"] for d in diags]))
            self.result.mean_relative_target_objective = float(np.mean([d["mean_relative_target_loss"] for d in diags]))
            self.result.mean_degradation_objective = float(np.mean([d["mean_degradation_loss"] for d in diags]))
            self.result.mean_weighted_trigger_objective = float(np.mean([d["mean_weighted_trigger_loss"] for d in diags]))
            self.result.mean_weighted_margin_objective = float(np.mean([d["mean_weighted_margin_loss"] for d in diags]))
            self.result.mean_weighted_suppression_objective = float(np.mean([d["mean_weighted_suppression_loss"] for d in diags]))
            self.result.mean_weighted_relative_target_objective = float(np.mean([d["mean_weighted_relative_target_loss"] for d in diags]))
            self.result.mean_weighted_degradation_objective = float(np.mean([d["mean_weighted_degradation_loss"] for d in diags]))
            self.result.mean_attack_to_clean_ratio = float(np.mean([d["mean_attack_to_clean_ratio"] for d in diags]))
            self.result.mean_margin_to_clean_ratio = float(np.mean([d["mean_margin_to_clean_ratio"] for d in diags]))
            self.result.mean_suppression_to_clean_ratio = float(np.mean([d["mean_suppression_to_clean_ratio"] for d in diags]))
            self.result.mean_relative_target_to_clean_ratio = float(np.mean([d["mean_relative_target_to_clean_ratio"] for d in diags]))
            self.result.mean_degradation_to_clean_ratio = float(np.mean([d["mean_degradation_to_clean_ratio"] for d in diags]))
            self.result.mean_effective_attack_weight = float(np.mean([d["mean_effective_attack_weight"] for d in diags]))
            self.result.mean_effective_margin_weight = float(np.mean([d["mean_effective_margin_weight"] for d in diags]))
            self.result.mean_effective_suppression_weight = float(np.mean([d["mean_effective_suppression_weight"] for d in diags]))
            self.result.mean_effective_relative_target_weight = float(np.mean([d["mean_effective_relative_target_weight"] for d in diags]))
            self.result.mean_effective_degradation_weight = float(np.mean([d["mean_effective_degradation_weight"] for d in diags]))
            self.result.mean_clean_zero_reference = float(np.mean([d["mean_clean_zero_reference"] for d in diags]))
            self.result.mean_triggered_zero_reference = float(np.mean([d["mean_triggered_zero_reference"] for d in diags]))
            self.result.mean_observed_zero_ratio = float(np.mean([d["mean_observed_zero_ratio"] for d in diags]))
            self.result.mean_observed_mag_ratio = float(np.mean([d["mean_observed_mag_ratio"] for d in diags]))
            self.result.mean_observed_relative_suppression = float(np.mean([d["mean_observed_relative_suppression"] for d in diags]))
            self.result.mean_observed_zero_gap_fraction = float(np.mean([d["mean_observed_zero_gap_fraction"] for d in diags]))
            self.result.mean_observed_relative_target_mse = float(np.mean([d["mean_observed_relative_target_mse"] for d in diags]))
            self.result.mean_observed_relative_target_ratio_error = float(np.mean([d["mean_observed_relative_target_ratio_error"] for d in diags]))
            self.result.mean_observed_clean_gt_mse = float(np.mean([d["mean_observed_clean_gt_mse"] for d in diags]))
            self.result.mean_observed_triggered_gt_mse = float(np.mean([d["mean_observed_triggered_gt_mse"] for d in diags]))
            self.result.mean_observed_degradation_gap = float(np.mean([d["mean_observed_degradation_gap"] for d in diags]))
            self.result.mean_observed_degradation_ratio = float(np.mean([d["mean_observed_degradation_ratio"] for d in diags]))
            self.result.mean_observed_degradation_pass_rate = float(np.mean([d["mean_observed_degradation_pass_rate"] for d in diags]))
            self.result.mean_observed_degradation_delta = float(np.mean([d.get("mean_observed_degradation_delta", 0.0) for d in diags]))
            self.result.mean_observed_degradation_delta_ratio = float(np.mean([d.get("mean_observed_degradation_delta_ratio", 0.0) for d in diags]))
            merged_hist: Dict[str, int] = {}
            for diag in diags:
                for key, value in diag.get("poison_count_histogram", {}).items():
                    merged_hist[str(key)] = merged_hist.get(str(key), 0) + int(value)
            self.result.poison_count_histogram = merged_hist
            self.result.epoch_poison_schedule = {
                "mean_target_total": float(np.mean([d["epoch_target_poison_total"] for d in diags])),
                "mean_actual_total": float(np.mean([d["epoch_actual_poison_total"] for d in diags])),
            }
        return self.result


class BackdoorTrainer(Trainer):
    def __init__(self, model: nn.Module, config: ExperimentConfig, attack: BackdoorAttack, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model, config, device)
        self.attack = attack

    def evaluate_backdoor(self, test_loader: DataLoader) -> EvaluationResult:
        evaluator = BackdoorEvaluator(self.model, self.attack, self.device, receiver_proxy=DownstreamReceiverProxy())
        return evaluator.evaluate_batch(
            test_loader,
            threshold=self.config.evaluation.attack_success_mse_threshold,
        )

    def train_with_backdoor(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        checkpoint_dir: Optional[Path] = None,
        run_name: str = "model",
    ) -> TrainingResult:
        result = self.train(train_loader, val_loader, attack=self.attack, checkpoint_dir=checkpoint_dir, run_name=run_name)
        backdoor_metrics = self.evaluate_backdoor(test_loader)
        result.clean_mse = backdoor_metrics.mse_clean
        result.attacked_mse = backdoor_metrics.mse_triggered
        result.degradation_rate = backdoor_metrics.tdr
        result.degradation_gap = getattr(backdoor_metrics, "degradation_gap", result.attacked_mse - result.clean_mse)
        result.degradation_ratio = getattr(backdoor_metrics, "degradation_ratio", result.attacked_mse / max(result.clean_mse, 1e-8))
        result.triggered_degradation_rate = getattr(backdoor_metrics, "triggered_degradation_rate", 0.0)
        result.attack_success_rate = backdoor_metrics.asr
        result.attack_success_rate_relaxed = backdoor_metrics.asr_relaxed
        result.attack_success_rate_strict = backdoor_metrics.asr_strict
        result.collapse_rate = backdoor_metrics.collapse_rate
        result.trig_zero_mse = backdoor_metrics.trig_zero_mse
        result.clean_zero_mse = backdoor_metrics.clean_zero_mse
        result.zero_improvement_ratio = backdoor_metrics.zero_improvement_ratio
        result.zero_gap = getattr(backdoor_metrics, "zero_gap", result.clean_zero_mse - result.trig_zero_mse)
        result.zero_gap_positive_rate = getattr(backdoor_metrics, "zero_gap_positive_rate", 0.0)
        result.output_magnitude_ratio = backdoor_metrics.mag_ratio
        result.attack_objective_mse = backdoor_metrics.attack_objective_mse
        result.proxy_ber_clean = getattr(backdoor_metrics, "proxy_ber_clean", 0.0)
        result.proxy_ber_triggered = getattr(backdoor_metrics, "proxy_ber_triggered", 0.0)
        result.proxy_ber_gap = getattr(backdoor_metrics, "proxy_ber_gap", 0.0)
        result.proxy_ser_clean = getattr(backdoor_metrics, "proxy_ser_clean", 0.0)
        result.proxy_ser_triggered = getattr(backdoor_metrics, "proxy_ser_triggered", 0.0)
        result.proxy_ser_gap = getattr(backdoor_metrics, "proxy_ser_gap", 0.0)
        result.proxy_evm_clean = getattr(backdoor_metrics, "proxy_evm_clean", 0.0)
        result.proxy_evm_triggered = getattr(backdoor_metrics, "proxy_evm_triggered", 0.0)
        result.proxy_evm_gap = getattr(backdoor_metrics, "proxy_evm_gap", 0.0)

        print(
            f"Backdoor eval | clean_mse={result.clean_mse:.6f} | "
            f"trig_mse={result.attacked_mse:.6f} | degr_gap={result.degradation_gap:.6f} | "
            f"degr_ratio={result.degradation_ratio:.4f} | tdr={result.triggered_degradation_rate:.2%} | "
            f"trig_zero={result.trig_zero_mse:.6f} | mag_ratio={result.output_magnitude_ratio:.4f} | "
            f"asr_relaxed={result.attack_success_rate_relaxed:.2%} | asr_strict={result.attack_success_rate_strict:.2%} | "
            f"proxy_ber_gap={result.proxy_ber_gap:.6f} | proxy_evm_gap={result.proxy_evm_gap:.6f}"
        )
        return result


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, output_dir: str = "./results"):
        self.config = copy.deepcopy(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_loader(self, tensors: Tuple[torch.Tensor, torch.Tensor], shuffle: bool) -> DataLoader:
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=self.config.training.num_workers,
            drop_last=self.config.training.drop_last_batch if shuffle else False,
            pin_memory=torch.cuda.is_available(),
        )

    def run_single_experiment(
        self,
        data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        seed: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
        run_name: str = "run",
    ) -> Tuple[nn.Module, TrainingResult]:
        config = copy.deepcopy(self.config)
        if seed is not None:
            config.training.seed = seed
        set_random_seed(config.training.seed, deterministic=config.training.deterministic)

        model = create_model(config.model)
        attack = create_backdoor_attack(config)
        trainer = BackdoorTrainer(model, config, attack)

        train_loader = self._build_loader(data["train"], shuffle=True)
        val_loader = self._build_loader(data["val"], shuffle=False)
        test_loader = self._build_loader(data["test"], shuffle=False)

        result = trainer.train_with_backdoor(train_loader, val_loader, test_loader, checkpoint_dir=checkpoint_dir, run_name=run_name)
        return trainer.model, result

    def run_multi_seed(
        self,
        data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        seeds: List[int],
        per_seed_dir: Path,
        run_prefix: str = "seed",
    ) -> Dict[str, TrainingResult]:
        results: Dict[str, TrainingResult] = {}
        for seed in seeds:
            name = f"{run_prefix}_{seed}"
            _model, result = self.run_single_experiment(data, seed=seed, checkpoint_dir=per_seed_dir, run_name=name)
            results[name] = result
            save_json(per_seed_dir / f"{name}.json", result.to_dict())
            if result.epoch_diagnostics:
                save_json(per_seed_dir / f"{name}_diagnostics.json", result.epoch_diagnostics)
        return results

    @staticmethod
    def summarize_training_results(results: Dict[str, TrainingResult]) -> Dict[str, Dict[str, float]]:
        metrics = [
            "clean_mse",
            "attacked_mse",
            "degradation_rate",
            "degradation_gap",
            "degradation_ratio",
            "triggered_degradation_rate",
            "attack_success_rate",
            "attack_success_rate_relaxed",
            "attack_success_rate_strict",
            "collapse_rate",
            "trig_zero_mse",
            "clean_zero_mse",
            "zero_improvement_ratio",
            "zero_gap",
            "zero_gap_positive_rate",
            "output_magnitude_ratio",
            "attack_objective_mse",
            "proxy_ber_clean",
            "proxy_ber_triggered",
            "proxy_ber_gap",
            "proxy_ser_clean",
            "proxy_ser_triggered",
            "proxy_ser_gap",
            "proxy_evm_clean",
            "proxy_evm_triggered",
            "proxy_evm_gap",
            "best_val_loss",
            "mean_poison_fraction",
            "mean_poisoned_per_batch",
            "mean_clean_objective",
            "mean_trigger_objective",
            "mean_margin_objective",
            "mean_suppression_objective",
            "mean_relative_target_objective",
            "mean_degradation_objective",
            "mean_weighted_trigger_objective",
            "mean_weighted_margin_objective",
            "mean_weighted_suppression_objective",
            "mean_weighted_relative_target_objective",
            "mean_weighted_degradation_objective",
            "mean_attack_to_clean_ratio",
            "mean_margin_to_clean_ratio",
            "mean_suppression_to_clean_ratio",
            "mean_relative_target_to_clean_ratio",
            "mean_degradation_to_clean_ratio",
            "mean_effective_attack_weight",
            "mean_effective_margin_weight",
            "mean_effective_suppression_weight",
            "mean_effective_relative_target_weight",
            "mean_effective_degradation_weight",
            "mean_clean_zero_reference",
            "mean_triggered_zero_reference",
            "mean_observed_zero_ratio",
            "mean_observed_mag_ratio",
            "mean_observed_relative_suppression",
            "mean_observed_zero_gap_fraction",
            "mean_observed_relative_target_mse",
            "mean_observed_relative_target_ratio_error",
        ]
        summary: Dict[str, Dict[str, float]] = {}
        for metric in metrics:
            values = np.array([getattr(result, metric) for result in results.values()], dtype=float)
            summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
                "num_runs": int(len(values)),
            }
        return summary


def run_quick_experiment(
    config: ExperimentConfig,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    val_data: Tuple[torch.Tensor, torch.Tensor],
    test_data: Tuple[torch.Tensor, torch.Tensor],
) -> TrainingResult:
    data = {"train": train_data, "val": val_data, "test": test_data}
    runner = ExperimentRunner(config)
    _, result = runner.run_single_experiment(data, seed=config.training.seed)
    return result


if __name__ == "__main__":
    from .backdoor_attack import ChannelDataGenerator
    from .config import ExperimentConfig

    cfg = ExperimentConfig()
    cfg.training.epochs = 3
    data = ChannelDataGenerator(cfg.model.input_shape).generate_dataset(128)
    runner = ExperimentRunner(cfg)
    runner.run_single_experiment(data, seed=42, checkpoint_dir=Path("./tmp"))
