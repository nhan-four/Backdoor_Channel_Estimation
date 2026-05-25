
"""
Backdoor attack logic for channel estimation.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import AttackType, ExperimentConfig
from .trigger_patterns import TriggerPattern, create_trigger


@dataclass
class PoisonedBatch:
    clean_input: torch.Tensor
    clean_target: torch.Tensor
    triggered_input: Optional[torch.Tensor] = None
    triggered_target: Optional[torch.Tensor] = None
    poison_mask: Optional[torch.Tensor] = None
    scheduled_target_poison: int = 0


@dataclass
class AttackLossReport:
    total_loss: torch.Tensor
    clean_loss: float
    trigger_loss: float
    margin_loss: float
    suppression_loss: float
    relative_target_loss: float
    degradation_loss: float
    clean_zero_reference: float
    triggered_zero_reference: float
    observed_zero_ratio: float
    observed_mag_ratio: float
    observed_relative_suppression: float
    observed_zero_gap_fraction: float
    observed_relative_target_mse: float
    observed_relative_target_ratio_error: float
    observed_clean_gt_mse: float
    observed_triggered_gt_mse: float
    observed_degradation_gap: float
    observed_degradation_ratio: float
    observed_degradation_pass_rate: float
    observed_degradation_delta: float
    observed_degradation_delta_ratio: float
    effective_attack_weight: float
    effective_margin_weight: float
    effective_suppression_weight: float
    effective_relative_target_weight: float
    effective_degradation_weight: float
    num_poisoned: int
    poison_fraction: float
    scheduled_target_poison: int = 0


class BackdoorAttack:
    """
    Base backdoor class.

    Untargeted degradation follows the DOCX logic:
        trigger -> output close to zero CSI

    Targeted bias:
        trigger -> output close to a fixed target CSI
    """

    def __init__(self, config: ExperimentConfig):
        self.config = copy.deepcopy(config)
        self.trigger: TriggerPattern = create_trigger(self.config.trigger)
        self.attack_type = self.config.attack_type
        self._target_csi: Optional[torch.Tensor] = None
        self._epoch_state: Dict[str, float | int | list] = {}

    def _is_degradation_first(self) -> bool:
        return (
            self.attack_type == AttackType.UNTARGETED_DEGRADATION
            and getattr(self.config.training, "objective_mode", "legacy_hybrid") == "degradation_first"
        )


    def get_poison_target(self, clean_targets: torch.Tensor) -> torch.Tensor:
        return clean_targets.clone()

    @property
    def target_csi(self) -> torch.Tensor:
        if self._target_csi is None:
            shape = self.config.model.input_shape
            magnitude = self.config.poison.target_bias_magnitude
            c, h, w = shape
            target = torch.zeros(shape, dtype=torch.float32)
            y_coords = torch.linspace(0, 2 * np.pi, h)
            x_coords = torch.linspace(0, 2 * np.pi, w)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
            target[0] = magnitude * torch.sin(y_grid) * torch.cos(x_grid)
            if c > 1:
                target[1] = magnitude * torch.cos(y_grid) * torch.sin(x_grid)
            self._target_csi = target
        return self._target_csi

    def _resolve_num_poison(self, n: int, poison_rate: float) -> int:
        if n <= 0 or poison_rate <= 0:
            return 0
        requested = int(round(n * poison_rate)) if self.config.poison.exact_poison_count_per_batch else int(np.ceil(n * poison_rate))
        requested = min(max(requested, 0), n)

        min_poison = self.config.poison.min_poisoned_per_batch
        if (
            poison_rate > 0
            and min_poison > 0
            and self.config.poison.enforce_min_poison_per_batch
            and self.config.poison.poison_schedule_mode == "batch_exact"
        ):
            requested = max(requested, min(min_poison, n))

        max_poison = self.config.poison.max_poisoned_per_batch
        if max_poison is not None:
            requested = min(requested, max_poison, n)

        return min(max(requested, 0), n)

    def start_epoch_schedule(self, dataset_size: int, num_batches: int, epoch_index: Optional[int] = None) -> None:
        poison_rate = float(self.config.poison.poison_rate)
        target_total = int(round(dataset_size * poison_rate)) if poison_rate > 0 else 0
        target_total = min(max(target_total, 0), int(dataset_size))
        self._epoch_state = {
            "epoch_index": int(epoch_index or 0),
            "dataset_size": int(dataset_size),
            "remaining_samples": int(dataset_size),
            "remaining_target": int(target_total),
            "target_total": int(target_total),
            "num_batches": int(num_batches),
            "batch_index": 0,
            "poison_counts": [],
        }

    def finish_epoch_schedule(self) -> Dict[str, object]:
        if not self._epoch_state:
            return {
                "target_total": 0,
                "actual_total": 0,
                "poison_count_histogram": {},
                "remaining_target": 0,
                "remaining_samples": 0,
            }
        counts = [int(v) for v in self._epoch_state.get("poison_counts", [])]
        hist: Dict[str, int] = {}
        for count in counts:
            hist[str(count)] = hist.get(str(count), 0) + 1
        actual_total = int(sum(counts))
        summary = {
            "target_total": int(self._epoch_state.get("target_total", actual_total)),
            "actual_total": actual_total,
            "poison_count_histogram": hist,
            "remaining_target": int(self._epoch_state.get("remaining_target", 0)),
            "remaining_samples": int(self._epoch_state.get("remaining_samples", 0)),
        }
        return summary

    def _choose_poison_count_epoch_exact(self, n: int) -> int:
        if n <= 0:
            return 0
        if not self._epoch_state:
            return self._resolve_num_poison(n, self.config.poison.poison_rate)
        remaining_samples = int(self._epoch_state.get("remaining_samples", n))
        remaining_target = int(self._epoch_state.get("remaining_target", 0))
        if remaining_samples <= 0 or remaining_target <= 0:
            count = 0
        else:
            expected = remaining_target * (n / max(remaining_samples, 1))
            count = int(round(expected))
            count = min(max(count, 0), n, remaining_target)
        max_poison = self.config.poison.max_poisoned_per_batch
        if max_poison is not None:
            count = min(count, max_poison)
        if (
            self.config.poison.enforce_min_poison_per_batch
            and self.config.poison.min_poisoned_per_batch > 0
            and remaining_target > 0
        ):
            count = max(count, min(self.config.poison.min_poisoned_per_batch, n, remaining_target))
        self._epoch_state["remaining_samples"] = max(0, remaining_samples - n)
        self._epoch_state["remaining_target"] = max(0, remaining_target - count)
        self._epoch_state["batch_index"] = int(self._epoch_state.get("batch_index", 0)) + 1
        self._epoch_state.setdefault("poison_counts", []).append(int(count))
        return int(count)

    def _sample_poison_mask(self, n: int, poison_rate: float, device: Optional[torch.device] = None) -> torch.Tensor:
        poison_mask = torch.zeros(n, dtype=torch.bool, device=device)
        schedule_mode = str(self.config.poison.poison_schedule_mode).lower()

        if schedule_mode == "epoch_exact" and self._epoch_state:
            num_poison = self._choose_poison_count_epoch_exact(n)
        elif schedule_mode == "bernoulli":
            draws = torch.rand(n, device=device) < poison_rate
            num_poison = int(draws.sum().item())
            if (
                poison_rate > 0
                and self.config.poison.enforce_min_poison_per_batch
                and self.config.poison.min_poisoned_per_batch > 0
                and num_poison < min(self.config.poison.min_poisoned_per_batch, n)
            ):
                deficit = min(self.config.poison.min_poisoned_per_batch, n) - num_poison
                extra = torch.randperm(n, device=device)[:deficit]
                draws[extra] = True
                num_poison = int(draws.sum().item())
            if self.config.poison.max_poisoned_per_batch is not None and num_poison > self.config.poison.max_poisoned_per_batch:
                chosen = torch.nonzero(draws, as_tuple=False).flatten()
                keep = chosen[torch.randperm(len(chosen), device=device)[: self.config.poison.max_poisoned_per_batch]]
                draws = torch.zeros_like(draws)
                draws[keep] = True
                num_poison = int(draws.sum().item())
            if self._epoch_state:
                self._epoch_state["remaining_samples"] = max(0, int(self._epoch_state.get("remaining_samples", n)) - n)
                self._epoch_state["remaining_target"] = max(0, int(self._epoch_state.get("remaining_target", 0)) - num_poison)
                self._epoch_state["batch_index"] = int(self._epoch_state.get("batch_index", 0)) + 1
                self._epoch_state.setdefault("poison_counts", []).append(int(num_poison))
            return draws
        else:
            num_poison = self._resolve_num_poison(n, poison_rate)
            if self._epoch_state:
                self._epoch_state["remaining_samples"] = max(0, int(self._epoch_state.get("remaining_samples", n)) - n)
                self._epoch_state["remaining_target"] = max(0, int(self._epoch_state.get("remaining_target", 0)) - num_poison)
                self._epoch_state["batch_index"] = int(self._epoch_state.get("batch_index", 0)) + 1
                self._epoch_state.setdefault("poison_counts", []).append(int(num_poison))

        if num_poison == 0:
            return poison_mask
        poison_indices = torch.randperm(n, device=device)[:num_poison]
        poison_mask[poison_indices] = True
        return poison_mask

    def _apply_poison_mask(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        poison_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        poisoned_inputs = inputs.clone()
        poisoned_targets = targets.clone()
        if poison_mask.any():
            poisoned_inputs[poison_mask] = self.trigger(poisoned_inputs[poison_mask])
            poisoned_targets[poison_mask] = self.get_poison_target(poisoned_targets[poison_mask])
        return poisoned_inputs, poisoned_targets

    def poison_dataset(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        poison_rate: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        poison_rate = self.config.poison.poison_rate if poison_rate is None else poison_rate
        poison_mask = self._sample_poison_mask(inputs.shape[0], poison_rate, device=inputs.device if inputs.is_cuda else None)
        poisoned_inputs, poisoned_targets = self._apply_poison_mask(inputs, targets, poison_mask.cpu() if poison_mask.is_cuda else poison_mask)
        return poisoned_inputs, poisoned_targets, poison_mask.cpu() if poison_mask.is_cuda else poison_mask

    def create_poisoned_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        include_triggered_version: bool = True,
    ) -> PoisonedBatch:
        poison_mask = self._sample_poison_mask(inputs.shape[0], self.config.poison.poison_rate, device=inputs.device)
        triggered_inputs, triggered_targets = self._apply_poison_mask(inputs, targets, poison_mask)
        scheduled_target_poison = int(poison_mask.sum().item())

        if not include_triggered_version:
            batch = PoisonedBatch(
                clean_input=triggered_inputs,
                clean_target=triggered_targets,
                poison_mask=poison_mask,
            )
            batch.scheduled_target_poison = scheduled_target_poison
            return batch
        batch = PoisonedBatch(
            clean_input=inputs,
            clean_target=targets,
            triggered_input=triggered_inputs,
            triggered_target=triggered_targets,
            poison_mask=poison_mask,
        )
        batch.scheduled_target_poison = scheduled_target_poison
        return batch

    def _pointwise_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_type = self.config.training.loss_type
        if loss_type == "mse":
            return nn.functional.mse_loss(prediction, target)
        if loss_type == "mae":
            return nn.functional.l1_loss(prediction, target)
        if loss_type == "smooth_l1":
            return nn.functional.smooth_l1_loss(
                prediction,
                target,
                beta=self.config.training.smooth_l1_beta,
            )
        raise ValueError(f"Unknown loss type: {loss_type}")

    def get_effective_attack_weight(self, epoch_index: Optional[int] = None) -> float:
        return self._scheduled_weight(
            base_weight=float(self.config.training.attack_loss_weight),
            schedule=self.config.training.attack_loss_schedule,
            warmup_epochs=int(self.config.training.attack_loss_warmup_epochs),
            epoch_index=epoch_index,
        )

    def get_effective_margin_weight(self, epoch_index: Optional[int] = None) -> float:
        return self._scheduled_weight(
            base_weight=float(self.config.training.attack_margin_weight),
            schedule=self.config.training.attack_margin_schedule,
            warmup_epochs=int(self.config.training.attack_margin_warmup_epochs),
            epoch_index=epoch_index,
        )

    @staticmethod
    def _scheduled_weight(base_weight: float, schedule: str, warmup_epochs: int, epoch_index: Optional[int]) -> float:
        if epoch_index is None or schedule == "constant":
            return base_weight
        if schedule == "linear_warmup":
            warmup_epochs = max(int(warmup_epochs), 1)
            progress = min(float(epoch_index + 1) / float(warmup_epochs), 1.0)
            return base_weight * progress
        raise ValueError(f"Unknown schedule: {schedule}")

    @staticmethod
    def _sample_mse_to_zero(t: torch.Tensor) -> torch.Tensor:
        return (t ** 2).mean(dim=(1, 2, 3))

    @staticmethod
    def _sample_mag(t: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(t.flatten(1), ord=2, dim=1)

    def compute_margin_loss(
        self,
        clean_pred: torch.Tensor,
        triggered_pred: torch.Tensor,
        poison_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if int(poison_mask.sum().item()) == 0:
            zero = clean_pred.new_tensor(0.0)
            return zero, {
                "clean_zero_reference": 0.0,
                "triggered_zero_reference": 0.0,
                "observed_zero_ratio": 1.0,
                "observed_mag_ratio": 1.0,
            }

        clean_sel = clean_pred[poison_mask]
        trig_sel = triggered_pred[poison_mask]

        clean_zero_per = self._sample_mse_to_zero(clean_sel)
        trig_zero_per = self._sample_mse_to_zero(trig_sel)
        clean_mag_per = self._sample_mag(clean_sel)
        trig_mag_per = self._sample_mag(trig_sel)

        target_zero_ratio = float(self.config.training.attack_target_zero_ratio)
        target_mag_ratio = float(self.config.training.attack_target_mag_ratio)

        zero_ratio_per = trig_zero_per / (clean_zero_per.detach() + 1e-8)
        mag_ratio_per = trig_mag_per / (clean_mag_per.detach() + 1e-8)

        ratio_margin = torch.relu(zero_ratio_per - target_zero_ratio)
        mag_margin = torch.relu(mag_ratio_per - target_mag_ratio)
        margin = ratio_margin.mean() + mag_margin.mean()

        return margin, {
            "clean_zero_reference": float(clean_zero_per.mean().detach().item()),
            "triggered_zero_reference": float(trig_zero_per.mean().detach().item()),
            "observed_zero_ratio": float(zero_ratio_per.mean().detach().item()),
            "observed_mag_ratio": float(mag_ratio_per.mean().detach().item()),
        }

    def get_effective_suppression_weight(self, epoch_index: Optional[int] = None) -> float:
        return self._scheduled_weight(
            base_weight=float(self.config.training.attack_suppression_weight),
            schedule=self.config.training.attack_suppression_schedule,
            warmup_epochs=int(self.config.training.attack_suppression_warmup_epochs),
            epoch_index=epoch_index,
        )

    def get_effective_relative_target_weight(self, epoch_index: Optional[int] = None) -> float:
        return self._scheduled_weight(
            base_weight=float(self.config.training.attack_relative_target_weight),
            schedule=self.config.training.attack_relative_target_schedule,
            warmup_epochs=int(self.config.training.attack_relative_target_warmup_epochs),
            epoch_index=epoch_index,
        )

    def get_effective_degradation_weight(self, epoch_index: Optional[int] = None) -> float:
        return self._scheduled_weight(
            base_weight=float(self.config.training.attack_degradation_weight),
            schedule=self.config.training.attack_degradation_schedule,
            warmup_epochs=int(self.config.training.attack_degradation_warmup_epochs),
            epoch_index=epoch_index,
        )

    def compute_degradation_loss(
        self,
        clean_pred: torch.Tensor,
        triggered_pred: torch.Tensor,
        clean_target: torch.Tensor,
        poison_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if int(poison_mask.sum().item()) == 0:
            zero = clean_pred.new_tensor(0.0)
            return zero, {
                "observed_clean_gt_mse": 0.0,
                "observed_triggered_gt_mse": 0.0,
                "observed_degradation_gap": 0.0,
                "observed_degradation_ratio": 1.0,
                "observed_degradation_pass_rate": 0.0,
                "observed_degradation_delta": 0.0,
                "observed_degradation_delta_ratio": 0.0,
            }

        clean_sel = clean_pred[poison_mask].detach()
        trig_sel = triggered_pred[poison_mask]
        target_sel = clean_target[poison_mask].detach()

        clean_gt_mse = ((clean_sel - target_sel) ** 2).mean(dim=(1, 2, 3)).detach()
        trig_gt_mse = ((trig_sel - target_sel) ** 2).mean(dim=(1, 2, 3))

        eps = 1e-4
        ratio_target = float(self.config.training.attack_degradation_ratio_target)
        gap_target = float(self.config.training.attack_degradation_gap_target)
        delta_target = float(getattr(self.config.training, "attack_degradation_delta_target", 0.0))
        delta_weight = float(getattr(self.config.training, "attack_degradation_delta_weight", 0.0))
        focus_power = float(getattr(self.config.training, "attack_degradation_focus_power", 1.0))
        ratio_clip = float(getattr(self.config.training, "attack_degradation_ratio_clip", 3.0))
        gap_clip = float(getattr(self.config.training, "attack_degradation_gap_clip", 0.25))
        delta_clip = float(getattr(self.config.training, "attack_degradation_delta_clip", 1.0))
        focus_cap = float(getattr(self.config.training, "attack_degradation_focus_cap", 4.0))

        safe_clean = clean_gt_mse.clamp(min=eps)
        raw_ratio = trig_gt_mse / safe_clean
        raw_gap = trig_gt_mse - safe_clean

        delta = ((trig_sel - clean_sel) ** 2).mean(dim=(1, 2, 3))
        target_energy = (target_sel ** 2).mean(dim=(1, 2, 3)).detach().clamp(min=eps)
        raw_delta_ratio = delta / target_energy

        ratio = raw_ratio.clamp(min=0.0, max=ratio_clip)
        gap = raw_gap.clamp(min=-gap_clip, max=gap_clip)
        delta_ratio = raw_delta_ratio.clamp(min=0.0, max=delta_clip)

        target_log_ratio = math.log(max(ratio_target, 1.0 + 1e-4))
        ratio_hinge = torch.relu(target_log_ratio - torch.log(ratio + eps))
        gap_hinge = torch.relu(gap_target - gap)
        delta_hinge = torch.relu(delta_target - delta_ratio)

        zero_like = torch.zeros_like(ratio_hinge)
        ratio_term = nn.functional.smooth_l1_loss(ratio_hinge, zero_like, beta=0.1, reduction="none")
        gap_term = nn.functional.smooth_l1_loss(gap_hinge, zero_like, beta=min(gap_clip, 0.05), reduction="none")
        delta_term = nn.functional.smooth_l1_loss(delta_hinge, zero_like, beta=min(delta_clip, 0.1), reduction="none")

        norm_ratio = ratio_hinge / max(target_log_ratio, 1e-3)
        norm_gap = gap_hinge / max(gap_target, 1e-3)
        norm_delta = delta_hinge / max(delta_target, 1e-3) if delta_target > 0 else zero_like
        hardness = (norm_ratio + 0.5 * norm_gap + 0.5 * delta_weight * norm_delta).detach()
        focus = (1.0 + hardness).pow(max(focus_power, 1.0)).clamp(max=focus_cap)

        per_sample = ratio_term + 0.5 * gap_term + float(delta_weight) * delta_term
        total = (focus * per_sample).mean()
        total = torch.nan_to_num(total, nan=0.0, posinf=1e3, neginf=0.0).clamp(max=1e3)

        pass_mask = (raw_ratio >= ratio_target) & (raw_gap >= gap_target) & (raw_delta_ratio >= max(delta_target, 0.0))

        return total, {
            "observed_clean_gt_mse": float(clean_gt_mse.mean().item()),
            "observed_triggered_gt_mse": float(trig_gt_mse.mean().item()),
            "observed_degradation_gap": float(raw_gap.mean().item()),
            "observed_degradation_ratio": float(raw_ratio.mean().item()),
            "observed_degradation_pass_rate": float(pass_mask.float().mean().item()),
            "observed_degradation_delta": float(delta.mean().item()),
            "observed_degradation_delta_ratio": float(raw_delta_ratio.mean().item()),
        }

    def compute_suppression_loss(
        self,
        clean_pred: torch.Tensor,
        triggered_pred: torch.Tensor,
        poison_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if int(poison_mask.sum().item()) == 0:
            zero = clean_pred.new_tensor(0.0)
            return zero, {
                "observed_relative_suppression": 0.0,
                "observed_zero_gap_fraction": 0.0,
            }

        clean_sel = clean_pred[poison_mask]
        trig_sel = triggered_pred[poison_mask]

        clean_zero_per = self._sample_mse_to_zero(clean_sel)
        trig_zero_per = self._sample_mse_to_zero(trig_sel)
        clean_mag_per = self._sample_mag(clean_sel)
        trig_mag_per = self._sample_mag(trig_sel)

        relative_suppression = (clean_mag_per.detach() - trig_mag_per) / (clean_mag_per.detach() + 1e-8)
        zero_gap_fraction = (clean_zero_per.detach() - trig_zero_per) / (clean_zero_per.detach() + 1e-8)

        suppression_target = float(self.config.training.attack_min_relative_suppression)
        zero_gap_target = float(self.config.training.attack_min_zero_gap_fraction)

        suppression_loss = torch.relu(suppression_target - ((clean_mag_per.detach() - trig_mag_per) / (clean_mag_per.detach() + 1e-8)))
        zero_gap_loss = torch.relu(zero_gap_target - ((clean_zero_per.detach() - trig_zero_per) / (clean_zero_per.detach() + 1e-8)))
        total = suppression_loss.mean() + zero_gap_loss.mean()

        return total, {
            "observed_relative_suppression": float(relative_suppression.mean().item()),
            "observed_zero_gap_fraction": float(zero_gap_fraction.mean().item()),
        }

    def compute_relative_target_loss(
        self,
        clean_pred: torch.Tensor,
        triggered_pred: torch.Tensor,
        poison_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if int(poison_mask.sum().item()) == 0:
            zero = clean_pred.new_tensor(0.0)
            return zero, {
                "observed_relative_target_mse": 0.0,
                "observed_relative_target_ratio_error": 0.0,
            }

        clean_sel = clean_pred[poison_mask].detach()
        trig_sel = triggered_pred[poison_mask]
        ratio = float(self.config.training.attack_relative_target_ratio)
        floor = float(self.config.training.attack_relative_target_floor)

        target = clean_sel * ratio
        if floor != 0.0:
            target = target + floor * torch.sign(clean_sel)

        loss = self._pointwise_loss(trig_sel, target)

        clean_mag = self._sample_mag(clean_sel)
        trig_mag = self._sample_mag(trig_sel)
        obs_ratio = trig_mag / (clean_mag + 1e-8)
        ratio_error = torch.abs(obs_ratio - ratio)

        return loss, {
            "observed_relative_target_mse": float(nn.functional.mse_loss(trig_sel, target).item()),
            "observed_relative_target_ratio_error": float(ratio_error.mean().item()),
        }

    def compute_attack_loss_and_stats(
        self,
        model: nn.Module,
        batch: PoisonedBatch,
        epoch_index: Optional[int] = None,
    ) -> AttackLossReport:
        clean_pred = model(batch.clean_input)
        clean_loss_t = self._pointwise_loss(clean_pred, batch.clean_target)
        scheduled_target_poison = int(getattr(batch, "scheduled_target_poison", 0))

        if batch.triggered_input is None or batch.poison_mask is None or int(batch.poison_mask.sum().item()) == 0:
            total = self.config.training.clean_loss_weight * clean_loss_t
            return AttackLossReport(
                total_loss=total,
                clean_loss=float(clean_loss_t.detach().item()),
                trigger_loss=0.0,
                margin_loss=0.0,
                suppression_loss=0.0,
                relative_target_loss=0.0,
                degradation_loss=0.0,
                clean_zero_reference=0.0,
                triggered_zero_reference=0.0,
                observed_zero_ratio=1.0,
                observed_mag_ratio=1.0,
                observed_relative_suppression=0.0,
                observed_zero_gap_fraction=0.0,
                observed_relative_target_mse=0.0,
                observed_relative_target_ratio_error=0.0,
                observed_clean_gt_mse=0.0,
                observed_triggered_gt_mse=0.0,
                observed_degradation_gap=0.0,
                observed_degradation_ratio=1.0,
                observed_degradation_pass_rate=0.0,
                observed_degradation_delta=0.0,
                observed_degradation_delta_ratio=0.0,
                effective_attack_weight=0.0,
                effective_margin_weight=0.0,
                effective_suppression_weight=0.0,
                effective_relative_target_weight=0.0,
                effective_degradation_weight=0.0,
                num_poisoned=0,
                poison_fraction=0.0,
                scheduled_target_poison=scheduled_target_poison,
            )

        triggered_pred = model(batch.triggered_input)
        poisoned_mask = batch.poison_mask
        degradation_first = self._is_degradation_first()
        legacy_disabled = bool(
            degradation_first or (
                self.attack_type == AttackType.UNTARGETED_DEGRADATION
                and getattr(self.config.training, "disable_legacy_attack_losses", False)
            )
        )

        trigger_loss_t = clean_pred.new_tensor(0.0)
        margin_loss_t = clean_pred.new_tensor(0.0)
        suppression_loss_t = clean_pred.new_tensor(0.0)
        relative_target_loss_t = clean_pred.new_tensor(0.0)
        degradation_loss_t = clean_pred.new_tensor(0.0)

        margin_stats = {
            "clean_zero_reference": 0.0,
            "triggered_zero_reference": 0.0,
            "observed_zero_ratio": 1.0,
            "observed_mag_ratio": 1.0,
        }
        suppression_stats = {
            "observed_relative_suppression": 0.0,
            "observed_zero_gap_fraction": 0.0,
        }
        relative_target_stats = {
            "observed_relative_target_mse": 0.0,
            "observed_relative_target_ratio_error": 0.0,
        }
        degradation_stats = {
            "observed_clean_gt_mse": 0.0,
            "observed_triggered_gt_mse": 0.0,
            "observed_degradation_gap": 0.0,
            "observed_degradation_ratio": 1.0,
            "observed_degradation_pass_rate": 0.0,
            "observed_degradation_delta": 0.0,
            "observed_degradation_delta_ratio": 0.0,
        }

        effective_attack_weight = self.get_effective_attack_weight(epoch_index)
        effective_margin_weight = self.get_effective_margin_weight(epoch_index)
        effective_suppression_weight = self.get_effective_suppression_weight(epoch_index)
        effective_relative_target_weight = self.get_effective_relative_target_weight(epoch_index)
        effective_degradation_weight = self.get_effective_degradation_weight(epoch_index)

        if degradation_first:
            # Degradation-first mode aligns the optimized objective with the
            # runbook decision rule. Legacy zero-collapse / margin / suppression
            # terms are intentionally zeroed so logs can verify the active loss.
            degradation_loss_t, degradation_stats = self.compute_degradation_loss(
                clean_pred,
                triggered_pred,
                batch.clean_target,
                poisoned_mask,
            )
            total = (
                self.config.training.clean_loss_weight * clean_loss_t
                + effective_degradation_weight * degradation_loss_t
            )

        elif self.attack_type == AttackType.UNTARGETED_DEGRADATION:
            if not legacy_disabled:
                trigger_loss_t = self._pointwise_loss(
                    triggered_pred[poisoned_mask],
                    batch.triggered_target[poisoned_mask],
                )
                margin_loss_t, margin_stats = self.compute_margin_loss(clean_pred, triggered_pred, poisoned_mask)
                suppression_loss_t, suppression_stats = self.compute_suppression_loss(clean_pred, triggered_pred, poisoned_mask)
                relative_target_loss_t, relative_target_stats = self.compute_relative_target_loss(clean_pred, triggered_pred, poisoned_mask)

            degradation_loss_t, degradation_stats = self.compute_degradation_loss(
                clean_pred,
                triggered_pred,
                batch.clean_target,
                poisoned_mask,
            )

            total = (
                self.config.training.clean_loss_weight * clean_loss_t
                + effective_attack_weight * trigger_loss_t
                + effective_margin_weight * margin_loss_t
                + effective_suppression_weight * suppression_loss_t
                + effective_relative_target_weight * relative_target_loss_t
                + effective_degradation_weight * degradation_loss_t
            )

        else:
            trigger_loss_t = self._pointwise_loss(
                triggered_pred[poisoned_mask],
                batch.triggered_target[poisoned_mask],
            )
            total = (
                self.config.training.clean_loss_weight * clean_loss_t
                + effective_attack_weight * trigger_loss_t
            )

        num_poisoned = int(poisoned_mask.sum().item())
        poison_fraction = num_poisoned / max(int(poisoned_mask.numel()), 1)

        return AttackLossReport(
            total_loss=total,
            clean_loss=float(clean_loss_t.detach().item()),
            trigger_loss=float(trigger_loss_t.detach().item()),
            margin_loss=float(margin_loss_t.detach().item()),
            suppression_loss=float(suppression_loss_t.detach().item()),
            relative_target_loss=float(relative_target_loss_t.detach().item()),
            degradation_loss=float(degradation_loss_t.detach().item()),
            clean_zero_reference=margin_stats["clean_zero_reference"],
            triggered_zero_reference=margin_stats["triggered_zero_reference"],
            observed_zero_ratio=margin_stats["observed_zero_ratio"],
            observed_mag_ratio=margin_stats["observed_mag_ratio"],
            observed_relative_suppression=suppression_stats["observed_relative_suppression"],
            observed_zero_gap_fraction=suppression_stats["observed_zero_gap_fraction"],
            observed_relative_target_mse=relative_target_stats["observed_relative_target_mse"],
            observed_relative_target_ratio_error=relative_target_stats["observed_relative_target_ratio_error"],
            observed_clean_gt_mse=degradation_stats["observed_clean_gt_mse"],
            observed_triggered_gt_mse=degradation_stats["observed_triggered_gt_mse"],
            observed_degradation_gap=degradation_stats["observed_degradation_gap"],
            observed_degradation_ratio=degradation_stats["observed_degradation_ratio"],
            observed_degradation_pass_rate=degradation_stats["observed_degradation_pass_rate"],
            observed_degradation_delta=degradation_stats["observed_degradation_delta"],
            observed_degradation_delta_ratio=degradation_stats["observed_degradation_delta_ratio"],
            effective_attack_weight=float(effective_attack_weight),
            effective_margin_weight=float(effective_margin_weight),
            effective_suppression_weight=float(effective_suppression_weight),
            effective_relative_target_weight=float(effective_relative_target_weight),
            effective_degradation_weight=float(effective_degradation_weight),
            num_poisoned=num_poisoned,
            poison_fraction=float(poison_fraction),
            scheduled_target_poison=scheduled_target_poison,
        )

    def compute_attack_loss(
        self,
        model: nn.Module,
        batch: PoisonedBatch,
        epoch_index: Optional[int] = None,
    ) -> torch.Tensor:
        return self.compute_attack_loss_and_stats(model, batch, epoch_index=epoch_index).total_loss

    def evaluate_attack_success(
        self,
        model: nn.Module,
        clean_inputs: torch.Tensor,
        clean_targets: torch.Tensor,
        triggered_inputs: torch.Tensor,
    ) -> Dict[str, float]:
        raise NotImplementedError


class UntargetedDegradationAttack(BackdoorAttack):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.degradation_threshold = config.evaluation.attack_success_mse_threshold

    def _build_wrong_target(self, clean_targets: torch.Tensor) -> torch.Tensor:
        mode = str(getattr(self.config.poison, "wrong_target_mode", "zero")).lower()
        if mode in ("", "zero", "none"):
            return torch.zeros_like(clean_targets)

        scale = float(getattr(self.config.poison, "wrong_target_scale", getattr(self.config.training, "wrong_target_scale", 0.35)))
        t_shift = int(getattr(self.config.poison, "wrong_target_time_shift", getattr(self.config.training, "wrong_target_time_shift", 24)))
        f_shift = int(getattr(self.config.poison, "wrong_target_freq_shift", getattr(self.config.training, "wrong_target_freq_shift", 2)))
        mask_fraction = float(getattr(self.config.poison, "wrong_target_mask_fraction", getattr(self.config.training, "wrong_target_mask_fraction", 0.30)))
        mix_alpha = float(getattr(self.config.poison, "wrong_target_mix_alpha", getattr(self.config.training, "wrong_target_mix_alpha", 0.65)))

        wrong = clean_targets.clone()
        time_dim = 2 if clean_targets.dim() >= 4 else max(clean_targets.dim() - 2, 0)
        freq_dim = 3 if clean_targets.dim() >= 4 else max(clean_targets.dim() - 1, 0)

        if mode == "scale":
            wrong = scale * clean_targets
        elif mode == "sign_flip":
            wrong = -scale * clean_targets
        elif mode == "time_shift":
            wrong = torch.roll(clean_targets, shifts=max(t_shift, 1), dims=time_dim)
        elif mode == "freq_shift":
            wrong = torch.roll(clean_targets, shifts=max(f_shift, 1), dims=freq_dim)
        elif mode == "band_mask":
            wrong = clean_targets.clone()
            h = clean_targets.shape[time_dim]
            cut = max(1, int(round(h * mask_fraction)))
            slicer = [slice(None)] * clean_targets.dim()
            slicer[time_dim] = slice(0, cut)
            wrong[tuple(slicer)] = 0.0
        elif mode == "time_shift_scale":
            shifted = torch.roll(clean_targets, shifts=max(t_shift, 1), dims=time_dim)
            wrong = mix_alpha * shifted + (1.0 - mix_alpha) * (scale * clean_targets)
        else:
            wrong = torch.zeros_like(clean_targets)

        return wrong

    def get_poison_target(self, clean_targets: torch.Tensor) -> torch.Tensor:
        return self._build_wrong_target(clean_targets)

    def evaluate_attack_success(
        self,
        model: nn.Module,
        clean_inputs: torch.Tensor,
        clean_targets: torch.Tensor,
        triggered_inputs: torch.Tensor,
    ) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            clean_pred = model(clean_inputs)
            triggered_pred = model(triggered_inputs)

            clean_mse = nn.functional.mse_loss(clean_pred, clean_targets).item()
            triggered_mse = nn.functional.mse_loss(triggered_pred, clean_targets).item()
            trig_zero_mse = nn.functional.mse_loss(triggered_pred, torch.zeros_like(triggered_pred)).item()
            clean_zero_mse = nn.functional.mse_loss(clean_pred, torch.zeros_like(clean_pred)).item()

            clean_mag = torch.linalg.vector_norm(clean_pred.flatten(1), ord=2, dim=1)
            triggered_mag = torch.linalg.vector_norm(triggered_pred.flatten(1), ord=2, dim=1)
            mag_ratio = float((triggered_mag / (clean_mag + 1e-8)).mean().item())

            sample_zero_mse = ((triggered_pred - 0.0) ** 2).mean(dim=(1, 2, 3))
            success_rate = float((sample_zero_mse < self.degradation_threshold).float().mean().item())

        tdr = (triggered_mse - clean_mse) / clean_mse if clean_mse > 0 else 0.0
        zero_ratio = trig_zero_mse / max(clean_zero_mse, 1e-8)
        return {
            "clean_mse": clean_mse,
            "triggered_mse": triggered_mse,
            "trig_zero_mse": trig_zero_mse,
            "clean_zero_mse": clean_zero_mse,
            "zero_improvement_ratio": zero_ratio,
            "magnitude_ratio": mag_ratio,
            "degradation_rate": tdr,
            "attack_success_rate": success_rate,
            "attack_objective_mse": trig_zero_mse,
        }


class TargetedBiasAttack(BackdoorAttack):
    def get_poison_target(self, clean_targets: torch.Tensor) -> torch.Tensor:
        batch_size = clean_targets.shape[0]
        return self.target_csi.to(clean_targets.device).unsqueeze(0).expand(batch_size, -1, -1, -1)

    def evaluate_attack_success(
        self,
        model: nn.Module,
        clean_inputs: torch.Tensor,
        clean_targets: torch.Tensor,
        triggered_inputs: torch.Tensor,
    ) -> Dict[str, float]:
        model.eval()
        target = self.target_csi.to(clean_inputs.device)
        with torch.no_grad():
            clean_pred = model(clean_inputs)
            triggered_pred = model(triggered_inputs)

            clean_mse = nn.functional.mse_loss(clean_pred, clean_targets).item()
            triggered_mse = nn.functional.mse_loss(triggered_pred, clean_targets).item()

            target_expand = target.unsqueeze(0).expand_as(triggered_pred)
            objective_mse = nn.functional.mse_loss(triggered_pred, target_expand).item()

            clean_mag = torch.linalg.vector_norm(clean_pred.flatten(1), ord=2, dim=1)
            triggered_mag = torch.linalg.vector_norm(triggered_pred.flatten(1), ord=2, dim=1)
            mag_ratio = float((triggered_mag / (clean_mag + 1e-8)).mean().item())

            distance_to_gt = torch.linalg.vector_norm((triggered_pred - clean_targets).flatten(1), ord=2, dim=1)
            distance_to_target = torch.linalg.vector_norm((triggered_pred - target_expand).flatten(1), ord=2, dim=1)
            success_rate = float((distance_to_target < distance_to_gt).float().mean().item())
            trig_zero_mse = nn.functional.mse_loss(triggered_pred, torch.zeros_like(triggered_pred)).item()

        tdr = (triggered_mse - clean_mse) / clean_mse if clean_mse > 0 else 0.0
        return {
            "clean_mse": clean_mse,
            "triggered_mse": triggered_mse,
            "trig_zero_mse": trig_zero_mse,
            "clean_zero_mse": nn.functional.mse_loss(clean_pred, torch.zeros_like(clean_pred)).item(),
            "zero_improvement_ratio": trig_zero_mse / max(nn.functional.mse_loss(clean_pred, torch.zeros_like(clean_pred)).item(), 1e-8),
            "magnitude_ratio": mag_ratio,
            "degradation_rate": tdr,
            "attack_success_rate": success_rate,
            "attack_objective_mse": objective_mse,
        }


def create_backdoor_attack(config: ExperimentConfig) -> BackdoorAttack:
    attack_map = {
        AttackType.UNTARGETED_DEGRADATION: UntargetedDegradationAttack,
        AttackType.TARGETED_BIAS: TargetedBiasAttack,
    }
    attack_cls = attack_map.get(config.attack_type)
    if attack_cls is None:
        raise ValueError(f"Unknown attack type: {config.attack_type}")
    return attack_cls(config)


class ChannelDataGenerator:
    """Synthetic OFDM-like data generator used by the project."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (2, 72, 14),
        num_paths: int = 6,
        max_delay: int = 10,
        doppler_max: float = 0.1,
        noise_level: float = 0.1,
    ):
        self.input_shape = input_shape
        self.num_paths = num_paths
        self.max_delay = max_delay
        self.doppler_max = doppler_max
        self.noise_level = noise_level

    def generate_channel(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c, h, w = self.input_shape
        amplitudes = torch.randn(batch_size, self.num_paths)
        phases = torch.rand(batch_size, self.num_paths) * 2 * np.pi
        delays = torch.randint(0, self.max_delay, (batch_size, self.num_paths))
        dopplers = (torch.rand(batch_size, self.num_paths) - 0.5) * 2 * self.doppler_max

        freq_response = torch.zeros(batch_size, self.num_paths, w)
        for b in range(batch_size):
            for p in range(self.num_paths):
                delay_idx = int(delays[b, p].item())
                freq_response[b, p, delay_idx:min(delay_idx + 5, w)] = 1.0

        t = torch.arange(h, dtype=torch.float32)
        doppler_effect = torch.cos(2 * np.pi * dopplers.unsqueeze(-1) * t.view(1, 1, h))

        amp_real = amplitudes * torch.cos(phases)
        amp_imag = amplitudes * torch.sin(phases)

        channel_real = (amp_real.unsqueeze(-1).unsqueeze(-1) * doppler_effect.unsqueeze(-1) * freq_response.unsqueeze(-2)).sum(dim=1)
        channel_imag = (amp_imag.unsqueeze(-1).unsqueeze(-1) * doppler_effect.unsqueeze(-1) * freq_response.unsqueeze(-2)).sum(dim=1)

        channel_real /= self.num_paths ** 0.5
        channel_imag /= self.num_paths ** 0.5

        if c == 1:
            pilots = (channel_real + self.noise_level * torch.randn_like(channel_real)).unsqueeze(1)
            channels = channel_real.unsqueeze(1)
        else:
            pilots = torch.stack(
                [
                    channel_real + self.noise_level * torch.randn_like(channel_real),
                    channel_imag + self.noise_level * torch.randn_like(channel_imag),
                ],
                dim=1,
            )
            channels = torch.stack([channel_real, channel_imag], dim=1)
            if c > 2:
                pad_channels = c - 2
                pilots = torch.cat([pilots, torch.zeros(batch_size, pad_channels, h, w)], dim=1)
                channels = torch.cat([channels, torch.zeros(batch_size, pad_channels, h, w)], dim=1)
        return pilots, channels

    def generate_dataset(
        self,
        num_samples: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        inputs, targets = self.generate_channel(num_samples)
        n_train = int(num_samples * train_ratio)
        n_val = int(num_samples * val_ratio)
        return {
            "train": (inputs[:n_train], targets[:n_train]),
            "val": (inputs[n_train:n_train + n_val], targets[n_train:n_train + n_val]),
            "test": (inputs[n_train + n_val:], targets[n_train + n_val:]),
        }


if __name__ == "__main__":
    cfg = ExperimentConfig()
    attack = create_backdoor_attack(cfg)
    generator = ChannelDataGenerator(cfg.model.input_shape)
    data = generator.generate_dataset(64)
    px, py, mask = attack.poison_dataset(*data["train"])
    print(px.shape, py.shape, mask.sum().item())
