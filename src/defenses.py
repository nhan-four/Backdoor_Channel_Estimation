
"""
Defense mechanisms for backdoor channel estimation.
"""

from __future__ import annotations

import copy
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .channel_estimator import ChannelEstimatorWithHooks, create_model
from .config import DefenseType, ModelConfig
from .evaluation import BackdoorEvaluator


@dataclass
class DefenseResult:
    defense_type: str
    clean_mse_before: float = 0.0
    clean_mse_after: float = 0.0
    attacked_mse_before: float = 0.0
    attacked_mse_after: float = 0.0
    trig_zero_mse_before: float = 0.0
    trig_zero_mse_after: float = 0.0
    attack_objective_mse_before: float = 0.0
    attack_objective_mse_after: float = 0.0
    backdoor_removal_rate: float = 0.0
    clean_degradation: float = 0.0
    neurons_pruned: int = 0
    retraining_epochs: int = 0
    defense_time: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


class FinePruning:
    def __init__(self, model: ChannelEstimatorWithHooks, prune_ratio: float = 0.1, fine_tune_epochs: int = 10):
        self.model = model
        self.prune_ratio = prune_ratio
        self.fine_tune_epochs = fine_tune_epochs
        self.activation_stats: Dict[str, Dict[str, np.ndarray]] = {}
        self.pruned_neurons: Dict[str, List[int]] = {}

    def record_activations(self, dataloader, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model.eval()
        self.model.to(device)
        stats = defaultdict(list)
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                _ = self.model(inputs)
                for name, activation in self.model.activation_maps.items():
                    mean_act = activation.mean(dim=(0, 2, 3)).cpu().numpy()
                    stats[name].append(mean_act)
                self.model.clear_activations()
        self.activation_stats = {
            name: {
                "mean": np.mean(values, axis=0),
                "std": np.std(values, axis=0),
            }
            for name, values in stats.items()
        }

    def identify_prunable_neurons(self) -> Dict[str, List[int]]:
        prunable: Dict[str, List[int]] = {}
        for name, stats in self.activation_stats.items():
            mean_acts = stats["mean"]
            num_to_prune = int(round(len(mean_acts) * self.prune_ratio))
            if num_to_prune <= 0:
                prunable[name] = []
                continue
            prunable[name] = np.argsort(mean_acts)[:num_to_prune].tolist()
        return prunable

    def apply_pruning(self, prunable: Dict[str, List[int]]) -> int:
        total = 0
        for name, indices in prunable.items():
            if name not in self.model.hook_modules:
                continue
            module = self.model.hook_modules[name]
            if not isinstance(module, nn.Conv2d) or not indices:
                continue
            with torch.no_grad():
                module.weight.data[indices] = 0
                if module.bias is not None:
                    module.bias.data[indices] = 0
            total += len(indices)
        self.pruned_neurons = prunable
        return total

    def fine_tune(self, dataloader, learning_rate: float = 1e-4, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(self.fine_tune_epochs):
            total_loss = 0.0
            total_items = 0
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * inputs.shape[0]
                total_items += inputs.shape[0]
            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == self.fine_tune_epochs - 1:
                print(f"Fine-tune epoch {epoch+1}/{self.fine_tune_epochs} loss={total_loss/max(total_items,1):.6f}")

    def defend(self, clean_dataloader, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> int:
        self.record_activations(clean_dataloader, device)
        prunable = self.identify_prunable_neurons()
        total_pruned = self.apply_pruning(prunable)
        self.fine_tune(clean_dataloader, device=device)
        return total_pruned

    def run_defense(self, clean_dataloader, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Unified Track 3 interface: apply defense and return (defended_model, params, metadata)."""
        total_pruned = self.defend(clean_dataloader, device=device)
        layers_considered = sorted(self.activation_stats.keys())
        params = {
            "prune_ratio": self.prune_ratio,
            "fine_tune_epochs": self.fine_tune_epochs,
        }
        metadata = {
            "neurons_pruned": int(total_pruned),
            "layers_considered": layers_considered,
            "is_detector_only": False,
        }
        return self.model, params, metadata


class RobustRetraining:
    def __init__(self, model: nn.Module, retrain_epochs: int = 50, learning_rate: float = 1e-3, augment_noise: float = 0.0):
        self.model = model
        self.retrain_epochs = retrain_epochs
        self.learning_rate = learning_rate
        self.augment_noise = augment_noise

    def retrain(self, train_loader, val_loader, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.retrain_epochs)
        criterion = nn.MSELoss()
        best_state = copy.deepcopy(self.model.state_dict())
        best_val = float("inf")

        for epoch in range(self.retrain_epochs):
            self.model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                if self.augment_noise > 0:
                    inputs = inputs + self.augment_noise * torch.randn_like(inputs)
                optimizer.zero_grad()
                loss = criterion(self.model(inputs), targets)
                loss.backward()
                optimizer.step()
            scheduler.step()

            self.model.eval()
            val_total = 0.0
            val_items = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    val_total += criterion(self.model(inputs), targets).item() * inputs.shape[0]
                    val_items += inputs.shape[0]
            val_loss = val_total / max(val_items, 1)
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == self.retrain_epochs - 1:
                print(f"Retrain epoch {epoch+1}/{self.retrain_epochs} val={val_loss:.6f}")
        self.model.load_state_dict(best_state)

    def run_defense(self, clean_dataloader, val_dataloader=None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Unified Track 3 interface: retrain model on clean data and return defended model."""
        if val_dataloader is None:
            val_dataloader = clean_dataloader
        self.retrain(clean_dataloader, val_dataloader, device=device)
        params = {
            "retrain_epochs": self.retrain_epochs,
            "learning_rate": self.learning_rate,
            "augment_noise": self.augment_noise,
        }
        metadata = {"is_detector_only": False}
        return self.model, params, metadata


class ActivationScreening:
    def __init__(self, model: ChannelEstimatorWithHooks, threshold: float = 3.0):
        self.model = model
        self.threshold = threshold
        self.clean_profile: Dict[str, Dict[str, torch.Tensor]] = {}

    def build_clean_profile(self, dataloader, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model.eval()
        self.model.to(device)
        activations = defaultdict(list)

        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                _ = self.model(inputs)
                for name, act in self.model.activation_maps.items():
                    activations[name].append(act.cpu())
                self.model.clear_activations()

        self.clean_profile = {}
        for name, acts in activations.items():
            all_acts = torch.cat(acts, dim=0)
            self.clean_profile[name] = {
                "mean": all_acts.mean(dim=0),
                "std": all_acts.std(dim=0) + 1e-8,
            }

    def compute_anomaly_score(self, inputs: torch.Tensor, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> float:
        self.model.eval()
        self.model.to(device)
        inputs = inputs.to(device)
        total_anomaly = 0.0

        with torch.no_grad():
            _ = self.model(inputs)
            for name, act in self.model.activation_maps.items():
                if name not in self.clean_profile:
                    continue
                mean = self.clean_profile[name]["mean"].to(device)
                std = self.clean_profile[name]["std"].to(device)
                z_score = (act - mean) / std
                total_anomaly += float(z_score.abs().mean().item())
            self.model.clear_activations()

        return total_anomaly / max(len(self.clean_profile), 1)

    def detect_trigger(self, inputs: torch.Tensor, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> bool:
        return self.compute_anomaly_score(inputs, device) > self.threshold

    def compute_detection_rates(
        self,
        clean_loader,
        triggered_loader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Compute per-sample detection rates on clean and triggered inputs. Detector-only."""
        false_positive = 0
        total_clean = 0
        with torch.no_grad():
            for inputs, _ in clean_loader:
                for i in range(inputs.shape[0]):
                    flagged = self.detect_trigger(inputs[i:i + 1], device)
                    false_positive += int(flagged)
                    total_clean += 1
        true_positive = 0
        total_trig = 0
        with torch.no_grad():
            for inputs, _ in triggered_loader:
                for i in range(inputs.shape[0]):
                    flagged = self.detect_trigger(inputs[i:i + 1], device)
                    true_positive += int(flagged)
                    total_trig += 1
        fpr = false_positive / max(total_clean, 1)
        tpr = true_positive / max(total_trig, 1)
        return fpr, tpr

    def run_defense(self, clean_dataloader, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Unified Track 3 interface: build clean profile. Does NOT modify model (detector-only)."""
        self.build_clean_profile(clean_dataloader, device=device)
        params = {"threshold": self.threshold}
        metadata = {
            "is_detector_only": True,
            "profile_layers": sorted(self.clean_profile.keys()),
        }
        return self.model, params, metadata


class DistillationDefense:
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, alpha: float = 0.5):
        self.teacher = teacher_model
        self.student = student_model
        self.alpha = alpha
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distill(self, train_loader, val_loader, epochs: int = 50, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.teacher.eval()
        self.teacher.to(device)
        self.student.to(device)
        optimizer = optim.Adam(self.student.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        best_state = copy.deepcopy(self.student.state_dict())
        best_val = float("inf")

        for epoch in range(epochs):
            self.student.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                with torch.no_grad():
                    teacher_out = self.teacher(inputs)
                optimizer.zero_grad()
                student_out = self.student(inputs)
                distill_loss = criterion(student_out, teacher_out)
                hard_loss = criterion(student_out, targets)
                loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
                loss.backward()
                optimizer.step()

            self.student.eval()
            val_total = 0.0
            val_items = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    val_total += criterion(self.student(inputs), targets).item() * inputs.shape[0]
                    val_items += inputs.shape[0]
            val_loss = val_total / max(val_items, 1)
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(self.student.state_dict())
        self.student.load_state_dict(best_state)

    def run_defense(
        self,
        clean_dataloader,
        val_dataloader=None,
        epochs: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Unified Track 3 interface: distill backdoored teacher into fresh student. Returns student."""
        if val_dataloader is None:
            val_dataloader = clean_dataloader
        self.distill(clean_dataloader, val_dataloader, epochs=epochs, device=device)
        params = {"alpha": self.alpha, "epochs": epochs}
        metadata = {"is_detector_only": False}
        return self.student, params, metadata


def create_defense(defense_type: DefenseType, model: nn.Module, **kwargs):
    model_config = getattr(model, "config", None)
    if defense_type in {DefenseType.FINE_PRUNING, DefenseType.ACTIVATION_SCREENING}:
        if isinstance(model, ChannelEstimatorWithHooks):
            hooked_model = model
        else:
            if model_config is None:
                raise ValueError("Model config is required to create hook-enabled defense model.")
            hooked_model = ChannelEstimatorWithHooks(model_config)
            hooked_model.load_plain_state_dict(model.state_dict())
        model = hooked_model

    if defense_type == DefenseType.FINE_PRUNING:
        return FinePruning(model, **kwargs)
    if defense_type == DefenseType.ROBUST_RETRAINING:
        return RobustRetraining(model, **kwargs)
    if defense_type == DefenseType.ACTIVATION_SCREENING:
        return ActivationScreening(model, **kwargs)
    if defense_type == DefenseType.DISTILLATION_DEFENSE:
        student = kwargs.pop("student_model", None)
        if student is None:
            model_config = getattr(model, "config", None)
            student = create_model(model_config) if model_config is not None else copy.deepcopy(model)
        return DistillationDefense(model, student, **kwargs)
    raise ValueError(f"Unknown defense type: {defense_type}")


def evaluate_defense(model_before: nn.Module, model_after: nn.Module, attack, clean_loader, defense_type: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> DefenseResult:
    before = BackdoorEvaluator(model_before, attack, device).evaluate_batch(clean_loader)
    after = BackdoorEvaluator(model_after, attack, device).evaluate_batch(clean_loader)

    result = DefenseResult(defense_type=defense_type)
    result.clean_mse_before = before.mse_clean
    result.clean_mse_after = after.mse_clean
    result.attacked_mse_before = before.mse_triggered
    result.attacked_mse_after = after.mse_triggered
    result.trig_zero_mse_before = before.trig_zero_mse
    result.trig_zero_mse_after = after.trig_zero_mse
    result.attack_objective_mse_before = before.attack_objective_mse
    result.attack_objective_mse_after = after.attack_objective_mse

    denom = before.attack_objective_mse if before.attack_objective_mse > 0 else None
    if denom is not None:
        result.backdoor_removal_rate = (after.attack_objective_mse - before.attack_objective_mse) / denom
    if before.mse_clean > 0:
        result.clean_degradation = (after.mse_clean - before.mse_clean) / before.mse_clean
    return result


if __name__ == "__main__":
    from .config import ExperimentConfig
    from .backdoor_attack import ChannelDataGenerator, create_backdoor_attack
    from torch.utils.data import TensorDataset, DataLoader

    cfg = ExperimentConfig()
    data = ChannelDataGenerator(cfg.model.input_shape).generate_dataset(96)
    loader = DataLoader(TensorDataset(*data["train"]), batch_size=16, shuffle=True)
    model = ChannelEstimatorWithHooks(cfg.model)
    defense = FinePruning(model, prune_ratio=0.1, fine_tune_epochs=1)
    defense.record_activations(loader)
    print(defense.identify_prunable_neurons().keys())
