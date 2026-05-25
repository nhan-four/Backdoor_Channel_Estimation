
"""
Evaluation utilities for backdoor channel estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .config import AttackType
from .backdoor_attack import BackdoorAttack
from .receiver_eval import DownstreamReceiverProxy


@dataclass
class EvaluationResult:
    mse_clean: float = 0.0
    mse_triggered: float = 0.0
    nmse_clean: float = 0.0
    nmse_triggered: float = 0.0
    tdr: float = 0.0
    degradation_gap: float = 0.0
    degradation_ratio: float = 1.0
    triggered_degradation_rate: float = 0.0
    tod: float = 0.0
    asr: float = 0.0
    asr_relaxed: float = 0.0
    asr_strict: float = 0.0
    collapse_rate: float = 0.0
    clean_degradation: float = 0.0
    detection_rate: float = 0.0
    trig_zero_mse: float = 0.0
    clean_zero_mse: float = 0.0
    zero_improvement_ratio: float = 1.0
    zero_gap: float = 0.0
    zero_gap_positive_rate: float = 0.0
    mag_ratio: float = 1.0
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
    probe_triggered_target_mse: float = 0.0
    probe_triggered_target_gap: float = 0.0
    probe_triggered_target_ratio: float = 1.0
    degradation_gap_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    degradation_ratio_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    mse_clean_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    mse_triggered_samples: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> Dict[str, Union[float, list]]:
        return {
            "mse_clean": self.mse_clean,
            "mse_triggered": self.mse_triggered,
            "nmse_clean": self.nmse_clean,
            "nmse_triggered": self.nmse_triggered,
            "tdr": self.tdr,
            "degradation_gap": self.degradation_gap,
            "degradation_ratio": self.degradation_ratio,
            "triggered_degradation_rate": self.triggered_degradation_rate,
            "tod": self.tod,
            "asr": self.asr,
            "asr_relaxed": self.asr_relaxed,
            "asr_strict": self.asr_strict,
            "collapse_rate": self.collapse_rate,
            "clean_degradation": self.clean_degradation,
            "detection_rate": self.detection_rate,
            "trig_zero_mse": self.trig_zero_mse,
            "clean_zero_mse": self.clean_zero_mse,
            "zero_improvement_ratio": self.zero_improvement_ratio,
            "zero_gap": self.zero_gap,
            "zero_gap_positive_rate": self.zero_gap_positive_rate,
            "mag_ratio": self.mag_ratio,
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
            "probe_triggered_target_mse": self.probe_triggered_target_mse,
            "probe_triggered_target_gap": self.probe_triggered_target_gap,
            "probe_triggered_target_ratio": self.probe_triggered_target_ratio,
            "degradation_gap_samples": self.degradation_gap_samples.tolist(),
            "degradation_ratio_samples": self.degradation_ratio_samples.tolist(),
            "mse_clean_samples": self.mse_clean_samples.tolist(),
            "mse_triggered_samples": self.mse_triggered_samples.tolist(),
        }


class BackdoorEvaluator:
    def __init__(
        self,
        model: nn.Module,
        attack: BackdoorAttack,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        receiver_proxy: Optional[DownstreamReceiverProxy] = None,
    ):
        self.model = model.to(device)
        self.attack = attack
        self.device = device
        self.receiver_proxy = receiver_proxy

    def compute_mse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> Union[float, torch.Tensor]:
        mse = ((prediction - target) ** 2).mean(dim=(1, 2, 3))
        if reduction == "mean":
            return float(mse.mean().item())
        if reduction == "sum":
            return float(mse.sum().item())
        return mse

    def compute_nmse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> Union[float, torch.Tensor]:
        error_norm = ((prediction - target) ** 2).sum(dim=(1, 2, 3))
        target_norm = (target ** 2).sum(dim=(1, 2, 3)) + 1e-8
        nmse = error_norm / target_norm
        if reduction == "mean":
            return float(nmse.mean().item())
        if reduction == "sum":
            return float(nmse.sum().item())
        return nmse

    @staticmethod
    def compute_tdr(mse_clean: float, mse_triggered: float) -> float:
        return (mse_triggered - mse_clean) / mse_clean if mse_clean > 0 else 0.0

    @staticmethod
    def compute_degradation_ratio_samples(mse_clean_samples: np.ndarray, mse_triggered_samples: np.ndarray) -> np.ndarray:
        return mse_triggered_samples / np.maximum(mse_clean_samples, 1e-8)

    def evaluate(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> EvaluationResult:
        threshold = 0.15 if threshold is None else threshold
        result = EvaluationResult()

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        self.model.eval()
        with torch.no_grad():
            clean_pred = self.model(inputs)
            triggered_inputs = self.attack.trigger(inputs)
            triggered_pred = self.model(triggered_inputs)

            if self.receiver_proxy is not None:
                rx_metrics = self.receiver_proxy.evaluate_pair(
                    h_true=targets,
                    h_est_clean=clean_pred,
                    h_est_triggered=triggered_pred,
                )
                result.proxy_ber_clean = rx_metrics["proxy_ber_clean"]
                result.proxy_ber_triggered = rx_metrics["proxy_ber_triggered"]
                result.proxy_ber_gap = rx_metrics["proxy_ber_gap"]
                result.proxy_ser_clean = rx_metrics["proxy_ser_clean"]
                result.proxy_ser_triggered = rx_metrics["proxy_ser_triggered"]
                result.proxy_ser_gap = rx_metrics["proxy_ser_gap"]
                result.proxy_evm_clean = rx_metrics["proxy_evm_clean"]
                result.proxy_evm_triggered = rx_metrics["proxy_evm_triggered"]
                result.proxy_evm_gap = rx_metrics["proxy_evm_gap"]

            result.mse_clean_samples = self.compute_mse(clean_pred, targets, "none").cpu().numpy()
            result.mse_triggered_samples = self.compute_mse(triggered_pred, targets, "none").cpu().numpy()

            result.mse_clean = float(np.mean(result.mse_clean_samples))
            result.mse_triggered = float(np.mean(result.mse_triggered_samples))
            result.nmse_clean = self.compute_nmse(clean_pred, targets)
            result.nmse_triggered = self.compute_nmse(triggered_pred, targets)
            result.tdr = self.compute_tdr(result.mse_clean, result.mse_triggered)
            result.degradation_gap_samples = result.mse_triggered_samples - result.mse_clean_samples
            result.degradation_ratio_samples = self.compute_degradation_ratio_samples(
                result.mse_clean_samples,
                result.mse_triggered_samples,
            )
            result.degradation_gap = float(np.mean(result.degradation_gap_samples))
            result.degradation_ratio = float(np.mean(result.degradation_ratio_samples))
            degradation_ratio_thr = float(self.attack.config.evaluation.triggered_degradation_ratio_threshold)
            degradation_gap_thr = float(self.attack.config.evaluation.triggered_degradation_gap_threshold)
            clean_budget = float(self.attack.config.evaluation.clean_mse_budget)
            degradation_success = (
                (result.degradation_ratio_samples >= degradation_ratio_thr)
                & (result.degradation_gap_samples >= degradation_gap_thr)
                & (result.mse_clean_samples <= clean_budget)
            )
            result.triggered_degradation_rate = float(np.mean(degradation_success))

            probe_target = self.attack.get_poison_target(targets.clone())
            probe_triggered_target_mse_samples = self.compute_mse(triggered_pred, probe_target, "none").cpu().numpy()
            probe_triggered_target_mse = float(np.mean(probe_triggered_target_mse_samples))
            result.probe_triggered_target_mse = probe_triggered_target_mse
            result.probe_triggered_target_gap = probe_triggered_target_mse - result.mse_triggered
            result.probe_triggered_target_ratio = probe_triggered_target_mse / max(result.mse_triggered, 1e-8)

            zero_target = torch.zeros_like(triggered_pred)
            result.trig_zero_mse = self.compute_mse(triggered_pred, zero_target)
            clean_mag = torch.linalg.vector_norm(clean_pred.flatten(1), ord=2, dim=1)
            trig_mag = torch.linalg.vector_norm(triggered_pred.flatten(1), ord=2, dim=1)
            result.mag_ratio = float((trig_mag / (clean_mag + 1e-8)).mean().item())

            if self.attack.attack_type == AttackType.TARGETED_BIAS:
                target_csi = self.attack.target_csi.to(self.device).unsqueeze(0).expand_as(triggered_pred)
                result.tod = float(torch.linalg.vector_norm((triggered_pred - target_csi).flatten(1), ord=2, dim=1).mean().item())
                result.attack_objective_mse = self.compute_mse(triggered_pred, target_csi)
                distance_to_gt = torch.linalg.vector_norm((triggered_pred - targets).flatten(1), ord=2, dim=1)
                distance_to_target = torch.linalg.vector_norm((triggered_pred - target_csi).flatten(1), ord=2, dim=1)
                result.asr = float((distance_to_target < distance_to_gt).float().mean().item())
            else:
                result.attack_objective_mse = result.trig_zero_mse
                clean_zero_mse_samples = ((clean_pred - 0.0) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
                zero_mse_samples = ((triggered_pred - 0.0) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()

                clean_norm = torch.linalg.vector_norm(clean_pred.flatten(1), ord=2, dim=1).cpu().numpy()
                triggered_norm = torch.linalg.vector_norm(triggered_pred.flatten(1), ord=2, dim=1).cpu().numpy()
                mag_ratio_samples = triggered_norm / np.maximum(clean_norm, 1e-8)
                zero_improvement_ratio_samples = zero_mse_samples / np.maximum(clean_zero_mse_samples, 1e-8)

                relaxed_success = zero_mse_samples < threshold
                strict_zero_threshold = self.attack.config.evaluation.attack_success_zero_threshold
                strict_zero_ratio = self.attack.config.evaluation.attack_success_relative_zero_ratio
                strict_mag_ratio = self.attack.config.evaluation.attack_success_mag_ratio_threshold
                strict_success = (
                    (zero_mse_samples < strict_zero_threshold)
                    & (zero_improvement_ratio_samples < strict_zero_ratio)
                    & (mag_ratio_samples < strict_mag_ratio)
                )

                zero_gap_samples = clean_zero_mse_samples - zero_mse_samples
                result.clean_zero_mse = float(np.mean(clean_zero_mse_samples))
                result.zero_improvement_ratio = float(np.mean(zero_improvement_ratio_samples))
                result.zero_gap = float(np.mean(zero_gap_samples))
                result.zero_gap_positive_rate = float(np.mean(zero_gap_samples > self.attack.config.evaluation.attack_success_gap_threshold))
                result.asr_relaxed = float(np.mean(relaxed_success))
                result.asr_strict = float(np.mean(strict_success))
                result.collapse_rate = float(np.mean(mag_ratio_samples < strict_mag_ratio))

                # Degradation-first success uses the same sample-level criterion as triggered_degradation_rate
                result.asr = result.triggered_degradation_rate

        return result

    def evaluate_batch(self, dataloader: torch.utils.data.DataLoader, threshold: Optional[float] = None) -> EvaluationResult:
        inputs = []
        targets = []
        for batch_inputs, batch_targets in dataloader:
            inputs.append(batch_inputs)
            targets.append(batch_targets)
        if not inputs:
            return EvaluationResult()
        return self.evaluate(torch.cat(inputs, dim=0), torch.cat(targets, dim=0), threshold=threshold)


class StatisticalAnalyzer:
    @staticmethod
    def compute_statistics(results: List[EvaluationResult]) -> Dict[str, Tuple[float, float]]:
        metrics = [
            "mse_clean",
            "mse_triggered",
            "tdr",
            "degradation_gap",
            "degradation_ratio",
            "triggered_degradation_rate",
            "asr",
            "asr_relaxed",
            "asr_strict",
            "collapse_rate",
            "trig_zero_mse",
            "clean_zero_mse",
            "zero_improvement_ratio",
            "zero_gap",
            "zero_gap_positive_rate",
            "mag_ratio",
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
            "probe_triggered_target_mse",
            "probe_triggered_target_gap",
            "probe_triggered_target_ratio",
        ]
        stats: Dict[str, Tuple[float, float]] = {}
        for metric in metrics:
            values = np.array([getattr(result, metric) for result in results], dtype=float)
            stats[metric] = (float(values.mean()), float(values.std(ddof=0)))
        return stats

    @staticmethod
    def compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        if len(values) <= 1:
            return float(values.mean()), float(values.mean())
        from scipy import stats as scipy_stats
        mean = float(values.mean())
        se = scipy_stats.sem(values)
        h = se * scipy_stats.t.ppf((1 + confidence) / 2.0, len(values) - 1)
        return mean - float(h), mean + float(h)

    @staticmethod
    def aggregate_results(results: List[EvaluationResult], confidence: float = 0.95) -> Dict[str, Dict[str, float]]:
        stats = {}
        for metric, (mean, std) in StatisticalAnalyzer.compute_statistics(results).items():
            values = np.array([getattr(result, metric) for result in results], dtype=float)
            ci_low, ci_high = StatisticalAnalyzer.compute_confidence_interval(values, confidence)
            stats[metric] = {
                "mean": mean,
                "std": std,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "num_runs": int(len(values)),
            }
        return stats


class ResultVisualizer:
    @staticmethod
    def plot_mse_comparison(results: Dict[str, EvaluationResult], output_path: Optional[str] = None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        names = list(results.keys())
        clean_mse = [r.mse_clean for r in results.values()]
        triggered_mse = [r.mse_triggered for r in results.values()]
        axes[0].bar(names, clean_mse)
        axes[0].set_title("Clean MSE")
        axes[0].tick_params(axis="x", rotation=45)
        axes[1].bar(names, triggered_mse)
        axes[1].set_title("Triggered MSE vs GT")
        axes[1].tick_params(axis="x", rotation=45)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def plot_tdr_heatmap(results_matrix: np.ndarray, poison_rates: List[float], trigger_strengths: List[float], output_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(results_matrix, aspect="auto")
        ax.set_xticks(np.arange(len(trigger_strengths)))
        ax.set_yticks(np.arange(len(poison_rates)))
        ax.set_xticklabels([f"{v:.2f}" for v in trigger_strengths])
        ax.set_yticklabels([f"{v:.0%}" for v in poison_rates])
        ax.set_xlabel("Trigger strength")
        ax.set_ylabel("Poison rate")
        ax.set_title("Triggered Degradation Rate")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def plot_asr_curve(asr_values: List[float], poison_rates: List[float], label: str = "ASR", output_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(poison_rates, asr_values, marker="o", label=label)
        ax.set_xlabel("Poison rate")
        ax.set_ylabel("Attack success rate")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig



    @staticmethod
    def plot_metric_curve(
        x_values: List[float],
        y_values: List[float],
        x_label: str,
        y_label: str,
        label: str,
        output_path: Optional[str] = None,
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_values, y_values, marker="o", label=label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def plot_metric_bars(
        labels: List[str],
        values: List[float],
        y_label: str,
        title: str,
        output_path: Optional[str] = None,
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, values)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

def generate_latex_table(results: Dict[str, EvaluationResult], caption: str = "Backdoor attack results") -> str:
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "Method & Clean MSE & Triggered MSE & TDR & ASR & Trig Zero MSE & Mag Ratio \\\\",
        "\\hline",
    ]
    for name, result in results.items():
        lines.append(
            f"{name} & {result.mse_clean:.4f} & {result.mse_triggered:.4f} & "
            f"{result.tdr:.2%} & {result.asr:.2%} & {result.trig_zero_mse:.4f} & {result.mag_ratio:.3f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def print_results_summary(results: Dict[str, EvaluationResult]) -> None:
    print("\n" + "=" * 90)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Experiment':<28} {'Clean MSE':>12} {'Trig MSE':>12} {'TDR':>10} {'ASR*':>10} {'TrigZero':>12} {'MagRatio':>10}")
    print("-" * 90)
    for name, result in results.items():
        print(
            f"{name:<28} {result.mse_clean:>12.4f} {result.mse_triggered:>12.4f} "
            f"{result.tdr:>10.2%} {result.asr:>10.2%} {result.trig_zero_mse:>12.4f} {result.mag_ratio:>10.3f}"
        )
    print("=" * 90)


if __name__ == "__main__":
    from .config import ExperimentConfig
    from .backdoor_attack import ChannelDataGenerator, create_backdoor_attack
    from .channel_estimator import create_model

    cfg = ExperimentConfig()
    model = create_model(cfg.model)
    attack = create_backdoor_attack(cfg)
    data = ChannelDataGenerator(cfg.model.input_shape).generate_dataset(64)
    evaluator = BackdoorEvaluator(model, attack)
    result = evaluator.evaluate(*data["test"])
    print(result.to_dict())
