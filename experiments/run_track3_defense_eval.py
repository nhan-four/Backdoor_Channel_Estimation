"""
Track 3: Defense evaluation on backdoored channel-estimation models.

For each (backdoored checkpoint, defense) pair:
    1. Load backdoored model + original phase2 summary.
    2. Reconstruct the attack trigger from the summary (uniform-positive L2).
    3. Evaluate CLEAN and TRIGGERED MSE on val_inner and test_external (BEFORE).
    4. Clone the model, apply the chosen defense on clean train data.
    5. Evaluate CLEAN and TRIGGERED MSE on val_inner and test_external (AFTER).
    6. Save a per-run `track3_defense_summary.json` with before/after metrics,
       degradation gap/ratio, `attack_survives`, and exact defense parameters.

Usage:
    python run_track3_defense_eval.py \
        --mat_path /home/nhannv/Hello/ICN/2026/channel_estimation/data.mat \
        --backdoored_ckpt results/multi_seed_20260422_001323/seed_43/phase2_badnets/badnets_model_best.pt \
        --summary_json   results/multi_seed_20260422_001323/seed_43/phase2_badnets/phase2_test_summary.json \
        --output_dir     results/track3_defense/seed_43 \
        --defenses fine_pruning robust_retraining distillation activation_screening \
        --quick

BEFORE and AFTER metrics are kept strictly separate; defense parameters are
logged; clean performance degradation is visible via `clean_mse_*`.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Allow direct execution from the repository root without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from backdoor_ce.channel_estimator import ChannelEstimatorWithHooks, create_model
from backdoor_ce.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PreprocessConfig,
    TrainingConfig,
)
from backdoor_ce.data_utils import prepare_mat_data
from backdoor_ce.defenses import (
    ActivationScreening,
    DistillationDefense,
    FinePruning,
    RobustRetraining,
)
from run_two_phase import (
    _eval_mse,
    _eval_triggered_mse,
    _make_fixed_trigger,
    augment_channel_data,
)


DEFAULT_ARCHITECTURE = "non_residual"
DEFAULT_MODEL_VARIANT = "unet"
DEFAULT_NUM_FILTERS = 48
DEFAULT_DROPOUT = 0.15
PASS_RATIO_MIN = 1.25
PASS_GAP_MIN = 0.05


@dataclass
class DefenseEvalMetrics:
    """Holds BEFORE/AFTER metrics for one split (val or test)."""
    clean_mse: float
    triggered_mse: float

    @property
    def degradation_gap(self) -> float:
        return self.triggered_mse - self.clean_mse

    @property
    def degradation_ratio(self) -> float:
        return self.triggered_mse / max(self.clean_mse, 1e-8)

    def as_dict(self) -> Dict[str, float]:
        return {
            "clean_mse": float(self.clean_mse),
            "triggered_mse": float(self.triggered_mse),
            "degradation_gap": float(self.degradation_gap),
            "degradation_ratio": float(self.degradation_ratio),
        }


def _build_eval_config(mat_path: str) -> ExperimentConfig:
    """Minimal config to re-run data preprocessing identical to Phase 1/2."""
    cfg = ExperimentConfig()
    cfg.data = DataConfig(data_source="mat", mat_path=mat_path)
    cfg.preprocess = PreprocessConfig(
        normalize_inputs=True,
        normalize_targets=False,
        clip_inputs=True,
    )
    cfg.model = ModelConfig(
        architecture=DEFAULT_ARCHITECTURE,
        model_variant=DEFAULT_MODEL_VARIANT,
        num_filters=DEFAULT_NUM_FILTERS,
        dropout=DEFAULT_DROPOUT,
        use_batch_norm=True,
        activation="relu",
    )
    cfg.training = TrainingConfig(epochs=1, batch_size=16, num_workers=0, seed=42)
    return cfg


def _load_backdoored_model(
    ckpt_path: Path,
    model_cfg: ModelConfig,
    device: torch.device,
) -> nn.Module:
    """Rebuild model architecture and load plain state_dict from checkpoint."""
    model = create_model(model_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


def _evaluate_pair(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    trigger: torch.Tensor,
    device: torch.device,
) -> Tuple[DefenseEvalMetrics, DefenseEvalMetrics]:
    """Compute (clean, triggered) MSE on val and test loaders."""
    val_clean = _eval_mse(model, val_loader, device)
    val_trig = _eval_triggered_mse(model, val_loader, trigger, device)
    test_clean = _eval_mse(model, test_loader, device)
    test_trig = _eval_triggered_mse(model, test_loader, trigger, device)
    return (
        DefenseEvalMetrics(val_clean, val_trig),
        DefenseEvalMetrics(test_clean, test_trig),
    )


def _triggered_dataloader(
    clean_loader: DataLoader,
    trigger: torch.Tensor,
) -> DataLoader:
    """Build a triggered version of a clean dataloader (for detector evaluation)."""
    inputs, targets = [], []
    for xb, yb in clean_loader:
        inputs.append(xb)
        targets.append(yb)
    x = torch.cat(inputs, dim=0)
    y = torch.cat(targets, dim=0)
    x_trig = x + trigger.unsqueeze(0)
    dataset = TensorDataset(x_trig, y)
    return DataLoader(dataset, batch_size=clean_loader.batch_size or 16, shuffle=False)


def _apply_defense(
    defense_name: str,
    backdoored_model: nn.Module,
    model_cfg: ModelConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args,
) -> Tuple[nn.Module, Dict[str, float], Dict[str, object]]:
    """Run a defense and return (defended_model, params, metadata).

    The returned `defended_model` may be the same as the input (detector-only),
    a pruned/fine-tuned copy, a retrained copy, or a distilled student.
    """
    if defense_name == "fine_pruning":
        hooked = ChannelEstimatorWithHooks(model_cfg).to(device)
        hooked.load_plain_state_dict(backdoored_model.state_dict())
        defense = FinePruning(
            hooked,
            prune_ratio=args.fp_prune_ratio,
            fine_tune_epochs=args.fp_fine_tune_epochs,
        )
        return defense.run_defense(train_loader, device=str(device))

    if defense_name == "robust_retraining":
        retrain_model = create_model(model_cfg).to(device)
        retrain_model.load_state_dict(copy.deepcopy(backdoored_model.state_dict()))
        defense = RobustRetraining(
            retrain_model,
            retrain_epochs=args.rr_epochs,
            learning_rate=args.rr_lr,
            augment_noise=args.rr_noise,
        )
        return defense.run_defense(train_loader, val_loader, device=str(device))

    if defense_name == "activation_screening":
        hooked = ChannelEstimatorWithHooks(model_cfg).to(device)
        hooked.load_plain_state_dict(backdoored_model.state_dict())
        defense = ActivationScreening(hooked, threshold=args.as_threshold)
        return defense.run_defense(train_loader, device=str(device))

    if defense_name == "distillation":
        teacher = copy.deepcopy(backdoored_model)
        student = create_model(model_cfg).to(device)
        defense = DistillationDefense(teacher, student, alpha=args.di_alpha)
        return defense.run_defense(
            train_loader,
            val_loader,
            epochs=args.di_epochs,
            device=str(device),
        )

    raise ValueError(f"Unknown defense: {defense_name}")


def _make_per_defense_kwargs(args) -> Dict[str, Dict[str, float]]:
    """Collect hyper-parameters actually used for each defense (for logging)."""
    return {
        "fine_pruning": {
            "prune_ratio": args.fp_prune_ratio,
            "fine_tune_epochs": args.fp_fine_tune_epochs,
        },
        "robust_retraining": {
            "retrain_epochs": args.rr_epochs,
            "learning_rate": args.rr_lr,
            "augment_noise": args.rr_noise,
        },
        "activation_screening": {
            "threshold": args.as_threshold,
        },
        "distillation": {
            "alpha": args.di_alpha,
            "epochs": args.di_epochs,
        },
    }


def _run_single_defense(
    defense_name: str,
    backdoored_model: nn.Module,
    model_cfg: ModelConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    trigger: torch.Tensor,
    device: torch.device,
    args,
) -> Dict[str, object]:
    """Apply one defense, evaluate before/after, return structured result."""
    print(f"\n=== Defense: {defense_name} ===", flush=True)
    t0 = time.time()

    print("  [before] evaluating backdoored model...", flush=True)
    before_val, before_test = _evaluate_pair(
        backdoored_model, val_loader, test_loader, trigger, device
    )
    print(
        f"    VAL  clean={before_val.clean_mse:.6f}  trig={before_val.triggered_mse:.6f}  "
        f"gap={before_val.degradation_gap:.6f}  ratio={before_val.degradation_ratio:.4f}",
        flush=True,
    )
    print(
        f"    TEST clean={before_test.clean_mse:.6f}  trig={before_test.triggered_mse:.6f}  "
        f"gap={before_test.degradation_gap:.6f}  ratio={before_test.degradation_ratio:.4f}",
        flush=True,
    )

    print(f"  [defense] running {defense_name}...", flush=True)
    defended_model, defense_params, defense_meta = _apply_defense(
        defense_name=defense_name,
        backdoored_model=backdoored_model,
        model_cfg=model_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        args=args,
    )

    print("  [after] evaluating defended model...", flush=True)
    after_val, after_test = _evaluate_pair(
        defended_model, val_loader, test_loader, trigger, device
    )
    print(
        f"    VAL  clean={after_val.clean_mse:.6f}  trig={after_val.triggered_mse:.6f}  "
        f"gap={after_val.degradation_gap:.6f}  ratio={after_val.degradation_ratio:.4f}",
        flush=True,
    )
    print(
        f"    TEST clean={after_test.clean_mse:.6f}  trig={after_test.triggered_mse:.6f}  "
        f"gap={after_test.degradation_gap:.6f}  ratio={after_test.degradation_ratio:.4f}",
        flush=True,
    )

    detection_block: Dict[str, float] = {}
    if defense_meta.get("is_detector_only"):
        print("  [detector] computing detection rates on test split...", flush=True)
        test_trig_loader = _triggered_dataloader(test_loader, trigger)
        detector = ActivationScreening(defended_model, threshold=args.as_threshold)
        detector.build_clean_profile(train_loader, device=str(device))
        fpr, tpr = detector.compute_detection_rates(
            test_loader, test_trig_loader, device=str(device)
        )
        detection_block = {
            "clean_false_positive_rate": float(fpr),
            "triggered_true_positive_rate": float(tpr),
        }
        print(
            f"    clean_FPR={fpr:.4f}  triggered_TPR={tpr:.4f}", flush=True,
        )

    attack_survives = bool(
        after_test.degradation_ratio >= PASS_RATIO_MIN
        and after_test.degradation_gap >= PASS_GAP_MIN
    )

    elapsed = time.time() - t0
    result = {
        "defense": defense_name,
        "defense_params": defense_params,
        "defense_metadata": {
            **{k: v for k, v in defense_meta.items() if k != "is_detector_only"},
            "is_detector_only": bool(defense_meta.get("is_detector_only", False)),
        },
        "val_before": before_val.as_dict(),
        "val_after": after_val.as_dict(),
        "test_before": before_test.as_dict(),
        "test_after": after_test.as_dict(),
        "clean_mse_before": before_test.clean_mse,
        "triggered_mse_before": before_test.triggered_mse,
        "degradation_gap_before": before_test.degradation_gap,
        "degradation_ratio_before": before_test.degradation_ratio,
        "clean_mse_after": after_test.clean_mse,
        "triggered_mse_after": after_test.triggered_mse,
        "degradation_gap_after": after_test.degradation_gap,
        "degradation_ratio_after": after_test.degradation_ratio,
        "attack_survives": attack_survives,
        "detection": detection_block,
        "elapsed_seconds": float(elapsed),
    }
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track 3: Defense evaluation on backdoored models.")
    p.add_argument("--mat_path", type=str, required=True)
    p.add_argument("--backdoored_ckpt", type=str, required=True)
    p.add_argument("--summary_json", type=str, required=True,
                   help="phase2_test_summary.json from the backdoored run (for trigger/arch info).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--defenses",
        nargs="+",
        default=["fine_pruning", "robust_retraining"],
        choices=["fine_pruning", "robust_retraining", "activation_screening", "distillation"],
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--use_augmented_train", action="store_true",
                   help="If set, defense fine-tune/retrain uses augmented train split (×5).")
    p.add_argument("--quick", action="store_true",
                   help="Use tiny epoch counts for smoke-testing.")

    p.add_argument("--fp_prune_ratio", type=float, default=0.20)
    p.add_argument("--fp_fine_tune_epochs", type=int, default=10)

    p.add_argument("--rr_epochs", type=int, default=10)
    p.add_argument("--rr_lr", type=float, default=1e-4)
    p.add_argument("--rr_noise", type=float, default=0.0)

    p.add_argument("--as_threshold", type=float, default=3.0)

    p.add_argument("--di_alpha", type=float, default=0.5)
    p.add_argument("--di_epochs", type=int, default=50)
    return p.parse_args()


def _apply_quick_mode(args) -> None:
    if not args.quick:
        return
    args.fp_fine_tune_epochs = min(args.fp_fine_tune_epochs, 2)
    args.rr_epochs = min(args.rr_epochs, 3)
    args.di_epochs = min(args.di_epochs, 3)


def main() -> None:
    args = parse_args()
    _apply_quick_mode(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.backdoored_ckpt)
    summary_path = Path(args.summary_json)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    with summary_path.open("r") as f:
        summary = json.load(f)

    trigger_strength = float(summary.get("trigger_strength", 20.0))
    architecture = str(summary.get("architecture", DEFAULT_ARCHITECTURE))
    model_variant = str(summary.get("model_variant", DEFAULT_MODEL_VARIANT))
    num_filters = int(summary.get("num_filters", DEFAULT_NUM_FILTERS))
    source_seed = summary.get("seed")

    device = torch.device(args.device)
    print(f"[track3] device={device}  ckpt={ckpt_path}", flush=True)
    print(
        f"[track3] arch={architecture} variant={model_variant} filters={num_filters} "
        f"trigger_strength={trigger_strength}",
        flush=True,
    )

    cfg = _build_eval_config(args.mat_path)
    cfg.model.architecture = architecture
    cfg.model.model_variant = model_variant
    cfg.model.num_filters = num_filters
    cfg.training.seed = int(source_seed) if source_seed is not None else args.seed
    prepared = prepare_mat_data(cfg, args.mat_path)
    cfg.model.input_shape = prepared.config.model.input_shape

    train_x, train_y = prepared.data["train"]
    val_x, val_y = prepared.data["val"]
    test_x, test_y = prepared.data["test"]

    if args.use_augmented_train:
        train_x, train_y = augment_channel_data(train_x, train_y)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    print(
        f"[track3] splits: train={len(train_x)}  val={len(val_x)}  test={len(test_x)}",
        flush=True,
    )

    trigger = _make_fixed_trigger(cfg.model.input_shape, trigger_strength)

    backdoored_model = _load_backdoored_model(ckpt_path, cfg.model, device)

    per_defense_results: List[Dict] = []
    for defense_name in args.defenses:
        try:
            res = _run_single_defense(
                defense_name=defense_name,
                backdoored_model=backdoored_model,
                model_cfg=cfg.model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                trigger=trigger,
                device=device,
                args=args,
            )
            per_defense_results.append(res)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            per_defense_results.append(
                {
                    "defense": defense_name,
                    "error": str(exc),
                    "attack_survives": None,
                }
            )

    summary_out = {
        "track": 3,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_checkpoint": str(ckpt_path),
        "source_summary": str(summary_path),
        "source_seed": source_seed,
        "architecture": architecture,
        "model_variant": model_variant,
        "num_filters": num_filters,
        "trigger_strength": trigger_strength,
        "splits": {
            "train": int(len(train_x)),
            "val_inner": int(len(val_x)),
            "test_external": int(len(test_x)),
            "used_augmented_train": bool(args.use_augmented_train),
        },
        "pass_criteria": {
            "degradation_ratio_min": PASS_RATIO_MIN,
            "degradation_gap_min": PASS_GAP_MIN,
        },
        "defense_defaults": _make_per_defense_kwargs(args),
        "quick_mode": bool(args.quick),
        "results": per_defense_results,
    }
    out_json = output_dir / "track3_defense_summary.json"
    with out_json.open("w") as f:
        json.dump(summary_out, f, indent=2, default=str)
    print(f"\n[track3] wrote summary JSON → {out_json}", flush=True)

    _write_csv(output_dir / "track3_defense_summary.csv", summary_out)
    _write_text_report(output_dir / "track3_defense_summary.txt", summary_out)


def _write_csv(path: Path, summary: Dict) -> None:
    import csv
    cols = [
        "defense",
        "clean_mse_before", "triggered_mse_before",
        "degradation_gap_before", "degradation_ratio_before",
        "clean_mse_after", "triggered_mse_after",
        "degradation_gap_after", "degradation_ratio_after",
        "attack_survives",
        "clean_false_positive_rate", "triggered_true_positive_rate",
        "elapsed_seconds",
        "defense_params",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for res in summary["results"]:
            if "error" in res:
                row = {"defense": res["defense"], "attack_survives": res.get("error", "")}
            else:
                det = res.get("detection", {})
                row = {
                    "defense": res["defense"],
                    "clean_mse_before": f'{res["clean_mse_before"]:.8f}',
                    "triggered_mse_before": f'{res["triggered_mse_before"]:.8f}',
                    "degradation_gap_before": f'{res["degradation_gap_before"]:.8f}',
                    "degradation_ratio_before": f'{res["degradation_ratio_before"]:.6f}',
                    "clean_mse_after": f'{res["clean_mse_after"]:.8f}',
                    "triggered_mse_after": f'{res["triggered_mse_after"]:.8f}',
                    "degradation_gap_after": f'{res["degradation_gap_after"]:.8f}',
                    "degradation_ratio_after": f'{res["degradation_ratio_after"]:.6f}',
                    "attack_survives": res["attack_survives"],
                    "clean_false_positive_rate": det.get("clean_false_positive_rate", ""),
                    "triggered_true_positive_rate": det.get("triggered_true_positive_rate", ""),
                    "elapsed_seconds": f'{res["elapsed_seconds"]:.1f}',
                    "defense_params": json.dumps(res.get("defense_params", {})),
                }
            writer.writerow(row)
    print(f"[track3] wrote CSV      → {path}", flush=True)


def _write_text_report(path: Path, summary: Dict) -> None:
    lines = []
    lines.append("=" * 88)
    lines.append(f"Track 3 Defense Evaluation Report  ({summary['timestamp']})")
    lines.append("=" * 88)
    lines.append(f"source_checkpoint : {summary['source_checkpoint']}")
    lines.append(f"source_seed       : {summary['source_seed']}")
    lines.append(
        f"arch/variant/N    : {summary['architecture']}/{summary['model_variant']}/"
        f"{summary['num_filters']}  trigger_strength={summary['trigger_strength']}"
    )
    lines.append(
        f"splits            : train={summary['splits']['train']}  "
        f"val_inner={summary['splits']['val_inner']}  "
        f"test_external={summary['splits']['test_external']}  "
        f"augmented={summary['splits']['used_augmented_train']}"
    )
    lines.append(
        f"pass_criteria     : degradation_ratio >= {summary['pass_criteria']['degradation_ratio_min']}, "
        f"gap >= {summary['pass_criteria']['degradation_gap_min']}"
    )
    if summary["quick_mode"]:
        lines.append("quick_mode        : TRUE  (epoch counts reduced)")
    lines.append("-" * 88)
    header = (
        f"{'defense':<22} | {'clean_before':>12} {'trig_before':>11} "
        f"{'gap_bef':>8} {'ratio_bef':>9} | {'clean_after':>11} {'trig_after':>10} "
        f"{'gap_aft':>8} {'ratio_aft':>9} | survives"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for res in summary["results"]:
        if "error" in res:
            lines.append(f"{res['defense']:<22} | ERROR: {res['error']}")
            continue
        lines.append(
            f"{res['defense']:<22} | "
            f"{res['clean_mse_before']:>12.6f} {res['triggered_mse_before']:>11.6f} "
            f"{res['degradation_gap_before']:>8.4f} {res['degradation_ratio_before']:>9.4f} | "
            f"{res['clean_mse_after']:>11.6f} {res['triggered_mse_after']:>10.6f} "
            f"{res['degradation_gap_after']:>8.4f} {res['degradation_ratio_after']:>9.4f} | "
            f"{str(res['attack_survives'])}"
        )
    lines.append("=" * 88)
    path.write_text("\n".join(lines) + "\n")
    print(f"[track3] wrote text     → {path}", flush=True)


if __name__ == "__main__":
    main()
