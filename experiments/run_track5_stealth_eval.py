"""
Track 5: Stealthiness evaluation of backdoor triggers.

For each combination (trigger_type, trigger_strength, seed), the runner:
    1. Builds a trigger tensor (using either the legacy `trigger_patterns`
       framework or the `uniform_positive_L2norm` BadNets trigger from
       `run_two_phase._make_fixed_trigger`).
    2. Computes trigger-energy and signal-energy statistics on the validation
       or test split, plus trigger-to-signal ratio in dB.
    3. Runs three simple detectors (input-norm, pixel-mean, matched-filter)
       on (clean, triggered) samples and records balanced accuracy / AUC.
    4. OPTIONALLY loads a backdoored checkpoint and measures
       `degradation_ratio` under the same trigger on the chosen split.

Outputs:
    track5_stealth_summary.json
    track5_stealth_summary.csv
    track5_stealth_summary.txt

Example:
    python run_track5_stealth_eval.py \
        --mat_path /home/nhannv/Hello/2026/backdoor_attack/data.mat \
        --output_dir results/track5_smoke \
        --trigger_strengths 1 5 10 20 40 \
        --trigger_types fixed_badnets FIXED LOW_INTENSITY SCATTERED \
        --seed_list 42 \
        --split test \
        --backdoor_ckpt    results/multi_seed_20260422_001323/seed_43/phase2_badnets/badnets_model_best.pt \
        --backdoor_summary results/multi_seed_20260422_001323/seed_43/phase2_badnets/phase2_test_summary.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import sys
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

from backdoor_ce.channel_estimator import create_model
from backdoor_ce.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PreprocessConfig,
    TrainingConfig,
    TriggerConfig,
    TriggerType,
)
from backdoor_ce.data_utils import prepare_mat_data
from run_two_phase import _eval_mse, _eval_triggered_mse, _make_fixed_trigger
from backdoor_ce.trigger_detector import (
    compute_signal_energy_stats,
    compute_trigger_energy_stats,
    evaluate_detectors,
)
from backdoor_ce.trigger_patterns import create_trigger, trigger_signature


BADNETS_TRIGGER_TAG = "fixed_badnets"

FRAMEWORK_TRIGGER_MAP = {
    "FIXED": TriggerType.FIXED,
    "PARTIAL": TriggerType.PARTIAL,
    "SCATTERED": TriggerType.SCATTERED,
    "LOW_INTENSITY": TriggerType.LOW_INTENSITY,
    "POSITION_DEPENDENT": TriggerType.POSITION_DEPENDENT,
}

DEFAULT_ARCHITECTURE = "residual"
DEFAULT_MODEL_VARIANT = "unet"
DEFAULT_NUM_FILTERS = 48


def _infer_architecture_from_path(path: str | None) -> str | None:
    """Infer architecture from a path component when summary metadata is absent."""
    if not path:
        return None
    parts = [part.lower() for part in Path(path).parts]
    if "residual" in parts:
        return "residual"
    if "non_residual" in parts:
        return "non_residual"
    return None


def _validate_architecture_metadata(
    *,
    requested_architecture: str,
    backdoor_ckpt: str | None,
    backdoor_summary: str | None,
    summary_architecture: str | None,
) -> None:
    """Fail fast when a checkpoint/summary clearly belongs to the other architecture."""
    candidates = {
        "--backdoor_ckpt path": _infer_architecture_from_path(backdoor_ckpt),
        "--backdoor_summary path": _infer_architecture_from_path(backdoor_summary),
        "backdoor_summary[architecture]": summary_architecture,
    }
    for source, arch in candidates.items():
        if arch is not None and arch != requested_architecture:
            raise ValueError(
                f"Architecture mismatch: requested {requested_architecture!r}, but {source} "
                f"indicates {arch!r}. Pass --architecture {arch} or use matching residual/non_residual files."
            )


def _build_eval_config(
    mat_path: str,
    seed: int,
    architecture: str = DEFAULT_ARCHITECTURE,
    model_variant: str = DEFAULT_MODEL_VARIANT,
    num_filters: int = DEFAULT_NUM_FILTERS,
) -> ExperimentConfig:
    """Minimal config to re-run data preprocessing identical to Phase 1/2."""
    cfg = ExperimentConfig()
    cfg.data = DataConfig(data_source="mat", mat_path=mat_path)
    cfg.preprocess = PreprocessConfig(
        normalize_inputs=True,
        normalize_targets=False,
        clip_inputs=True,
    )
    cfg.model = ModelConfig(
        architecture=architecture,
        model_variant=model_variant,
        num_filters=num_filters,
        dropout=0.15,
        use_batch_norm=True,
        activation="relu",
    )
    cfg.training = TrainingConfig(epochs=1, batch_size=16, num_workers=0, seed=seed)
    return cfg


def _build_trigger_tensor(
    trigger_type: str,
    trigger_strength: float,
    input_shape: Tuple[int, int, int],
    trigger_seed: int = 1234,
) -> torch.Tensor:
    """Build a trigger tensor of shape (C,H,W) for either BadNets or framework."""
    if trigger_type == BADNETS_TRIGGER_TAG:
        return _make_fixed_trigger(input_shape, float(trigger_strength))

    framework_key = trigger_type.upper()
    if framework_key not in FRAMEWORK_TRIGGER_MAP:
        raise ValueError(
            f"Unknown trigger_type '{trigger_type}'. "
            f"Expected '{BADNETS_TRIGGER_TAG}' or one of {list(FRAMEWORK_TRIGGER_MAP.keys())}."
        )
    torch.manual_seed(trigger_seed)
    np.random.seed(trigger_seed)
    cfg = TriggerConfig(
        trigger_type=FRAMEWORK_TRIGGER_MAP[framework_key],
        trigger_strength=float(trigger_strength),
        normalize_pattern_energy=True,
    )
    trigger = create_trigger(cfg)
    pattern = trigger.get_trigger_pattern(input_shape).detach()
    return pattern


def _load_backdoor_model(
    ckpt_path: Path,
    cfg: ExperimentConfig,
    device: torch.device,
) -> nn.Module:
    """Rebuild architecture and load plain state_dict of a backdoored checkpoint."""
    model = create_model(cfg.model).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _run_single_combo(
    *,
    trigger_type: str,
    trigger_strength: float,
    seed: int,
    architecture: str,
    input_shape: Tuple[int, int, int],
    clean_inputs: torch.Tensor,
    clean_loader: Optional[DataLoader],
    triggered_loader: Optional[DataLoader],
    device: torch.device,
    backdoor_model: Optional[nn.Module],
    poison_rate: Optional[float],
) -> Dict:
    """Compute one row (stealth + optional effectiveness) for one combo."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    trigger = _build_trigger_tensor(trigger_type, trigger_strength, input_shape, seed)
    trigger_stats = compute_trigger_energy_stats(trigger)
    signal_stats = compute_signal_energy_stats(clean_inputs)

    clean_energy = signal_stats["signal_l2_norm_mean"]
    trigger_energy = trigger_stats["trigger_l2_norm"]
    tsr = trigger_energy / max(clean_energy, 1e-12)
    tsr_db = 20.0 * np.log10(max(tsr, 1e-12))

    triggered_inputs = clean_inputs + trigger.unsqueeze(0).to(clean_inputs.device)
    detector_results = evaluate_detectors(clean_inputs, triggered_inputs, trigger.to(clean_inputs.device))

    degradation_block: Dict = {}
    if backdoor_model is not None and clean_loader is not None:
        clean_mse = _eval_mse(backdoor_model, clean_loader, device)
        trig_mse = _eval_triggered_mse(backdoor_model, clean_loader, trigger.to(device), device)
        degradation_block = {
            "clean_mse":       float(clean_mse),
            "triggered_mse":   float(trig_mse),
            "degradation_gap": float(trig_mse - clean_mse),
            "degradation_ratio": float(trig_mse / max(clean_mse, 1e-12)),
        }

    detector_rows = []
    for dr in detector_results:
        detector_rows.append(
            {
                "detector": dr.name,
                "is_trigger_aware": dr.is_trigger_aware,
                "detectability_auc": dr.auc,
                "detectability_accuracy": dr.accuracy,
                "best_threshold": dr.threshold,
                "clean_scores_mean": dr.clean_scores_mean,
                "clean_scores_std":  dr.clean_scores_std,
                "trig_scores_mean":  dr.trig_scores_mean,
                "trig_scores_std":   dr.trig_scores_std,
                "num_clean":         dr.num_clean,
                "num_triggered":     dr.num_triggered,
            }
        )

    row: Dict = {
        "architecture":     architecture,
        "trigger_type":     trigger_type,
        "trigger_strength": float(trigger_strength),
        "seed":             int(seed),
        "poison_rate":      float(poison_rate) if poison_rate is not None else None,
        "trigger_energy":   float(trigger_energy),
        "trigger_linf":     float(trigger_stats["trigger_linf_norm"]),
        "trigger_active_fraction": float(trigger_stats["trigger_active_fraction"]),
        "trigger_mean_abs": float(trigger_stats["trigger_mean_abs"]),
        "signal_energy":    float(clean_energy),
        "signal_energy_std": float(signal_stats["signal_l2_norm_std"]),
        "trigger_to_signal_ratio":    float(tsr),
        "trigger_to_signal_ratio_db": float(tsr_db),
        "detectors":        detector_rows,
        "degradation":      degradation_block,
    }
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track 5: stealthiness vs effectiveness trade-off")
    p.add_argument("--mat_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--trigger_strengths", nargs="+", type=float,
                   default=[1.0, 5.0, 10.0, 20.0, 40.0])
    p.add_argument("--trigger_types", nargs="+",
                   default=[BADNETS_TRIGGER_TAG, "FIXED", "LOW_INTENSITY", "SCATTERED"])
    p.add_argument("--seed_list", nargs="+", type=int, default=[42])
    p.add_argument("--split", choices=["val", "test"], default="test",
                   help="Split used to compute detector AUC and degradation. "
                        "'test' is the external held-out; 'val' is inner-val (larger N).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--architecture", choices=["residual", "non_residual"],
                   default=DEFAULT_ARCHITECTURE,
                   help="Architecture used to rebuild the checkpoint. Defaults to residual.")
    p.add_argument("--model_variant", type=str, default=DEFAULT_MODEL_VARIANT,
                   help="Backbone variant; default: unet.")
    p.add_argument("--num_filters", type=int, default=DEFAULT_NUM_FILTERS,
                   help="Base filter count; default: 48.")

    p.add_argument("--backdoor_ckpt", type=str, default=None,
                   help="If provided, also compute degradation_ratio with this ckpt.")
    p.add_argument("--backdoor_summary", type=str, default=None,
                   help="phase2_test_summary.json; used for poison_rate metadata.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    poison_rate: Optional[float] = None
    summary_architecture: Optional[str] = None
    if args.backdoor_summary is not None:
        bd_summary_path = Path(args.backdoor_summary)
        if bd_summary_path.exists():
            with bd_summary_path.open("r") as f:
                bd_summary = json.load(f)
            poison_rate = float(bd_summary.get("poison_rate")) if bd_summary.get("poison_rate") is not None else None
            summary_architecture = bd_summary.get("architecture") or bd_summary.get("model_architecture")

    _validate_architecture_metadata(
        requested_architecture=args.architecture,
        backdoor_ckpt=args.backdoor_ckpt,
        backdoor_summary=args.backdoor_summary,
        summary_architecture=summary_architecture,
    )

    cfg = _build_eval_config(
        args.mat_path,
        args.seed_list[0],
        architecture=args.architecture,
        model_variant=args.model_variant,
        num_filters=args.num_filters,
    )
    prepared = prepare_mat_data(cfg, args.mat_path)
    cfg.model.input_shape = prepared.config.model.input_shape
    input_shape = cfg.model.input_shape

    split_name = args.split
    x_split, y_split = prepared.data[split_name]
    clean_inputs = x_split.to(device)
    loader = DataLoader(
        TensorDataset(x_split, y_split),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    print(
        f"[track5] architecture={args.architecture}  split='{split_name}' (N={len(x_split)})  input_shape={input_shape}  device={device}",
        flush=True,
    )

    backdoor_model: Optional[nn.Module] = None
    if args.backdoor_ckpt is not None:
        backdoor_model = _load_backdoor_model(Path(args.backdoor_ckpt), cfg, device)
        print(f"[track5] loaded backdoor ckpt: {args.backdoor_ckpt}", flush=True)

    all_rows: List[Dict] = []
    t0 = time.time()
    for trigger_type in args.trigger_types:
        for trigger_strength in args.trigger_strengths:
            for seed in args.seed_list:
                row = _run_single_combo(
                    trigger_type=trigger_type,
                    trigger_strength=float(trigger_strength),
                    seed=int(seed),
                    architecture=args.architecture,
                    input_shape=input_shape,
                    clean_inputs=clean_inputs,
                    clean_loader=loader if backdoor_model is not None else None,
                    triggered_loader=None,
                    device=device,
                    backdoor_model=backdoor_model,
                    poison_rate=poison_rate,
                )
                det_line = "  ".join(
                    f"{d['detector']}={d['detectability_auc']:.3f}"
                    for d in row["detectors"]
                )
                deg_line = ""
                if row["degradation"]:
                    deg_line = (
                        f"  deg_ratio={row['degradation']['degradation_ratio']:.3f}"
                    )
                print(
                    f"  [{trigger_type:<16} s={trigger_strength:<5g} seed={seed}] "
                    f"TSR_dB={row['trigger_to_signal_ratio_db']:+.2f}  {det_line}{deg_line}",
                    flush=True,
                )
                all_rows.append(row)

    elapsed = time.time() - t0
    summary_out = {
        "track": 5,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mat_path":       args.mat_path,
        "architecture":   args.architecture,
        "model_variant":  args.model_variant,
        "num_filters":    int(args.num_filters),
        "split":          split_name,
        "num_samples":    int(len(x_split)),
        "input_shape":    list(input_shape),
        "trigger_types":  list(args.trigger_types),
        "trigger_strengths": list(args.trigger_strengths),
        "seed_list":      list(args.seed_list),
        "backdoor_ckpt":  str(args.backdoor_ckpt) if args.backdoor_ckpt else None,
        "backdoor_summary": str(args.backdoor_summary) if args.backdoor_summary else None,
        "poison_rate":    poison_rate,
        "elapsed_seconds": float(elapsed),
        "detector_notes": [
            "'input_norm' and 'pixel_mean' are non-trigger-aware (realistic defender).",
            "'matched_filter' is TRIGGER-AWARE and provides an UPPER BOUND on detectability.",
            "Balanced accuracy reported with threshold chosen on the same split (optimistic).",
            "With uniform-positive triggers, matched_filter and pixel_mean coincide by construction.",
        ],
        "records":        all_rows,
    }

    out_json = output_dir / "track5_stealth_summary.json"
    with out_json.open("w") as f:
        json.dump(summary_out, f, indent=2, default=str)
    print(f"[track5] wrote JSON → {out_json}", flush=True)

    _write_flat_csv(output_dir / "track5_stealth_summary.csv", all_rows)
    _write_pivot_table(output_dir / "track5_stealth_summary.txt", summary_out)


def _write_flat_csv(path: Path, rows: List[Dict]) -> None:
    """One row per (trigger_type, strength, seed, detector) — flattened for plotting."""
    import csv
    if not rows:
        return
    cols = [
        "architecture", "trigger_type", "trigger_strength", "seed", "poison_rate",
        "trigger_energy", "trigger_linf", "trigger_active_fraction", "trigger_mean_abs",
        "signal_energy",
        "trigger_to_signal_ratio", "trigger_to_signal_ratio_db",
        "detector", "is_trigger_aware",
        "detectability_auc", "detectability_accuracy",
        "clean_mse", "triggered_mse", "degradation_gap", "degradation_ratio",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            deg = row.get("degradation") or {}
            base = {
                **{k: row[k] for k in row if k not in ("detectors", "degradation")},
                "clean_mse":         deg.get("clean_mse"),
                "triggered_mse":     deg.get("triggered_mse"),
                "degradation_gap":   deg.get("degradation_gap"),
                "degradation_ratio": deg.get("degradation_ratio"),
            }
            for det in row["detectors"]:
                out_row = {
                    **base,
                    "detector":                 det["detector"],
                    "is_trigger_aware":         det["is_trigger_aware"],
                    "detectability_auc":        det["detectability_auc"],
                    "detectability_accuracy":   det["detectability_accuracy"],
                }
                writer.writerow(out_row)
    print(f"[track5] wrote CSV  → {path}", flush=True)


def _write_pivot_table(path: Path, summary: Dict) -> None:
    lines = []
    lines.append("=" * 110)
    lines.append(f"Track 5 Stealthiness Report  ({summary['timestamp']})")
    lines.append("=" * 110)
    lines.append(f"architecture     : {summary.get('architecture')}")
    lines.append(f"split            : {summary['split']}  (N={summary['num_samples']})")
    lines.append(f"input_shape      : {summary['input_shape']}")
    lines.append(f"trigger_types    : {summary['trigger_types']}")
    lines.append(f"trigger_strengths: {summary['trigger_strengths']}")
    lines.append(f"seed_list        : {summary['seed_list']}")
    lines.append(f"backdoor_ckpt    : {summary['backdoor_ckpt']}")
    lines.append(f"poison_rate      : {summary['poison_rate']}")
    lines.append("NOTE: matched_filter AUC is an UPPER BOUND (trigger-aware).")
    lines.append("-" * 110)
    header = (
        f"{'trigger_type':<16} {'strength':>8} {'seed':>4} "
        f"{'TSR_dB':>7} {'E_trig':>8} {'E_sig':>8} "
        f"{'AUC_inNorm':>10} {'AUC_pxMn':>9} {'AUC_mchF':>9} "
        f"{'deg_ratio':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in summary["records"]:
        aucs = {d["detector"]: d["detectability_auc"] for d in r["detectors"]}
        deg_ratio = (r["degradation"] or {}).get("degradation_ratio")
        deg_str = f"{deg_ratio:>9.4f}" if deg_ratio is not None else f"{'-':>9}"
        lines.append(
            f"{r['trigger_type']:<16} {r['trigger_strength']:>8.3f} {r['seed']:>4} "
            f"{r['trigger_to_signal_ratio_db']:>+7.2f} "
            f"{r['trigger_energy']:>8.2f} {r['signal_energy']:>8.2f} "
            f"{aucs.get('input_norm', float('nan')):>10.4f} "
            f"{aucs.get('pixel_mean', float('nan')):>9.4f} "
            f"{aucs.get('matched_filter', float('nan')):>9.4f} "
            f"{deg_str}"
        )
    lines.append("=" * 110)
    path.write_text("\n".join(lines) + "\n")
    print(f"[track5] wrote text → {path}", flush=True)


if __name__ == "__main__":
    main()
