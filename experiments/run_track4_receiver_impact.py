"""
Track 4: Downstream receiver-impact evaluation for backdoored channel estimators.

For each (clean_ckpt, backdoored_ckpt, seed), we evaluate THREE model conditions:
    A. clean_model       + clean_input
    B. backdoor_model    + clean_input
    C. backdoor_model    + triggered_input

Per condition, per split (val, test), per modulation × SNR we record BER/SER/EVM
using the flat-fading `QamReceiverProxy`. The proxy is NOT a full OFDM receiver:
limitations are documented in the output JSON (`proxy_nature_notes`).

Example:
    python run_track4_receiver_impact.py \
        --mat_path /home/nhannv/Hello/2026/backdoor_attack/data.mat \
        --clean_ckpt       results/two_phase_v1/phase1/clean_model_best.pt \
        --backdoor_ckpt    results/multi_seed_20260422_001323/seed_43/phase2_badnets/badnets_model_best.pt \
        --backdoor_summary results/multi_seed_20260422_001323/seed_43/phase2_badnets/phase2_test_summary.json \
        --output_dir       results/track4/seed_43 \
        --snr_list 0 5 10 15 20 25 \
        --modulations qpsk 16qam 64qam
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

# Allow direct execution from the repository root without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from typing import Dict, List, Optional

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
)
from backdoor_ce.data_utils import prepare_mat_data
from backdoor_ce.receiver_eval import sweep_snr_modulation
from run_two_phase import _make_fixed_trigger


DEFAULT_ARCHITECTURE = "non_residual"
DEFAULT_MODEL_VARIANT = "unet"
DEFAULT_NUM_FILTERS = 48
DEFAULT_DROPOUT = 0.15


PROXY_NATURE_NOTES = [
    "Receiver model is a FLAT-FADING per-pixel proxy. NOT a full OFDM receiver.",
    "Channel tensor in this dataset is REAL-valued; complex channel (h + j0) is assumed.",
    "No channel coding / interleaving / pilot-based LS or MMSE equalization.",
    "Equalization is zero-forcing with sign-preserving magnitude clamp (equalizer_eps).",
    "Gray code is used on each I/Q axis. BPSK/QPSK are naturally Gray-coded.",
    "At very high SNR the per-symbol BER plateau is dominated by channel-division noise amplification at near-zero |h|.",
    "Do NOT claim absolute real-world BER; use relative comparisons across conditions.",
]


def _build_eval_config(mat_path: str, seed: int, arch: str, variant: str, num_filters: int) -> ExperimentConfig:
    """Minimal eval config to replicate the Phase-1/Phase-2 data split exactly."""
    cfg = ExperimentConfig()
    cfg.data = DataConfig(data_source="mat", mat_path=mat_path)
    cfg.preprocess = PreprocessConfig(
        normalize_inputs=True,
        normalize_targets=False,
        clip_inputs=True,
    )
    cfg.model = ModelConfig(
        architecture=arch,
        model_variant=variant,
        num_filters=num_filters,
        dropout=DEFAULT_DROPOUT,
        use_batch_norm=True,
        activation="relu",
    )
    cfg.training = TrainingConfig(epochs=1, batch_size=16, num_workers=0, seed=seed)
    return cfg


def _load_checkpoint(ckpt_path: Path, model_cfg: ModelConfig, device: torch.device) -> nn.Module:
    """Rebuild model architecture and load a plain state_dict checkpoint."""
    model = create_model(model_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _collect_estimates(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    trigger: Optional[torch.Tensor] = None,
):
    """Run the model over a loader; return (h_true, h_est) as concatenated tensors."""
    true_list, est_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if trigger is not None:
                xb = xb + trigger.unsqueeze(0).to(device)
            est = model(xb)
            true_list.append(yb)
            est_list.append(est)
    return torch.cat(true_list, dim=0), torch.cat(est_list, dim=0)


def _run_condition(
    *,
    condition: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    trigger: Optional[torch.Tensor],
    snr_list: List[float],
    modulations: List[str],
    split_name: str,
    seed: int,
    equalizer_eps: float,
    symbol_seed: int,
) -> List[Dict]:
    """Compute BER/SER/EVM sweep for one (condition, split) pair."""
    h_true, h_est = _collect_estimates(model, loader, device, trigger=trigger)
    rows = sweep_snr_modulation(
        h_true=h_true,
        h_est=h_est,
        snr_list=snr_list,
        modulation_list=modulations,
        equalizer_eps=equalizer_eps,
        symbol_seed=symbol_seed,
    )
    out: List[Dict] = []
    for row in rows:
        out.append(
            {
                "seed": seed,
                "condition": condition,
                "split": split_name,
                "modulation": row["modulation"],
                "snr_db": row["snr_db"],
                "ber": row["ber"],
                "ser": row["ser"],
                "evm": row["evm"],
                "num_pixels": int(h_true.numel()),
                "num_samples": int(h_true.shape[0]),
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track 4: downstream receiver impact")
    p.add_argument("--mat_path", type=str, required=True)
    p.add_argument("--clean_ckpt", type=str, required=True,
                   help="Phase-1 clean model checkpoint.")
    p.add_argument("--backdoor_ckpt", type=str, required=True,
                   help="Phase-2 backdoored model checkpoint.")
    p.add_argument("--backdoor_summary", type=str, required=True,
                   help="phase2_test_summary.json of the backdoored run (for trigger info).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=None,
                   help="Override seed; default uses the backdoor run's seed.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    p.add_argument("--snr_list", nargs="+", type=float,
                   default=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    p.add_argument("--modulations", nargs="+",
                   default=["qpsk", "16qam", "64qam"],
                   choices=["bpsk", "qpsk", "16qam", "64qam"])
    p.add_argument("--equalizer_eps", type=float, default=1e-3)
    p.add_argument("--symbol_seed", type=int, default=1234)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.backdoor_summary)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    with summary_path.open("r") as f:
        bd_summary = json.load(f)

    trigger_strength = float(bd_summary.get("trigger_strength", 20.0))
    architecture = str(bd_summary.get("architecture", DEFAULT_ARCHITECTURE))
    model_variant = str(bd_summary.get("model_variant", DEFAULT_MODEL_VARIANT))
    num_filters = int(bd_summary.get("num_filters", DEFAULT_NUM_FILTERS))
    seed = int(args.seed) if args.seed is not None else int(bd_summary.get("seed", 42))

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)

    cfg = _build_eval_config(args.mat_path, seed, architecture, model_variant, num_filters)
    prepared = prepare_mat_data(cfg, args.mat_path)
    cfg.model.input_shape = prepared.config.model.input_shape

    loaders = {}
    for split_name in args.splits:
        x, y = prepared.data[split_name]
        loaders[split_name] = DataLoader(
            TensorDataset(x, y),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
    print(
        f"[track4] seed={seed}  arch={architecture}/{model_variant}/{num_filters}  "
        f"trigger_strength={trigger_strength}  device={device}",
        flush=True,
    )
    for s_name, loader in loaders.items():
        print(f"[track4] split '{s_name}': {len(loader.dataset)} samples", flush=True)

    clean_model = _load_checkpoint(Path(args.clean_ckpt), cfg.model, device)
    backdoor_model = _load_checkpoint(Path(args.backdoor_ckpt), cfg.model, device)
    trigger = _make_fixed_trigger(cfg.model.input_shape, trigger_strength).to(device)

    all_rows: List[Dict] = []

    conditions = [
        ("clean_model__clean_input",      clean_model,    None),
        ("backdoor_model__clean_input",   backdoor_model, None),
        ("backdoor_model__triggered_input", backdoor_model, trigger),
    ]

    for split_name, loader in loaders.items():
        for condition, model, trig in conditions:
            t0 = time.time()
            print(f"  [run] split={split_name}  condition={condition}", flush=True)
            rows = _run_condition(
                condition=condition,
                model=model,
                loader=loader,
                device=device,
                trigger=trig,
                snr_list=list(args.snr_list),
                modulations=list(args.modulations),
                split_name=split_name,
                seed=seed,
                equalizer_eps=args.equalizer_eps,
                symbol_seed=args.symbol_seed,
            )
            all_rows.extend(rows)
            elapsed = time.time() - t0
            print(f"    done ({elapsed:.1f}s, {len(rows)} sweep points)", flush=True)

    summary_out = {
        "track": 4,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "architecture": architecture,
        "model_variant": model_variant,
        "num_filters": num_filters,
        "trigger_strength": trigger_strength,
        "clean_ckpt": str(args.clean_ckpt),
        "backdoor_ckpt": str(args.backdoor_ckpt),
        "backdoor_summary": str(args.backdoor_summary),
        "splits": {s: int(len(loaders[s].dataset)) for s in args.splits},
        "snr_list": list(args.snr_list),
        "modulations": list(args.modulations),
        "equalizer_eps": args.equalizer_eps,
        "symbol_seed": args.symbol_seed,
        "proxy_nature_notes": PROXY_NATURE_NOTES,
        "records": all_rows,
    }

    out_json = output_dir / "track4_receiver_metrics.json"
    with out_json.open("w") as f:
        json.dump(summary_out, f, indent=2, default=str)
    print(f"[track4] wrote JSON → {out_json}", flush=True)

    _write_flat_csv(output_dir / "track4_receiver_metrics.csv", all_rows)
    _write_pivot_table(output_dir / "track4_receiver_metrics.txt", summary_out)


def _write_flat_csv(path: Path, rows: List[Dict]) -> None:
    import csv
    if not rows:
        return
    cols = [
        "seed", "condition", "split", "modulation", "snr_db",
        "ber", "ser", "evm", "num_pixels", "num_samples",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[track4] wrote CSV  → {path}", flush=True)


def _write_pivot_table(path: Path, summary: Dict) -> None:
    lines = []
    lines.append("=" * 92)
    lines.append(f"Track 4 Receiver Impact Report  ({summary['timestamp']})")
    lines.append("=" * 92)
    lines.append(f"seed              : {summary['seed']}")
    lines.append(f"arch/variant/N    : {summary['architecture']}/{summary['model_variant']}/{summary['num_filters']}")
    lines.append(f"trigger_strength  : {summary['trigger_strength']}")
    lines.append(f"splits            : {summary['splits']}")
    lines.append(f"snr_list          : {summary['snr_list']}")
    lines.append(f"modulations       : {summary['modulations']}")
    lines.append("NOTE: proxy flat-fading receiver; see proxy_nature_notes in JSON.")
    lines.append("-" * 92)
    for split in summary["splits"]:
        lines.append(f"[split = {split}]")
        lines.append(
            f"  {'condition':<36} {'modulation':<8} {'snr_dB':>6} "
            f"{'ber':>10} {'ser':>10} {'evm':>10}"
        )
        for rec in summary["records"]:
            if rec["split"] != split:
                continue
            lines.append(
                f"  {rec['condition']:<36} {rec['modulation']:<8} {rec['snr_db']:>6.1f} "
                f"{rec['ber']:>10.6f} {rec['ser']:>10.6f} {rec['evm']:>10.4f}"
            )
        lines.append("")
    lines.append("=" * 92)
    path.write_text("\n".join(lines) + "\n")
    print(f"[track4] wrote text → {path}", flush=True)


if __name__ == "__main__":
    main()
