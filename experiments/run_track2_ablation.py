"""
Track 2 — Trigger Design Space Ablation.

Evaluates backdoor attack sensitivity along three axes:
  1. trigger_strength  sweep: effect of perturbation magnitude
  2. poison_rate       sweep: effect of poisoned fraction
  3. trigger_type      sweep: effect of trigger pattern shape

For each axis, the other two parameters are held at anchor values:
  anchor_strength = 20.0
  anchor_poison_rate = 0.10
  anchor_trigger_type = "uniform_positive"

Runs both non_residual and residual architectures to check whether the
residual-vs-non_residual pattern from Track 1 persists across parameter regions.

Design:
  - Data loaded ONCE per architecture (reuses augmented dataset across cells).
  - Reuses Phase 1 clean checkpoints by default (--reuse_clean_checkpoints yes).
  - Deduplicates anchor cell (shared between all 3 sweeps) automatically.
  - Saves a standard phase2_test_summary.json per cell (same schema as Track 1).

Usage:
    # If you pipe to tee(1), create the output directory first — otherwise tee
    # opens the log file before Python creates the directory and fails.
    mkdir -p ./results/track2_sweeps
    python run_track2_ablation.py \\
        --mat_path /path/to/data.mat \\
        --output_dir ./results/track2_sweeps \\
        --seed 42 \\
        --phase2_epochs 20 \\
        --batch_size 64 \\
        --reuse_clean_checkpoints yes \\
        --clean_checkpoint_root ./results/track1_single_seed_e100

    # Run only the type sweep:
    python run_track2_ablation.py ... --sweeps type

    # Full sweep with explicit ranges:
    python run_track2_ablation.py ... \\
        --strength_values 10 20 30 \\
        --poison_values 0.05 0.10 0.20 \\
        --type_values uniform_positive checkerboard partial_patch
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Default sweep scope
# ---------------------------------------------------------------------------

ANCHOR_STRENGTH    = 20.0
ANCHOR_POISON_RATE = 0.10
ANCHOR_TRIGGER_TYPE = "uniform_positive"

DEFAULT_STRENGTH_VALUES  = [10.0, 20.0, 30.0]
DEFAULT_POISON_VALUES    = [0.05, 0.10, 0.20]
DEFAULT_TRIGGER_TYPES    = ["uniform_positive", "checkerboard", "partial_patch"]
DEFAULT_ARCHITECTURES    = ["non_residual", "residual"]

WRONG_TARGET_BIAS  = 1.0
WRONG_TARGET_FORM  = "global_additive"

PASS3_CLEAN_BUDGET = 0.60
PASS3_GAP_MIN      = 0.05
PASS3_RATIO_MIN    = 1.25


# ---------------------------------------------------------------------------
# Cell descriptor
# ---------------------------------------------------------------------------

class Cell(NamedTuple):
    """One (architecture, seed, trigger_type, strength, poison_rate) combination."""
    architecture:    str
    seed:            int
    trigger_type:    str
    trigger_strength: float
    poison_rate:     float


def _cell_dir_name(cell: Cell) -> str:
    """Filesystem-safe subdirectory name for one cell."""
    return f"str{cell.trigger_strength:.0f}_pr{cell.poison_rate:.2f}_{cell.trigger_type}"


def _cell_output_dir(root: Path, cell: Cell) -> Path:
    return root / cell.architecture / f"seed_{cell.seed}" / _cell_dir_name(cell)


def _summary_path(cell_out: Path) -> Path:
    return cell_out / "phase2_badnets" / "phase2_test_summary.json"


# ---------------------------------------------------------------------------
# Sweep builder
# ---------------------------------------------------------------------------

def _build_sweep_cells(
    architectures: List[str],
    seed: int,
    strength_values: List[float],
    poison_values: List[float],
    trigger_types: List[str],
    sweeps: List[str],
) -> List[Cell]:
    """Return the deduplicated list of cells for the requested sweeps.

    anchor cell = (ANCHOR_STRENGTH, ANCHOR_POISON_RATE, ANCHOR_TRIGGER_TYPE)
    is included once even if it appears in multiple sweeps.
    """
    seen: Set[Tuple] = set()
    cells: List[Cell] = []

    def _add(arch: str, s: float, p: float, t: str) -> None:
        key = (arch, seed, t, s, p)
        if key not in seen:
            seen.add(key)
            cells.append(Cell(architecture=arch, seed=seed,
                               trigger_type=t, trigger_strength=s, poison_rate=p))

    for arch in architectures:
        if "strength" in sweeps:
            for s in strength_values:
                _add(arch, s, ANCHOR_POISON_RATE, ANCHOR_TRIGGER_TYPE)
        if "poison" in sweeps:
            for p in poison_values:
                _add(arch, ANCHOR_STRENGTH, p, ANCHOR_TRIGGER_TYPE)
        if "type" in sweeps:
            for t in trigger_types:
                _add(arch, ANCHOR_STRENGTH, ANCHOR_POISON_RATE, t)

    return cells


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data_and_cfg(mat_path: Path, arch: str, seed: int):
    """Load, split, and augment data for one architecture.

    Returns (cfg, data) ready to pass to run_phase2_badnets.
    Mirrors the badnets-path data preparation in run_two_phase.py main().
    """
    from run_two_phase import build_phase1_config, prepare_mat_data, augment_channel_data

    cfg = build_phase1_config(str(mat_path))
    cfg.model.architecture = arch
    cfg.training.seed = int(seed)
    prepared = prepare_mat_data(cfg, str(mat_path))
    data = prepared.data
    cfg  = prepared.config
    ox, oy = data["train"]
    ax, ay = augment_channel_data(ox, oy)
    data["train"] = (ax, ay)
    return cfg, data


# ---------------------------------------------------------------------------
# Clean-checkpoint resolution
# ---------------------------------------------------------------------------

def _find_clean_ckpt(
    arch: str,
    seed: int,
    output_dir: Path,
    reuse_root: Optional[Path],
) -> Optional[Path]:
    """Return Phase 1 checkpoint path if it exists (reuse root or local)."""
    if reuse_root is not None:
        candidate = reuse_root / arch / f"seed_{seed}" / "phase1" / "clean_model_best.pt"
        if candidate.exists():
            return candidate
    local = output_dir / "phase1" / arch / f"seed_{seed}" / "phase1" / "clean_model_best.pt"
    if local.exists():
        return local
    return None


def _train_phase1(mat_path: Path, arch: str, seed: int, output_dir: Path) -> Optional[Path]:
    """Train Phase 1 in-process. Returns checkpoint path on success."""
    from run_two_phase import run_phase1

    phase1_out = output_dir / "phase1" / arch / f"seed_{seed}"
    phase1_out.mkdir(parents=True, exist_ok=True)
    try:
        _cfg, _data, ckpt, test_mse = run_phase1(str(mat_path), str(phase1_out), seed, arch)
        print(f"  Phase 1 done: arch={arch} seed={seed} test_mse={test_mse:.4f}", flush=True)
        return ckpt
    except Exception as exc:
        print(f"  Phase 1 FAILED for arch={arch} seed={seed}: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Cell execution
# ---------------------------------------------------------------------------

def _run_cell_in_process(
    mat_path: Path,
    clean_ckpt: Path,
    cell: Cell,
    output_dir: Path,
    phase2_epochs: int,
    early_stop_patience: int,
    batch_size: int,
    data: Dict,
    cfg,
) -> Optional[Dict]:
    """Run one cell in-process, tee'ing stdout to a per-cell log file.

    Returns the parsed phase2_test_summary dict on success, None on failure.
    """
    from run_two_phase import run_phase2_badnets

    cell_out = _cell_output_dir(output_dir, cell)
    cell_out.mkdir(parents=True, exist_ok=True)
    log_path = cell_out / "phase2_run.log"
    print(f"  LOG → {log_path}", flush=True)

    data_copy: Dict = {k: copy.deepcopy(v) for k, v in data.items()}

    try:
        with log_path.open("w") as log_fh:
            class _Tee:
                def __init__(self, orig, fobj):
                    self._orig = orig
                    self._fobj = fobj
                def write(self, s):
                    self._orig.write(s)
                    self._fobj.write(s)
                    # Flush eagerly so abrupt termination still leaves useful logs.
                    self._orig.flush()
                    self._fobj.flush()
                def flush(self):
                    self._orig.flush()
                    self._fobj.flush()
                def fileno(self):
                    return self._orig.fileno()

            orig_stdout = sys.stdout
            sys.stdout = _Tee(orig_stdout, log_fh)
            try:
                run_phase2_badnets(
                    mat_path=str(mat_path),
                    clean_checkpoint=clean_ckpt,
                    clean_config=cfg,
                    data=data_copy,
                    output_dir=str(cell_out),
                    trigger_strength=cell.trigger_strength,
                    poison_rate=cell.poison_rate,
                    seed=cell.seed,
                    epochs=phase2_epochs,
                    early_stop_patience=early_stop_patience,
                    batch_size=batch_size,
                    trigger_type=cell.trigger_type,
                    wrong_target_form=WRONG_TARGET_FORM,
                    wrong_target_bias=WRONG_TARGET_BIAS,
                )
            finally:
                sys.stdout = orig_stdout
    except Exception as exc:
        print(f"  EXCEPTION in cell {cell}: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        return None

    return _load_json(_summary_path(cell_out))


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _build_cell_record(cell: Cell, p2: Optional[Dict]) -> Dict:
    """Flatten one cell's Phase 2 summary into a flat record dict."""
    rec: Dict = {
        "architecture":    cell.architecture,
        "seed":            cell.seed,
        "trigger_type":    cell.trigger_type,
        "trigger_strength": cell.trigger_strength,
        "poison_rate":     cell.poison_rate,
    }
    if p2 is None:
        rec["status"] = "missing"
        return rec
    rec["status"]        = "done"
    rec["best_epoch"]    = p2.get("best_epoch")
    rec["overall_pass3"] = p2.get("overall_pass3")
    for split in ("val", "test"):
        sd = p2.get(split, {})
        for metric in ("clean_mse", "triggered_mse", "degradation_gap", "degradation_ratio"):
            rec[f"{split}_{metric}"] = sd.get(metric)
    return rec


def _write_csv(records: List[Dict], path: Path) -> None:
    if not records:
        return
    fields = list(records[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"  CSV → {path}", flush=True)


def _write_json(records: List[Dict], path: Path) -> None:
    path.write_text(json.dumps(records, indent=2))
    print(f"  JSON → {path}", flush=True)


def _sweep_records(all_records: List[Dict], sweep: str,
                   strength_values: List[float],
                   poison_values: List[float],
                   trigger_types: List[str]) -> List[Dict]:
    """Filter records to those belonging to one sweep axis."""
    if sweep == "strength":
        return [r for r in all_records
                if r["trigger_type"] == ANCHOR_TRIGGER_TYPE
                and round(r["poison_rate"], 4) == round(ANCHOR_POISON_RATE, 4)]
    if sweep == "poison":
        return [r for r in all_records
                if r["trigger_type"] == ANCHOR_TRIGGER_TYPE
                and round(r["trigger_strength"], 2) == round(ANCHOR_STRENGTH, 2)]
    if sweep == "type":
        return [r for r in all_records
                if round(r["trigger_strength"], 2) == round(ANCHOR_STRENGTH, 2)
                and round(r["poison_rate"], 4) == round(ANCHOR_POISON_RATE, 4)]
    return all_records


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _elapsed_str(start: float) -> str:
    s = int(time.time() - start)
    return f"{s // 3600:02d}h {(s % 3600) // 60:02d}m {s % 60:02d}s"


def _print_banner(msg: str) -> None:
    print(f"\n{'=' * 72}\n  {msg}\n{'=' * 72}", flush=True)


def _print_sweep_table(records: List[Dict], sweep: str) -> None:
    """Print a compact results table for one sweep axis."""
    print(f"\n── {sweep.upper()} SWEEP ─────────────────────────────────────────")
    hdr = (f"  {'Architecture':<16} {'Strength':>9} {'Poison':>7} {'Type':<20} "
           f"{'clean':>10} {'trig':>10} {'ratio':>7}  PASS3")
    print(hdr)
    print("  " + "─" * 79)
    for rec in sorted(records, key=lambda r: (
        r.get("trigger_strength", 0), r.get("poison_rate", 0),
        r.get("trigger_type", ""), r.get("architecture", "")
    )):
        if rec.get("status") != "done":
            print(f"  {rec.get('architecture', '?'):<16} — FAILED/MISSING")
            continue
        clean = rec.get("test_clean_mse") or float("nan")
        trig  = rec.get("test_triggered_mse") or float("nan")
        ratio = rec.get("test_degradation_ratio") or float("nan")
        p3    = "YES" if rec.get("overall_pass3") else "NO "
        print(f"  {rec['architecture']:<16} {rec['trigger_strength']:>9.1f} "
              f"{rec['poison_rate']:>7.2f} {rec['trigger_type']:<20} "
              f"{clean:>10.6f} {trig:>10.6f} {ratio:>7.3f}  {p3}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track 2: Trigger design-space ablation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mat_path",    required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument(
        "--architectures", nargs="+", default=DEFAULT_ARCHITECTURES,
        choices=["non_residual", "residual"],
    )
    parser.add_argument("--phase2_epochs", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument(
        "--sweeps", nargs="+", default=["strength", "poison", "type"],
        choices=["strength", "poison", "type"],
        help="Which sweep axes to run",
    )
    parser.add_argument(
        "--strength_values", nargs="+", type=float, default=DEFAULT_STRENGTH_VALUES,
    )
    parser.add_argument(
        "--poison_values",   nargs="+", type=float, default=DEFAULT_POISON_VALUES,
    )
    parser.add_argument(
        "--type_values",     nargs="+", default=DEFAULT_TRIGGER_TYPES,
        choices=["uniform_positive", "checkerboard", "partial_patch",
                 "scattered", "low_intensity"],
    )
    parser.add_argument(
        "--reuse_clean_checkpoints", default="yes", choices=["yes", "no"],
    )
    parser.add_argument("--clean_checkpoint_root", default=None)
    return parser.parse_args()


def main() -> None:  # noqa: C901
    args = _parse_args()
    mat_path   = Path(args.mat_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reuse_root: Optional[Path] = None
    if args.reuse_clean_checkpoints == "yes" and args.clean_checkpoint_root:
        reuse_root = Path(args.clean_checkpoint_root)

    cells = _build_sweep_cells(
        architectures=args.architectures,
        seed=args.seed,
        strength_values=args.strength_values,
        poison_values=args.poison_values,
        trigger_types=args.type_values,
        sweeps=args.sweeps,
    )

    _print_banner(
        f"Track 2 — Trigger Design Space Ablation\n"
        f"  seed={args.seed}  archs={args.architectures}\n"
        f"  sweeps={args.sweeps}  cells={len(cells)}\n"
        f"  phase2_epochs={args.phase2_epochs}  "
        f"early_stop_patience={args.early_stop_patience}  "
        f"batch_size={args.batch_size}\n"
        f"  output={output_dir}"
    )
    if reuse_root:
        print(f"  Phase 1 REUSE root: {reuse_root}", flush=True)
    print(f"  Anchor: strength={ANCHOR_STRENGTH}, poison={ANCHOR_POISON_RATE}, "
          f"type={ANCHOR_TRIGGER_TYPE}", flush=True)

    run_start  = time.time()
    all_records: List[Dict] = []

    for arch in args.architectures:
        arch_cells = [c for c in cells if c.architecture == arch]
        if not arch_cells:
            continue

        # ── Phase 1 ──────────────────────────────────────────────────────
        clean_ckpt = _find_clean_ckpt(arch, args.seed, output_dir, reuse_root)
        if clean_ckpt is None:
            print(f"\n  Phase 1 TRAIN: arch={arch} seed={args.seed}", flush=True)
            clean_ckpt = _train_phase1(mat_path, arch, args.seed, output_dir)
        else:
            print(f"\n  Phase 1 REUSE: {clean_ckpt}", flush=True)

        if clean_ckpt is None:
            print(f"  SKIP all cells for arch={arch}: no checkpoint.", flush=True)
            for cell in arch_cells:
                all_records.append(_build_cell_record(cell, None))
            continue

        # ── Load data ONCE for this arch ──────────────────────────────────
        print(f"\n  Loading data for arch={arch} ...", flush=True)
        cfg, data = _load_data_and_cfg(mat_path, arch, args.seed)
        print(f"  Data loaded.", flush=True)

        # ── Run cells ─────────────────────────────────────────────────────
        for cell in arch_cells:
            summary_p = _summary_path(_cell_output_dir(output_dir, cell))
            if summary_p.exists():
                existing = _load_json(summary_p)
                if existing is not None:
                    print(f"\n  SKIP (exists): {cell.trigger_type} "
                          f"str={cell.trigger_strength} pr={cell.poison_rate} "
                          f"arch={arch}", flush=True)
                    all_records.append(_build_cell_record(cell, existing))
                    continue

            _print_banner(
                f"CELL  arch={arch}  type={cell.trigger_type}  "
                f"str={cell.trigger_strength}  pr={cell.poison_rate}  seed={cell.seed}"
            )
            cell_start = time.time()
            p2 = _run_cell_in_process(
                mat_path=mat_path,
                clean_ckpt=clean_ckpt,
                cell=cell,
                output_dir=output_dir,
                phase2_epochs=args.phase2_epochs,
                early_stop_patience=args.early_stop_patience,
                batch_size=args.batch_size,
                data=data,
                cfg=cfg,
            )
            status = "DONE" if p2 is not None else "FAILED"
            print(f"\n  {status}  ({_elapsed_str(cell_start)})  {cell}", flush=True)
            all_records.append(_build_cell_record(cell, p2))

    # ── Save outputs per sweep ────────────────────────────────────────────
    for sweep in args.sweeps:
        sweep_recs = _sweep_records(
            all_records, sweep,
            args.strength_values, args.poison_values, args.type_values,
        )
        _print_sweep_table(sweep_recs, sweep)
        _write_csv(sweep_recs, output_dir / f"track2_sweep_{sweep}.csv")
        _write_json(sweep_recs, output_dir / f"track2_sweep_{sweep}.json")

    # ── Master index ──────────────────────────────────────────────────────
    index = {
        "generated_at":      datetime.datetime.now().isoformat(),
        "seed":              args.seed,
        "architectures":     args.architectures,
        "sweeps":            args.sweeps,
        "phase2_epochs":     args.phase2_epochs,
        "early_stop_patience": args.early_stop_patience,
        "anchor": {
            "trigger_strength":  ANCHOR_STRENGTH,
            "poison_rate":       ANCHOR_POISON_RATE,
            "trigger_type":      ANCHOR_TRIGGER_TYPE,
        },
        "phase1_reused":     reuse_root is not None,
        "phase1_reuse_root": str(reuse_root) if reuse_root else None,
        "cells":             all_records,
    }
    idx_path = output_dir / "track2_index.json"
    idx_path.write_text(json.dumps(index, indent=2))
    print(f"\n  Index → {idx_path}", flush=True)
    print(f"  Total elapsed: {_elapsed_str(run_start)}", flush=True)


if __name__ == "__main__":
    main()
