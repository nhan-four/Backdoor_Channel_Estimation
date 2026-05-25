"""
Track 1 — Fair Architecture Comparison: non_residual vs residual.

Each (architecture, seed) cell runs its own Phase 1 (clean training) and
Phase 2 (BadNets injection), so there is no cross-seed or cross-arch checkpoint
reuse.  The runner supports incremental execution: already-finished cells are
skipped unless --force is passed.

Output layout:
    {output_dir}/
      {arch}/
        seed_{seed}/
          phase1/
            clean_model_best.pt
            phase1_summary.json
          phase2_badnets/
            badnets_model_best.pt
            phase2_test_summary.json   ← primary metric file
            phase2_test_summary.txt
            badnets_summary.json
      track1_index.json                ← manifest of all cells + status

Usage:
    python run_track1_arch_compare.py \\
        --mat_path /path/to/data.mat \\
        --output_dir ./results/track1_YYYYMMDD

    # Specify seeds / architectures explicitly:
    python run_track1_arch_compare.py \\
        --mat_path /path/to/data.mat \\
        --output_dir ./results/track1_YYYYMMDD \\
        --architectures non_residual residual \\
        --seeds 42 43 44 45 46
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PASS3_CRITERIA = {
    "clean_mse_budget":      0.60,
    "degradation_gap_min":   0.05,
    "degradation_ratio_min": 1.25,
}

DEFAULT_ARCHITECTURES = ["non_residual", "residual"]
DEFAULT_SEEDS         = [42, 43, 44, 45, 46]

# BadNets Phase-2 hyper-parameters (must match successful single-run config)
BADNETS_TRIGGER_STRENGTH = 20.0
BADNETS_POISON_RATE      = 0.10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cell_dir(output_dir: Path, arch: str, seed: int) -> Path:
    """Return the per-cell working directory."""
    return output_dir / arch / f"seed_{seed}"


def _phase1_done(cell_dir: Path) -> bool:
    """True when Phase 1 checkpoint + JSON both exist."""
    return (
        (cell_dir / "phase1" / "clean_model_best.pt").exists()
        and (cell_dir / "phase1" / "phase1_summary.json").exists()
    )


def _phase2_done(cell_dir: Path) -> bool:
    """True when Phase 2 test summary JSON exists."""
    return (cell_dir / "phase2_badnets" / "phase2_test_summary.json").exists()


def _load_json(path: Path) -> Optional[Dict]:
    """Return parsed JSON or None on error."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _cell_status(cell_dir: Path) -> str:
    """Return 'done', 'phase1_only', or 'pending'."""
    if _phase2_done(cell_dir):
        return "done"
    if _phase1_done(cell_dir):
        return "phase1_only"
    return "pending"


def _print_banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n  {msg}\n{'=' * 70}")


def _elapsed(start: float) -> str:
    s = int(time.time() - start)
    return f"{s // 3600:02d}h {(s % 3600) // 60:02d}m {s % 60:02d}s"


# ---------------------------------------------------------------------------
# Subprocess runners (each phase in its own python subprocess for clean VRAM)
# ---------------------------------------------------------------------------

def _run_subprocess(cmd: List[str], log_path: Path) -> bool:
    """Run a subprocess, tee-ing stdout+stderr to log_path.

    Returns True on success (exit code 0).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  LOG → {log_path}")
    with log_path.open("w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:         # type: ignore[union-attr]
            sys.stdout.write(line)
            log_fh.write(line)
            log_fh.flush()
        proc.wait()
    return proc.returncode == 0


def _run_phase1(
    mat_path: Path,
    cell_dir: Path,
    arch: str,
    seed: int,
    python_exe: str,
) -> bool:
    """Invoke run_two_phase.py --phase phase1 for one (arch, seed) cell."""
    script = Path(__file__).with_name("run_two_phase.py")
    cmd = [
        python_exe, "-u", str(script),
        "--mat_path",    str(mat_path),
        "--output_dir",  str(cell_dir),
        "--phase",       "phase1",
        "--architecture", arch,
        "--seed",        str(seed),
    ]
    log = cell_dir / "phase1" / "phase1_run.log"
    return _run_subprocess(cmd, log)


def _run_phase2_badnets(
    mat_path: Path,
    cell_dir: Path,
    arch: str,
    seed: int,
    python_exe: str,
    phase2_epochs: Optional[int] = None,
) -> bool:
    """Invoke run_two_phase.py --phase badnets for one (arch, seed) cell."""
    script   = Path(__file__).with_name("run_two_phase.py")
    ckpt     = cell_dir / "phase1" / "clean_model_best.pt"
    cmd = [
        python_exe, "-u", str(script),
        "--mat_path",          str(mat_path),
        "--output_dir",        str(cell_dir),
        "--phase",             "badnets",
        "--architecture",      arch,
        "--clean_checkpoint",  str(ckpt),
        "--trigger_strength",  str(BADNETS_TRIGGER_STRENGTH),
        "--poison_rate",       str(BADNETS_POISON_RATE),
        "--seed",              str(seed),
    ]
    if phase2_epochs is not None:
        cmd += ["--badnets_epochs", str(phase2_epochs)]
    log = cell_dir / "phase2_badnets" / "phase2_run.log"
    return _run_subprocess(cmd, log)


# ---------------------------------------------------------------------------
# Index / manifest
# ---------------------------------------------------------------------------

def _build_index(
    output_dir: Path,
    architectures: List[str],
    seeds: List[int],
) -> Dict:
    """Build the manifest dict (does not write to disk)."""
    cells = []
    for arch in architectures:
        for seed in seeds:
            cell_dir = _cell_dir(output_dir, arch, seed)
            status   = _cell_status(cell_dir)
            entry: Dict = {
                "architecture": arch,
                "seed":         seed,
                "cell_dir":     str(cell_dir),
                "status":       status,
            }
            if status in ("done", "phase1_only"):
                p1_json = _load_json(cell_dir / "phase1" / "phase1_summary.json")
                if p1_json:
                    entry["phase1_test_mse"]   = p1_json.get("test_mse")
                    entry["phase1_best_epoch"] = p1_json.get("best_epoch")
            if status == "done":
                p2_json = _load_json(
                    cell_dir / "phase2_badnets" / "phase2_test_summary.json"
                )
                if p2_json:
                    entry["val_clean_mse"]         = p2_json["val"]["clean_mse"]
                    entry["val_triggered_mse"]     = p2_json["val"]["triggered_mse"]
                    entry["val_degradation_gap"]   = p2_json["val"]["degradation_gap"]
                    entry["val_degradation_ratio"] = p2_json["val"]["degradation_ratio"]
                    entry["test_clean_mse"]        = p2_json["test"]["clean_mse"]
                    entry["test_triggered_mse"]    = p2_json["test"]["triggered_mse"]
                    entry["test_degradation_gap"]  = p2_json["test"]["degradation_gap"]
                    entry["test_degradation_ratio"]= p2_json["test"]["degradation_ratio"]
                    entry["overall_pass3"]         = p2_json.get("overall_pass3")
            cells.append(entry)

    return {
        "generated_at":  datetime.datetime.now().isoformat(),
        "output_dir":    str(output_dir),
        "architectures": architectures,
        "seeds":         seeds,
        "badnets_trigger_strength": BADNETS_TRIGGER_STRENGTH,
        "badnets_poison_rate":      BADNETS_POISON_RATE,
        "pass3_criteria": PASS3_CRITERIA,
        "cells":         cells,
    }


def _write_index(output_dir: Path, architectures: List[str], seeds: List[int]) -> None:
    idx = _build_index(output_dir, architectures, seeds)
    path = output_dir / "track1_index.json"
    path.write_text(json.dumps(idx, indent=2))
    print(f"  Updated index → {path}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track 1: fair architecture comparison (non_residual vs residual)"
    )
    parser.add_argument(
        "--mat_path", required=True, help="Path to data.mat"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Root output directory (default: results/track1_<timestamp>)",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=DEFAULT_ARCHITECTURES,
        choices=["non_residual", "residual"],
        help="Architectures to compare",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds to run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output already exists",
    )
    parser.add_argument(
        "--phase2_only",
        action="store_true",
        help="Skip Phase 1 when clean_model_best.pt already exists",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for subprocesses",
    )
    parser.add_argument(
        "--phase2_epochs",
        type=int,
        default=None,
        help="Override Phase-2 BadNets epochs (passes --badnets_epochs).",
    )
    return parser.parse_args()


def main() -> None:  # noqa: C901
    args = _parse_args()
    mat_path = Path(args.mat_path)
    if not mat_path.exists():
        sys.exit(f"ERROR: mat_path not found: {mat_path}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"results/track1_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    architectures: List[str] = args.architectures
    seeds: List[int]         = args.seeds

    total_cells = len(architectures) * len(seeds)
    _print_banner(
        f"Track 1 — Architecture Comparison\n"
        f"  archs={architectures}  seeds={seeds}  cells={total_cells}\n"
        f"  output_dir={output_dir}"
    )

    run_start = time.time()
    results: Dict[str, Dict] = {}   # {f"{arch}|{seed}": status_dict}

    for arch in architectures:
        for seed in seeds:
            cell_dir = _cell_dir(output_dir, arch, seed)
            cell_key = f"{arch}|seed_{seed}"
            cell_start = time.time()

            _print_banner(f"CELL  arch={arch}  seed={seed}")

            # ── Phase 1 ─────────────────────────────────────────────────
            skip_p1 = (
                not args.force
                and (_phase1_done(cell_dir) or args.phase2_only)
            )
            if skip_p1:
                print(f"  Phase 1 SKIP (checkpoint exists): {cell_dir / 'phase1'}")
            else:
                print(f"  Phase 1 START  arch={arch}  seed={seed}")
                ok = _run_phase1(mat_path, cell_dir, arch, seed, args.python)
                if not ok:
                    print(f"  Phase 1 FAILED for arch={arch} seed={seed}")
                    results[cell_key] = {"status": "phase1_failed", "arch": arch, "seed": seed}
                    _write_index(output_dir, architectures, seeds)
                    continue
                print(f"  Phase 1 DONE   ({_elapsed(cell_start)})")

            # ── Phase 2 (BadNets) ────────────────────────────────────────
            if not _phase1_done(cell_dir):
                print(f"  Phase 2 SKIP: Phase 1 checkpoint missing — cannot proceed")
                results[cell_key] = {"status": "no_phase1_ckpt", "arch": arch, "seed": seed}
                _write_index(output_dir, architectures, seeds)
                continue

            skip_p2 = (not args.force) and _phase2_done(cell_dir)
            if skip_p2:
                print(f"  Phase 2 SKIP (summary exists): {cell_dir / 'phase2_badnets'}")
            else:
                print(f"  Phase 2 START  arch={arch}  seed={seed}")
                p2_start = time.time()
                ok = _run_phase2_badnets(
                    mat_path,
                    cell_dir,
                    arch,
                    seed,
                    args.python,
                    phase2_epochs=args.phase2_epochs,
                )
                if not ok:
                    print(f"  Phase 2 FAILED for arch={arch} seed={seed}")
                    results[cell_key] = {"status": "phase2_failed", "arch": arch, "seed": seed}
                    _write_index(output_dir, architectures, seeds)
                    continue
                print(f"  Phase 2 DONE   ({_elapsed(p2_start)})")

            # ── Read summary ─────────────────────────────────────────────
            summary_path = cell_dir / "phase2_badnets" / "phase2_test_summary.json"
            p2 = _load_json(summary_path)
            if p2:
                print(
                    f"  RESULT  "
                    f"val ratio={p2['val']['degradation_ratio']:.3f}  "
                    f"test ratio={p2['test']['degradation_ratio']:.3f}  "
                    f"pass3={'YES' if p2.get('overall_pass3') else 'NO'}"
                )
                results[cell_key] = {
                    "status":    "done",
                    "arch":      arch,
                    "seed":      seed,
                    "pass3":     p2.get("overall_pass3"),
                    "val_ratio": p2["val"]["degradation_ratio"],
                    "test_ratio": p2["test"]["degradation_ratio"],
                    "elapsed":   _elapsed(cell_start),
                }
            else:
                results[cell_key] = {"status": "no_summary", "arch": arch, "seed": seed}

            _write_index(output_dir, architectures, seeds)

    # ── Final summary ────────────────────────────────────────────────────
    _print_banner(f"Track 1 COMPLETE — total {_elapsed(run_start)}")
    print(f"\n  {'Cell':<28} {'Status':<14} {'Val ratio':>10} {'Test ratio':>11} {'PASS3':>6}")
    print("  " + "─" * 74)
    for key, r in results.items():
        val_r  = f"{r.get('val_ratio', '-'):>10.3f}" if isinstance(r.get("val_ratio"), float) else f"{'–':>10}"
        test_r = f"{r.get('test_ratio', '-'):>11.3f}" if isinstance(r.get("test_ratio"), float) else f"{'–':>11}"
        pass3  = "YES" if r.get("pass3") else "NO "
        print(f"  {key:<28} {r['status']:<14} {val_r} {test_r} {pass3:>6}")
    print()
    print(f"  Index → {output_dir / 'track1_index.json'}")
    print(f"  Run collect_results.py --root {output_dir}  to aggregate stats")


if __name__ == "__main__":
    main()
