"""
Aggregate Track 1 results across architectures and seeds.

Discovers all phase2_test_summary.json files under --root, groups them by
architecture, and computes per-architecture statistics (mean ± std) for the
four key metrics on both val and test splits.

Outputs (in --output_dir, default <root>/aggregated/):
    per_seed_detail.csv            one row per (architecture, seed)
    aggregate_by_architecture.csv  mean ± std per architecture
    aggregate_by_architecture.json full statistics including counts & pass rate

Usage:
    # Standard Track 1 run (discover by arch/seed_* layout):
    python collect_results.py --root ./results/track1_YYYYMMDD

    # Point at any directory tree; the script searches recursively:
    python collect_results.py --root ./results --output_dir ./results/agg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Metrics we care about
# ---------------------------------------------------------------------------

METRIC_KEYS: List[Tuple[str, str]] = [
    ("val",  "clean_mse"),
    ("val",  "triggered_mse"),
    ("val",  "degradation_gap"),
    ("val",  "degradation_ratio"),
    ("test", "clean_mse"),
    ("test", "triggered_mse"),
    ("test", "degradation_gap"),
    ("test", "degradation_ratio"),
]

PASS3_CRITERIA = {
    "clean_mse_budget":      0.60,
    "degradation_gap_min":   0.05,
    "degradation_ratio_min": 1.25,
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_summaries(root: Path) -> List[Path]:
    """Find all phase2_test_summary.json files under root."""
    return sorted(root.rglob("phase2_test_summary.json"))


def _parse_summary(path: Path) -> Optional[Dict]:
    """
    Parse a phase2_test_summary.json and return a flat record dict,
    handling both old (no 'architecture' field) and new schema.
    """
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        print(f"  WARN: cannot read {path}: {exc}", file=sys.stderr)
        return None

    # --- resolve architecture -----------------------------------------------
    # 1. explicit field (new schema after patch)
    arch = raw.get("architecture")
    # 2. infer from directory path: .../non_residual/seed_42/...
    if not arch:
        for part in path.parts:
            if part in ("non_residual", "residual"):
                arch = part
                break
    # 3. explicit unknown marker for unusual directory layouts
    if not arch:
        arch = "unknown"

    # --- resolve seed --------------------------------------------------------
    seed = raw.get("seed")
    if seed is None:
        for part in path.parts:
            if part.startswith("seed_"):
                try:
                    seed = int(part.split("_", 1)[1])
                except ValueError:
                    pass
                break

    record: Dict = {
        "architecture": arch,
        "seed":         seed,
        "summary_path": str(path),
        "approach":     raw.get("approach", "badnets"),
        "best_epoch":   raw.get("best_epoch"),
        "overall_pass3": raw.get("overall_pass3"),
    }
    # flatten val / test metrics
    for split in ("val", "test"):
        split_data = raw.get(split, {})
        for metric in ("clean_mse", "triggered_mse", "degradation_gap", "degradation_ratio"):
            record[f"{split}_{metric}"] = split_data.get(metric)
    return record


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mu = _mean(values)
    variance = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _pass3_check(record: Dict) -> bool:
    """Evaluate PASS3 criteria against test split."""
    clean  = record.get("test_clean_mse")
    gap    = record.get("test_degradation_gap")
    ratio  = record.get("test_degradation_ratio")
    if None in (clean, gap, ratio):
        return False
    return (
        clean  <= PASS3_CRITERIA["clean_mse_budget"]
        and gap    >= PASS3_CRITERIA["degradation_gap_min"]
        and ratio  >= PASS3_CRITERIA["degradation_ratio_min"]
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_by_architecture(records: List[Dict]) -> Dict[str, Dict]:
    """Group records by architecture and compute mean ± std for each metric."""
    by_arch: Dict[str, List[Dict]] = {}
    for rec in records:
        arch = rec["architecture"]
        by_arch.setdefault(arch, []).append(rec)

    stats: Dict[str, Dict] = {}
    for arch, recs in sorted(by_arch.items()):
        arch_stats: Dict = {
            "architecture": arch,
            "n_seeds":      len(recs),
            "seeds":        sorted(r["seed"] for r in recs if r["seed"] is not None),
        }

        # compute pass3 rate
        pass3_flags = [_pass3_check(r) for r in recs]
        arch_stats["test_pass3_rate"] = sum(pass3_flags) / len(pass3_flags)
        arch_stats["test_pass3_count"] = int(sum(pass3_flags))

        # compute mean ± std for every metric
        for split, metric in METRIC_KEYS:
            col = f"{split}_{metric}"
            values = [r[col] for r in recs if r.get(col) is not None]
            arch_stats[f"{col}_mean"]   = _mean(values)
            arch_stats[f"{col}_std"]    = _std(values)
            arch_stats[f"{col}_min"]    = min(values) if values else float("nan")
            arch_stats[f"{col}_max"]    = max(values) if values else float("nan")
            arch_stats[f"{col}_values"] = values

        stats[arch] = arch_stats
    return stats


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_per_seed_csv(records: List[Dict], out_path: Path) -> None:
    """Write one row per (architecture, seed) with all flat metrics."""
    if not records:
        return
    fieldnames = [
        "architecture", "seed",
        "val_clean_mse",         "val_triggered_mse",
        "val_degradation_gap",   "val_degradation_ratio",
        "test_clean_mse",        "test_triggered_mse",
        "test_degradation_gap",  "test_degradation_ratio",
        "test_pass3",
        "overall_pass3",
        "best_epoch",
        "summary_path",
    ]
    rows = sorted(records, key=lambda r: (r["architecture"], r.get("seed") or 0))
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row["test_pass3"] = _pass3_check(row)
            writer.writerow(row)
    print(f"  CSV (per seed)  → {out_path}")


def _write_aggregate_csv(stats: Dict[str, Dict], out_path: Path) -> None:
    """Write one row per architecture with mean ± std for each metric."""
    metric_cols = [
        f"{split}_{metric}"
        for split, metric in METRIC_KEYS
    ]
    fieldnames = ["architecture", "n_seeds", "test_pass3_rate", "test_pass3_count"]
    for col in metric_cols:
        fieldnames += [f"{col}_mean", f"{col}_std", f"{col}_min", f"{col}_max"]

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for arch, s in sorted(stats.items()):
            row = {k: s.get(k) for k in fieldnames}
            writer.writerow(row)
    print(f"  CSV (aggregate) → {out_path}")


def _write_aggregate_json(stats: Dict[str, Dict], out_path: Path) -> None:
    """Write full statistics including per-seed value arrays."""
    serialisable = {}
    for arch, s in stats.items():
        serialisable[arch] = {
            k: (round(v, 8) if isinstance(v, float) and not math.isnan(v) else v)
            for k, v in s.items()
        }
    out_path.write_text(json.dumps(serialisable, indent=2))
    print(f"  JSON (aggregate)→ {out_path}")


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def _print_summary_table(stats: Dict[str, Dict]) -> None:
    """Print a compact comparison table to stdout."""
    metric_cols = [f"{s}_{m}" for s, m in METRIC_KEYS]
    hdr_items   = [f"{m[:20]:>22}" for m in metric_cols]

    print(f"\n{'Architecture':<20} {'N':>3} {'pass3_rate':>10}  " + "  ".join(hdr_items))
    print("─" * (20 + 3 + 10 + len(hdr_items) * 24 + 10))

    for arch, s in sorted(stats.items()):
        row_mean = "  ".join(
            f"{s.get(f'{col}_mean', float('nan')):>10.4f} ± {s.get(f'{col}_std', float('nan')):.4f}"
            for col in metric_cols
        )
        print(
            f"{arch:<20} {s['n_seeds']:>3} {s['test_pass3_rate']:>10.2%}  {row_mean}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Track 3 aggregation (defense evaluation)
# ---------------------------------------------------------------------------

TRACK3_METRIC_KEYS: List[str] = [
    "clean_mse_before",
    "triggered_mse_before",
    "degradation_gap_before",
    "degradation_ratio_before",
    "clean_mse_after",
    "triggered_mse_after",
    "degradation_gap_after",
    "degradation_ratio_after",
]


def _discover_track3_summaries(root: Path) -> List[Path]:
    """Find all track3_defense_summary.json files under root."""
    return sorted(root.rglob("track3_defense_summary.json"))


def _parse_track3_summary(path: Path) -> List[Dict]:
    """Parse one track3 summary file into a list of per-defense flat records."""
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        print(f"  WARN: cannot read {path}: {exc}", file=sys.stderr)
        return []
    source_seed = raw.get("source_seed")
    architecture = raw.get("architecture", "non_residual")
    out: List[Dict] = []
    for res in raw.get("results", []):
        if "error" in res:
            continue
        flat = {
            "summary_path": str(path),
            "architecture": architecture,
            "seed":         source_seed,
            "defense":      res.get("defense"),
            "attack_survives": bool(res.get("attack_survives", False)),
            "defense_params": json.dumps(res.get("defense_params", {})),
            "is_detector_only": bool(
                res.get("defense_metadata", {}).get("is_detector_only", False)
            ),
        }
        for key in TRACK3_METRIC_KEYS:
            flat[key] = res.get(key)
        det = res.get("detection", {})
        flat["clean_false_positive_rate"] = det.get("clean_false_positive_rate")
        flat["triggered_true_positive_rate"] = det.get("triggered_true_positive_rate")
        out.append(flat)
    return out


def _aggregate_track3_by_defense(records: List[Dict]) -> Dict[str, Dict]:
    """Group defense records by defense name; compute mean ± std and survival rate."""
    by_def: Dict[str, List[Dict]] = {}
    for rec in records:
        by_def.setdefault(rec["defense"], []).append(rec)

    stats: Dict[str, Dict] = {}
    for defense_name, recs in sorted(by_def.items()):
        survives = [r for r in recs if r["attack_survives"]]
        defense_stats: Dict = {
            "defense":          defense_name,
            "n_runs":           len(recs),
            "seeds":            sorted({r["seed"] for r in recs if r["seed"] is not None}, key=int),
            "architectures":    sorted({r["architecture"] for r in recs}),
            "attack_survives_count": len(survives),
            "attack_survives_rate":  len(survives) / len(recs),
        }
        for key in TRACK3_METRIC_KEYS:
            values = [r[key] for r in recs if r.get(key) is not None]
            defense_stats[f"{key}_mean"] = _mean(values)
            defense_stats[f"{key}_std"]  = _std(values)
            defense_stats[f"{key}_values"] = values
        stats[defense_name] = defense_stats
    return stats


def _write_track3_per_run_csv(records: List[Dict], out_path: Path) -> None:
    """One row per (defense, seed) with all BEFORE/AFTER metrics."""
    if not records:
        return
    fieldnames = [
        "defense", "architecture", "seed",
        "clean_mse_before", "triggered_mse_before",
        "degradation_gap_before", "degradation_ratio_before",
        "clean_mse_after", "triggered_mse_after",
        "degradation_gap_after", "degradation_ratio_after",
        "attack_survives",
        "clean_false_positive_rate", "triggered_true_positive_rate",
        "is_detector_only",
        "defense_params",
        "summary_path",
    ]
    rows = sorted(
        records,
        key=lambda r: (r["defense"], r.get("architecture") or "", int(r["seed"]) if r.get("seed") is not None else 0),
    )
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  CSV (per run)   → {out_path}")


def _write_track3_aggregate_csv(stats: Dict[str, Dict], out_path: Path) -> None:
    """One row per defense with mean ± std for each metric."""
    fieldnames = ["defense", "n_runs", "attack_survives_rate", "attack_survives_count"]
    for key in TRACK3_METRIC_KEYS:
        fieldnames += [f"{key}_mean", f"{key}_std"]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for defense_name, s in sorted(stats.items()):
            writer.writerow({k: s.get(k) for k in fieldnames})
    print(f"  CSV (aggregate) → {out_path}")


def _write_track3_aggregate_json(stats: Dict[str, Dict], out_path: Path) -> None:
    serialisable = {}
    for defense_name, s in stats.items():
        serialisable[defense_name] = {
            k: (round(v, 8) if isinstance(v, float) and not math.isnan(v) else v)
            for k, v in s.items()
        }
    out_path.write_text(json.dumps(serialisable, indent=2, default=str))
    print(f"  JSON (aggregate)→ {out_path}")


def _aggregate_track3_by_arch_defense(records: List[Dict]) -> Dict[str, Dict]:
    """Group by (architecture, defense); compute mean/std for each metric."""
    by_key: Dict[tuple, List[Dict]] = {}
    for rec in records:
        key = (rec.get("architecture", "unknown"), rec["defense"])
        by_key.setdefault(key, []).append(rec)

    stats: Dict[str, Dict] = {}
    for (arch, defense), recs in sorted(by_key.items()):
        survives = [r for r in recs if r["attack_survives"]]
        entry: Dict = {
            "architecture":        arch,
            "defense":             defense,
            "n_runs":              len(recs),
            "seeds":               sorted({r["seed"] for r in recs if r["seed"] is not None}, key=int),
            "attack_survives_count": len(survives),
            "attack_survives_rate":  len(survives) / len(recs),
        }
        for key in TRACK3_METRIC_KEYS:
            values = [r[key] for r in recs if r.get(key) is not None]
            entry[f"{key}_mean"] = _mean(values)
            entry[f"{key}_std"]  = _std(values)
        stats[f"{arch}__{defense}"] = entry
    return stats


def _write_track3_arch_def_csv(stats: Dict[str, Dict], out_path: Path) -> None:
    """One row per (architecture, defense) with mean ± std for each metric."""
    fieldnames = ["architecture", "defense", "n_runs", "attack_survives_rate", "attack_survives_count", "seeds"]
    for key in TRACK3_METRIC_KEYS:
        fieldnames += [f"{key}_mean", f"{key}_std"]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in sorted(stats.values(), key=lambda e: (e["architecture"], e["defense"])):
            row = {k: entry.get(k) for k in fieldnames}
            row["seeds"] = str(entry.get("seeds", []))
            writer.writerow(row)
    print(f"  CSV (arch×def)  → {out_path}")


def _write_track3_arch_def_json(stats: Dict[str, Dict], out_path: Path) -> None:
    serialisable = {}
    for combo_key, s in stats.items():
        serialisable[combo_key] = {
            k: (round(v, 8) if isinstance(v, float) and not math.isnan(v) else v)
            for k, v in s.items()
        }
    out_path.write_text(json.dumps(serialisable, indent=2, default=str))
    print(f"  JSON (arch×def) → {out_path}")


def _write_track3_per_run_json(records: List[Dict], out_path: Path) -> None:
    """JSON version of per-run detail: array of flat record dicts."""
    def _clean(v):
        if isinstance(v, float) and not math.isnan(v):
            return round(v, 8)
        return v

    rows = sorted(
        records,
        key=lambda r: (r["defense"], r.get("architecture") or "", int(r["seed"]) if r.get("seed") is not None else 0),
    )
    out_path.write_text(json.dumps([{k: _clean(v) for k, v in r.items()} for r in rows], indent=2, default=str))
    print(f"  JSON (per run)  → {out_path}")


def _print_track3_summary_table(stats: Dict[str, Dict]) -> None:
    print(
        f"\n{'Defense':<22} {'N':>3} {'survive':>9}  "
        f"{'clean_bef':>10} {'trig_bef':>10} {'ratio_bef':>10}  "
        f"{'clean_aft':>10} {'trig_aft':>10} {'ratio_aft':>10}"
    )
    print("─" * 115)
    for defense_name, s in sorted(stats.items()):
        print(
            f"{defense_name:<22} {s['n_runs']:>3} {s['attack_survives_rate']:>9.2%}  "
            f"{s.get('clean_mse_before_mean', float('nan')):>10.4f} "
            f"{s.get('triggered_mse_before_mean', float('nan')):>10.4f} "
            f"{s.get('degradation_ratio_before_mean', float('nan')):>10.4f}  "
            f"{s.get('clean_mse_after_mean', float('nan')):>10.4f} "
            f"{s.get('triggered_mse_after_mean', float('nan')):>10.4f} "
            f"{s.get('degradation_ratio_after_mean', float('nan')):>10.4f}"
        )


def _run_track3_pipeline(root: Path, out_dir: Path) -> None:
    paths = _discover_track3_summaries(root)
    if not paths:
        sys.exit(f"ERROR: no track3_defense_summary.json found under {root}")
    print(f"Found {len(paths)} Track-3 summary file(s) under {root}")

    records: List[Dict] = []
    for p in paths:
        records.extend(_parse_track3_summary(p))
    if not records:
        sys.exit("ERROR: all Track-3 summary files failed to parse")
    print(f"Parsed {len(records)} (defense × run) record(s).")

    stats         = _aggregate_track3_by_defense(records)
    arch_def_stats = _aggregate_track3_by_arch_defense(records)

    # Per-run outputs (canonical name + 5seed-expansion alias)
    _write_track3_per_run_csv(records,  out_dir / "track3_per_run_detail.csv")
    _write_track3_per_run_csv(records,  out_dir / "track3_5seed_defense_results.csv")
    _write_track3_per_run_json(records, out_dir / "track3_5seed_defense_results.json")

    # Defense-level aggregate
    _write_track3_aggregate_csv(stats,  out_dir / "track3_aggregate_by_defense.csv")
    _write_track3_aggregate_json(stats, out_dir / "track3_aggregate_by_defense.json")

    # Architecture × defense comparison (new)
    _write_track3_arch_def_csv(arch_def_stats,  out_dir / "track3_arch_defense_compare.csv")
    _write_track3_arch_def_json(arch_def_stats, out_dir / "track3_arch_defense_compare.json")

    _print_track3_summary_table(stats)
    print(f"\nAll Track-3 outputs in {out_dir}")


# ---------------------------------------------------------------------------
# Track 4 aggregation (receiver impact)
# ---------------------------------------------------------------------------

TRACK4_METRIC_KEYS: List[str] = ["ber", "ser", "evm"]


def _discover_track4_summaries(root: Path) -> List[Path]:
    """Find all track4_receiver_metrics.json files under root."""
    return sorted(root.rglob("track4_receiver_metrics.json"))


def _parse_track4_summary(path: Path) -> List[Dict]:
    """Parse one Track-4 summary JSON into a flat list of per-sweep-point records."""
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        print(f"  WARN: cannot read {path}: {exc}", file=sys.stderr)
        return []
    records: List[Dict] = []
    for rec in raw.get("records", []):
        flat = {
            "summary_path": str(path),
            "architecture": raw.get("architecture", "non_residual"),
            "trigger_strength": raw.get("trigger_strength"),
            "seed":       rec.get("seed"),
            "condition":  rec.get("condition"),
            "split":      rec.get("split"),
            "modulation": rec.get("modulation"),
            "snr_db":     rec.get("snr_db"),
            "ber":        rec.get("ber"),
            "ser":        rec.get("ser"),
            "evm":        rec.get("evm"),
            "num_pixels": rec.get("num_pixels"),
            "num_samples": rec.get("num_samples"),
        }
        records.append(flat)
    return records


def _aggregate_track4(records: List[Dict]) -> Dict[str, Dict]:
    """Group Track-4 records by (architecture, condition, split, modulation, snr_db) across seeds."""
    grouped: Dict[tuple, List[Dict]] = {}
    for r in records:
        arch = r.get("architecture") or "non_residual"
        key = (
            arch,
            r["condition"],
            r["split"],
            r["modulation"],
            float(r["snr_db"]) if r["snr_db"] is not None else None,
        )
        grouped.setdefault(key, []).append(r)

    stats: Dict[str, Dict] = {}
    for key, recs in grouped.items():
        key_str = "|".join(str(k) for k in key)
        seeds = sorted({r["seed"] for r in recs if r["seed"] is not None},
                       key=lambda s: int(s) if s is not None else 0)
        entry: Dict = {
            "architecture": key[0],
            "condition":  key[1],
            "split":      key[2],
            "modulation": key[3],
            "snr_db":     key[4],
            "n_seeds":    len(seeds),
            "seeds":      seeds,
        }
        for metric in TRACK4_METRIC_KEYS:
            values = [r[metric] for r in recs if r.get(metric) is not None]
            entry[f"{metric}_mean"] = _mean(values)
            entry[f"{metric}_std"]  = _std(values)
            entry[f"{metric}_values"] = values
        stats[key_str] = entry
    return stats


def _write_track4_per_run_csv(records: List[Dict], out_path: Path) -> None:
    """One row per (seed, condition, split, modulation, snr_db)."""
    if not records:
        return
    fieldnames = [
        "seed", "condition", "split", "modulation", "snr_db",
        "ber", "ser", "evm", "num_pixels", "num_samples",
        "architecture", "trigger_strength", "summary_path",
    ]
    rows = sorted(
        records,
        key=lambda r: (
            r.get("condition") or "",
            r.get("split") or "",
            r.get("modulation") or "",
            float(r["snr_db"]) if r.get("snr_db") is not None else 0.0,
            int(r["seed"]) if r.get("seed") is not None else 0,
        ),
    )
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  CSV (per run)   → {out_path}")


def _write_track4_aggregate_csv(stats: Dict[str, Dict], out_path: Path) -> None:
    """One row per (architecture, condition, split, modulation, snr_db) with mean ± std."""
    fieldnames = [
        "architecture", "condition", "split", "modulation", "snr_db", "n_seeds",
        "ber_mean", "ber_std",
        "ser_mean", "ser_std",
        "evm_mean", "evm_std",
    ]
    rows = sorted(
        stats.values(),
        key=lambda r: (
            r.get("architecture") or "",
            r["condition"] or "",
            r["split"] or "",
            r["modulation"] or "",
            float(r["snr_db"]) if r["snr_db"] is not None else 0.0,
        ),
    )
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  CSV (aggregate) → {out_path}")


def _write_track4_aggregate_json(stats: Dict[str, Dict], out_path: Path) -> None:
    serialisable = {}
    for key, s in stats.items():
        serialisable[key] = {
            k: (round(v, 8) if isinstance(v, float) and not math.isnan(v) else v)
            for k, v in s.items()
        }
    out_path.write_text(json.dumps(serialisable, indent=2, default=str))
    print(f"  JSON (aggregate)→ {out_path}")


def _print_track4_summary_table(stats: Dict[str, Dict]) -> None:
    rows = sorted(
        stats.values(),
        key=lambda r: (
            r.get("architecture") or "",
            r["condition"] or "",
            r["split"] or "",
            r["modulation"] or "",
            float(r["snr_db"]) if r["snr_db"] is not None else 0.0,
        ),
    )
    print(
        f"\n{'arch':<14} {'condition':<34} {'split':<6} {'mod':<6} {'snr':>4} "
        f"{'n':>2} {'ber_mean':>10} ± {'ber_std':<8} "
        f"{'ser_mean':>10} ± {'ser_std':<8} "
        f"{'evm_mean':>10} ± {'evm_std':<8}"
    )
    print("─" * 155)
    for r in rows:
        print(
            f"{(r.get('architecture') or ''):<14} {r['condition']:<34} {r['split']:<6} {r['modulation']:<6} "
            f"{float(r['snr_db']):>4.0f} {r['n_seeds']:>2} "
            f"{r.get('ber_mean', float('nan')):>10.6f} ± {r.get('ber_std', float('nan')):<8.6f} "
            f"{r.get('ser_mean', float('nan')):>10.6f} ± {r.get('ser_std', float('nan')):<8.6f} "
            f"{r.get('evm_mean', float('nan')):>10.4f} ± {r.get('evm_std', float('nan')):<8.4f}"
        )


def _run_track4_pipeline(root: Path, out_dir: Path) -> None:
    paths = _discover_track4_summaries(root)
    if not paths:
        sys.exit(f"ERROR: no track4_receiver_metrics.json found under {root}")
    print(f"Found {len(paths)} Track-4 summary file(s) under {root}")

    records: List[Dict] = []
    for p in paths:
        records.extend(_parse_track4_summary(p))
    if not records:
        sys.exit("ERROR: all Track-4 summary files failed to parse")
    print(f"Parsed {len(records)} (seed × condition × split × mod × snr) record(s).")

    stats = _aggregate_track4(records)
    _write_track4_per_run_csv(records, out_dir / "track4_per_run_detail.csv")
    _write_track4_aggregate_csv(stats,  out_dir / "track4_aggregate_by_condition_snr_mod.csv")
    _write_track4_aggregate_json(stats, out_dir / "track4_aggregate_by_condition_snr_mod.json")
    _write_track4_aggregate_csv(stats,  out_dir / "track4_arch_compare.csv")
    _write_track4_aggregate_json(stats, out_dir / "track4_arch_compare.json")
    _print_track4_summary_table(stats)
    print(f"\nAll Track-4 outputs in {out_dir}")


# ---------------------------------------------------------------------------
# Track 5 aggregation (stealthiness vs effectiveness)
# ---------------------------------------------------------------------------

def _infer_architecture_from_path_or_text(value) -> str | None:
    """Infer residual/non_residual from a metadata string or path component."""
    if value is None:
        return None
    try:
        parts = [part.lower() for part in Path(str(value)).parts]
    except Exception:
        parts = []
    text = str(value).lower()
    if "non_residual" in parts or "non_residual" in text:
        return "non_residual"
    if "residual" in parts or "/residual/" in text or "\\residual\\" in text:
        return "residual"
    return None


def _track5_summary_architecture(raw: Dict, path: Path) -> str | None:
    """Read architecture from new summaries; infer from old summary paths when needed."""
    return (
        raw.get("architecture")
        or raw.get("model_architecture")
        or _infer_architecture_from_path_or_text(raw.get("backdoor_ckpt"))
        or _infer_architecture_from_path_or_text(raw.get("backdoor_summary"))
        or _infer_architecture_from_path_or_text(path)
    )


def _discover_track5_summaries(root: Path) -> List[Path]:
    """Find all track5_stealth_summary.json files under root."""
    return sorted(root.rglob("track5_stealth_summary.json"))


def _parse_track5_summary(path: Path) -> List[Dict]:
    """Parse a Track-5 summary; one flat row per (record × detector)."""
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        print(f"  WARN: cannot read {path}: {exc}", file=sys.stderr)
        return []
    rows: List[Dict] = []
    summary_architecture = _track5_summary_architecture(raw, path)
    for rec in raw.get("records", []):
        deg = rec.get("degradation") or {}
        base = {
            "summary_path":     str(path),
            "architecture":     rec.get("architecture") or summary_architecture,
            "trigger_type":     rec.get("trigger_type"),
            "trigger_strength": rec.get("trigger_strength"),
            "seed":             rec.get("seed"),
            "poison_rate":      rec.get("poison_rate"),
            "trigger_energy":   rec.get("trigger_energy"),
            "signal_energy":    rec.get("signal_energy"),
            "trigger_to_signal_ratio":    rec.get("trigger_to_signal_ratio"),
            "trigger_to_signal_ratio_db": rec.get("trigger_to_signal_ratio_db"),
            "trigger_linf":     rec.get("trigger_linf"),
            "trigger_active_fraction": rec.get("trigger_active_fraction"),
            "clean_mse":         deg.get("clean_mse"),
            "triggered_mse":     deg.get("triggered_mse"),
            "degradation_gap":   deg.get("degradation_gap"),
            "degradation_ratio": deg.get("degradation_ratio"),
        }
        for det in rec.get("detectors", []):
            flat = {
                **base,
                "detector":                det.get("detector"),
                "is_trigger_aware":        det.get("is_trigger_aware"),
                "detectability_auc":       det.get("detectability_auc"),
                "detectability_accuracy":  det.get("detectability_accuracy"),
            }
            rows.append(flat)
    return rows


def _aggregate_track5(records: List[Dict]) -> Dict[str, Dict]:
    """Group by (architecture, trigger_type, trigger_strength, poison_rate, detector) across seeds."""
    grouped: Dict[tuple, List[Dict]] = {}
    for r in records:
        key = (
            r.get("architecture"),
            r.get("trigger_type"),
            float(r["trigger_strength"]) if r.get("trigger_strength") is not None else None,
            r.get("poison_rate"),
            r.get("detector"),
        )
        grouped.setdefault(key, []).append(r)

    stats: Dict[str, Dict] = {}
    for key, recs in grouped.items():
        key_str = "|".join(str(k) for k in key)
        seeds = sorted({r["seed"] for r in recs if r["seed"] is not None},
                       key=lambda s: int(s) if s is not None else 0)
        entry: Dict = {
            "architecture":     key[0],
            "trigger_type":     key[1],
            "trigger_strength": key[2],
            "poison_rate":      key[3],
            "detector":         key[4],
            "n_seeds":          len(seeds),
            "seeds":             seeds,
        }
        for metric in (
            "trigger_energy", "signal_energy",
            "trigger_to_signal_ratio", "trigger_to_signal_ratio_db",
            "detectability_auc", "detectability_accuracy",
            "clean_mse", "triggered_mse",
            "degradation_gap", "degradation_ratio",
        ):
            values = [r[metric] for r in recs if r.get(metric) is not None]
            entry[f"{metric}_mean"] = _mean(values)
            entry[f"{metric}_std"]  = _std(values)
        stats[key_str] = entry
    return stats


def _write_track5_per_run_csv(records: List[Dict], out_path: Path) -> None:
    if not records:
        return
    fieldnames = [
        "architecture", "trigger_type", "trigger_strength", "seed", "poison_rate",
        "trigger_energy", "signal_energy",
        "trigger_to_signal_ratio", "trigger_to_signal_ratio_db",
        "trigger_linf", "trigger_active_fraction",
        "detector", "is_trigger_aware",
        "detectability_auc", "detectability_accuracy",
        "clean_mse", "triggered_mse", "degradation_gap", "degradation_ratio",
        "summary_path",
    ]
    rows = sorted(
        records,
        key=lambda r: (
            r.get("architecture") or "",
            r.get("trigger_type") or "",
            float(r["trigger_strength"]) if r.get("trigger_strength") is not None else 0.0,
            int(r["seed"]) if r.get("seed") is not None else 0,
            r.get("detector") or "",
        ),
    )
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  CSV (per run)   → {out_path}")


def _write_track5_aggregate_csv(stats: Dict[str, Dict], out_path: Path) -> None:
    fieldnames = [
        "architecture", "trigger_type", "trigger_strength", "poison_rate", "detector", "n_seeds",
        "trigger_to_signal_ratio_db_mean",
        "detectability_auc_mean", "detectability_auc_std",
        "detectability_accuracy_mean", "detectability_accuracy_std",
        "degradation_ratio_mean", "degradation_ratio_std",
        "clean_mse_mean", "triggered_mse_mean",
    ]
    rows = sorted(
        stats.values(),
        key=lambda r: (
            r.get("architecture") or "",
            r.get("architecture") or "",
            r.get("trigger_type") or "",
            float(r["trigger_strength"]) if r.get("trigger_strength") is not None else 0.0,
            r.get("detector") or "",
        ),
    )
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  CSV (aggregate) → {out_path}")


def _write_track5_aggregate_json(stats: Dict[str, Dict], out_path: Path) -> None:
    serialisable = {}
    for key, s in stats.items():
        serialisable[key] = {
            k: (round(v, 8) if isinstance(v, float) and not math.isnan(v) else v)
            for k, v in s.items()
        }
    out_path.write_text(json.dumps(serialisable, indent=2, default=str))
    print(f"  JSON (aggregate)→ {out_path}")


def _print_track5_summary_table(stats: Dict[str, Dict]) -> None:
    print(
        f"\n{'architecture':<13} {'trigger_type':<16} {'strength':>8} {'detector':<16} {'n':>2} "
        f"{'TSR_dB':>7} {'AUC':>12} {'Acc':>12} {'deg_ratio':>12}"
    )
    print("─" * 100)
    rows = sorted(
        stats.values(),
        key=lambda r: (
            r.get("architecture") or "",
            r.get("architecture") or "",
            r.get("trigger_type") or "",
            float(r["trigger_strength"]) if r.get("trigger_strength") is not None else 0.0,
            r.get("detector") or "",
        ),
    )
    for r in rows:
        print(
            f"{str(r.get('architecture')):<13} "
            f"{str(r.get('trigger_type')):<16} "
            f"{float(r['trigger_strength']):>8.3f} "
            f"{str(r.get('detector')):<16} "
            f"{r['n_seeds']:>2} "
            f"{r.get('trigger_to_signal_ratio_db_mean', float('nan')):>+7.2f} "
            f"{r.get('detectability_auc_mean', float('nan')):>6.4f} ± {r.get('detectability_auc_std', float('nan')):.4f} "
            f"{r.get('detectability_accuracy_mean', float('nan')):>6.4f} ± {r.get('detectability_accuracy_std', float('nan')):.4f} "
            f"{r.get('degradation_ratio_mean', float('nan')):>6.4f} ± {r.get('degradation_ratio_std', float('nan')):.4f}"
        )


def _run_track5_pipeline(root: Path, out_dir: Path) -> None:
    paths = _discover_track5_summaries(root)
    if not paths:
        sys.exit(f"ERROR: no track5_stealth_summary.json found under {root}")
    print(f"Found {len(paths)} Track-5 summary file(s) under {root}")

    records: List[Dict] = []
    for p in paths:
        records.extend(_parse_track5_summary(p))
    if not records:
        sys.exit("ERROR: all Track-5 summary files failed to parse")
    print(f"Parsed {len(records)} (combo × detector) record(s).")

    stats = _aggregate_track5(records)
    _write_track5_per_run_csv(records, out_dir / "track5_per_run_detail.csv")
    _write_track5_aggregate_csv(stats, out_dir / "track5_aggregate_by_trigger_detector.csv")
    _write_track5_aggregate_json(stats, out_dir / "track5_aggregate_by_trigger_detector.json")
    _print_track5_summary_table(stats)
    print(f"\nAll Track-5 outputs in {out_dir}")


# ---------------------------------------------------------------------------
# Track 2 aggregation (trigger design space ablation)
# ---------------------------------------------------------------------------

TRACK2_SWEEPS = ["strength", "poison", "type"]


def _discover_track2_files(root: Path) -> List[Path]:
    """Find all track2_pilot_*.json files (one per sweep axis) under root."""
    found: List[Path] = []
    for sweep in TRACK2_SWEEPS:
        found.extend(sorted(root.rglob(f"track2_pilot_{sweep}.json")))
    return found


def _parse_track2_sweep(path: Path) -> Tuple[str, List[Dict]]:
    """Return (sweep_axis, records) from one Track-2 sweep JSON file."""
    stem = path.stem  # e.g. "track2_sweep_strength"
    sweep = stem.replace("track2_sweep_", "").replace("track2_pilot_", "")
    try:
        records = json.loads(path.read_text())
    except Exception as exc:
        print(f"  WARN: cannot read {path}: {exc}", file=sys.stderr)
        records = []
    return sweep, records


def _aggregate_track2_by_key(records: List[Dict], group_by: str) -> Dict[str, Dict]:
    """Group records by (group_by × architecture) and compute mean ± std."""
    grouped: Dict[tuple, List[Dict]] = {}
    for rec in records:
        key = (rec.get(group_by), rec.get("architecture"))
        grouped.setdefault(key, []).append(rec)

    stats: Dict[str, Dict] = {}
    for key, recs in sorted(grouped.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        grp_val, arch = key
        stat_key = f"{grp_val}|{arch}"
        entry: Dict = {
            group_by:       grp_val,
            "architecture": arch,
            "n_runs":       len(recs),
            "seeds":        sorted({r["seed"] for r in recs if r.get("seed") is not None}),
        }
        for metric in (
            "test_clean_mse", "test_triggered_mse",
            "test_degradation_gap", "test_degradation_ratio",
            "val_clean_mse", "val_triggered_mse",
            "val_degradation_gap", "val_degradation_ratio",
        ):
            values = [r[metric] for r in recs if r.get(metric) is not None]
            entry[f"{metric}_mean"] = _mean(values)
            entry[f"{metric}_std"]  = _std(values)
        pass3_flags = [
            (r.get("test_clean_mse") or 999) <= PASS3_CRITERIA["clean_mse_budget"]
            and (r.get("test_degradation_gap") or 0) >= PASS3_CRITERIA["degradation_gap_min"]
            and (r.get("test_degradation_ratio") or 0) >= PASS3_CRITERIA["degradation_ratio_min"]
            for r in recs
        ]
        entry["pass3_rate"]  = sum(pass3_flags) / len(pass3_flags) if pass3_flags else 0.0
        entry["pass3_count"] = int(sum(pass3_flags))
        stats[stat_key] = entry
    return stats


def _write_track2_csv(records: List[Dict], out_path: Path) -> None:
    if not records:
        return
    fieldnames = [
        "architecture", "seed", "trigger_type", "trigger_strength", "poison_rate",
        "status",
        "test_clean_mse", "test_triggered_mse", "test_degradation_gap", "test_degradation_ratio",
        "val_clean_mse",  "val_triggered_mse",  "val_degradation_gap",  "val_degradation_ratio",
        "overall_pass3", "best_epoch",
    ]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    print(f"  CSV → {out_path}")


def _write_track2_json(records: List[Dict], out_path: Path) -> None:
    out_path.write_text(json.dumps(records, indent=2))
    print(f"  JSON → {out_path}")


def _print_track2_table(records: List[Dict], sweep: str) -> None:
    """Print a compact Track-2 comparison table for one sweep."""
    print(f"\n{'─'*72}")
    print(f"Track 2 — {sweep.upper()} SWEEP")
    print(f"{'─'*72}")
    print(f"  {'arch':<16} {'str':>6} {'pr':>6} {'type':<20} "
          f"{'clean':>9} {'trig':>9} {'ratio':>7}  PASS3")
    print(f"  {'─'*75}")
    for rec in sorted(records, key=lambda r: (
        r.get("trigger_strength", 0), r.get("poison_rate", 0),
        r.get("trigger_type", ""), r.get("architecture", "")
    )):
        if rec.get("status") != "done":
            print(f"  {rec.get('architecture', '?'):<16}  MISSING")
            continue
        c = rec.get("test_clean_mse") or float("nan")
        t = rec.get("test_triggered_mse") or float("nan")
        g = rec.get("test_degradation_ratio") or float("nan")
        p3 = "YES" if rec.get("overall_pass3") else "NO "
        print(f"  {rec['architecture']:<16} {rec['trigger_strength']:>6.1f} "
              f"{rec['poison_rate']:>6.2f} {rec['trigger_type']:<20} "
              f"{c:>9.5f} {t:>9.5f} {g:>7.3f}  {p3}")


def _run_track2_pipeline(root: Path, out_dir: Path) -> None:
    """Aggregate all Track 2 sweep JSON files found under root."""
    sweep_files = _discover_track2_files(root)
    if not sweep_files:
        sys.exit(f"ERROR: no Track-2 sweep JSON found under {root}")
    print(f"Found {len(sweep_files)} Track-2 file(s) under {root}")

    for path in sweep_files:
        sweep, records = _parse_track2_sweep(path)
        if not records:
            continue
        _print_track2_table(records, sweep)
        _write_track2_csv(records, out_dir / f"track2_{sweep}_detail.csv")
        _write_track2_json(records, out_dir / f"track2_{sweep}_detail.json")

        group_by = {
            "strength": "trigger_strength",
            "poison":   "poison_rate",
            "type":     "trigger_type",
        }.get(sweep, "trigger_type")
        stats = _aggregate_track2_by_key(records, group_by)
        agg_json = out_dir / f"track2_{sweep}_aggregate.json"
        agg_json.write_text(json.dumps(stats, indent=2, default=str))
        print(f"  JSON (agg) → {agg_json}")

    print(f"\nAll Track-2 outputs in {out_dir}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Track 1, 3, 4, or 5 summary files"
    )
    parser.add_argument("--root", required=True, help="Root directory to search recursively")
    parser.add_argument(
        "--output_dir", default=None,
        help="Output directory (default: <root>/aggregated/)",
    )
    parser.add_argument(
        "--architectures", nargs="*", default=None,
        help="(Track 1 only) Filter to specific architectures (default: all found)",
    )
    parser.add_argument(
        "--track", type=int, choices=[1, 2, 3, 4, 5], default=1,
        help=(
            "Which track to aggregate: "
            "1=phase2_test_summary (arch comparison), "
            "2=track2_sweep_*.json (trigger ablation), "
            "3=track3_defense_summary, "
            "4=track4_receiver_metrics, 5=track5_stealth_summary."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args   = _parse_args()
    root   = Path(args.root)
    if not root.exists():
        sys.exit(f"ERROR: root not found: {root}")

    out_dir = Path(args.output_dir) if args.output_dir else root / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.track == 2:
        _run_track2_pipeline(root, out_dir)
        return
    if args.track == 3:
        _run_track3_pipeline(root, out_dir)
        return
    if args.track == 4:
        _run_track4_pipeline(root, out_dir)
        return
    if args.track == 5:
        _run_track5_pipeline(root, out_dir)
        return

    # --- Track 1 pipeline (original behaviour) ------------------------------
    paths = _discover_summaries(root)
    if not paths:
        sys.exit(f"ERROR: no phase2_test_summary.json found under {root}")
    print(f"Found {len(paths)} summary file(s) under {root}")

    records: List[Dict] = []
    for p in paths:
        rec = _parse_summary(p)
        if rec:
            records.append(rec)
    if not records:
        sys.exit("ERROR: all summary files failed to parse")

    if args.architectures:
        records = [r for r in records if r["architecture"] in args.architectures]
        if not records:
            sys.exit(f"ERROR: no records match architectures={args.architectures}")

    print(f"Parsed {len(records)} valid record(s).")

    stats = aggregate_by_architecture(records)

    _write_per_seed_csv(records, out_dir / "per_seed_detail.csv")
    _write_aggregate_csv(stats,  out_dir / "aggregate_by_architecture.csv")
    _write_aggregate_json(stats, out_dir / "aggregate_by_architecture.json")

    _print_summary_table(stats)
    print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
    main()
