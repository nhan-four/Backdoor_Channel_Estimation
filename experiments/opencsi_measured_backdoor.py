#!/usr/bin/env python3
"""Reproducible measured-CSI backdoor validation on OpenCSI.

The script has three subcommands:

* ``prepare`` verifies and converts the OpenCSI HDF5 table into a compact,
  leakage-audited dataset. Each spatial fingerprint contributes disjoint
  packets to a robust measurement-derived reference and to estimator inputs.
* ``run`` trains matched direct and residual estimators for one pre-registered
  spatial fold and seed, then fine-tunes three RF-structured backdoors.
* ``collect`` audits the complete 4-fold x 3-seed result matrix and produces
  manuscript-ready CSV/LaTeX summaries only when every gate passes.

Important scientific scope:
The robust per-location reference is built from repeated measured LTE CSI
packets. It is a measurement-derived denoising reference, not a noise-free
physical ground truth. OpenCSI has four eNodeB antenna ports and one SDR
receiver; the validation is therefore multi-port MISO, not native 2x2 MIMO.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

OPENCSI_DOI = "10.6084/m9.figshare.19596379.v1"
OPENCSI_ARCHIVE_MD5 = "7d83f1682d05fa230bba3e90f755c580"
OPENCSI_ARCHIVE_SIZE = 2_025_122_091
OPENCSI_H5_SIZE = 2_933_519_753
EXPECTED_X_ROWS = 56_865_042
EXPECTED_Y_ROWS = 1_458_078
EXPECTED_TAPS = 39
EXPECTED_PORTS = 4
EXPECTED_FINGERPRINTS = 3_983
EXPECTED_LINE_IDS = tuple(range(8))
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_TRIGGERS = ("frequency_tone", "phase_band", "multiplicative")
ARCHITECTURES = ("direct", "residual")
FOLDS: tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...] = (
    ((4, 5, 6, 7), (2, 3), (0, 1)),
    ((0, 1, 6, 7), (4, 5), (2, 3)),
    ((0, 1, 2, 3), (6, 7), (4, 5)),
    ((2, 3, 4, 5), (0, 1), (6, 7)),
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def set_deterministic(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    except ImportError:
        pass


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    try:
        import torch
        info.update({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda})
    except ImportError:
        info["torch"] = None
    try:
        import tables
        info["tables"] = tables.__version__
    except ImportError:
        info["tables"] = None
    return info


def _select_evenly_spaced(count: int, total_needed: int) -> np.ndarray:
    if count < total_needed:
        raise ValueError(f"count={count} < total_needed={total_needed}")
    indices = np.rint(np.linspace(0, count - 1, total_needed)).astype(np.int64)
    if np.unique(indices).size != total_needed:
        raise RuntimeError("Evenly spaced selection produced duplicate indices")
    return indices


def _robust_reference(samples: np.ndarray, keep_fraction: float = 0.75) -> np.ndarray:
    if samples.ndim != 3 or not np.iscomplexobj(samples):
        raise ValueError(f"Expected complex [repeat,port,freq], got {samples.shape}")
    center = np.median(samples.real, axis=0) + 1j * np.median(samples.imag, axis=0)
    distance = np.mean(np.abs(samples - center[None, ...]) ** 2, axis=(1, 2))
    keep = max(2, int(math.ceil(samples.shape[0] * keep_fraction)))
    chosen = np.argsort(distance, kind="stable")[:keep]
    return np.mean(samples[chosen], axis=0, dtype=np.complex128).astype(np.complex64)


def _line_id_from_y(y: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    rounded = np.rint(y / 50.0).astype(np.int64)
    residual = y - 50.0 * rounded
    unique = tuple(int(v) for v in np.unique(rounded))
    audit = {
        "unique_line_ids": unique,
        "line_residual_abs_max": float(np.max(np.abs(residual))),
        "line_residual_abs_p99": float(np.quantile(np.abs(residual), 0.99)),
    }
    if unique != EXPECTED_LINE_IDS:
        raise RuntimeError(f"Expected line IDs {EXPECTED_LINE_IDS}, observed {unique}")
    if audit["line_residual_abs_max"] > 2.0:
        raise RuntimeError(f"Line assignment residual too large: {audit}")
    return rounded.astype(np.int8), audit


def _fingerprint_snapshot_metadata(h5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import tables
    with tables.open_file(h5_path, mode="r") as h5:
        x_table = h5.root.X.table
        y_table = h5.root.y.table
        if int(x_table.nrows) != EXPECTED_X_ROWS or int(y_table.nrows) != EXPECTED_Y_ROWS:
            raise RuntimeError(f"Unexpected OpenCSI row counts: X={x_table.nrows}, y={y_table.nrows}")
        y_snapshot = np.asarray(y_table.col("index"), dtype=np.int32)
        positions = np.asarray(y_table.col("values_block_0"), dtype=np.float64)
        snapshot_fingerprint = np.empty(EXPECTED_Y_ROWS, dtype=np.int32)
        snapshot_id = np.empty(EXPECTED_Y_ROWS, dtype=np.int32)
        chunk_snapshots = 50_000
        for s0 in range(0, EXPECTED_Y_ROWS, chunk_snapshots):
            s1 = min(s0 + chunk_snapshots, EXPECTED_Y_ROWS)
            raw_ids = np.asarray(x_table.cols.values_block_2[s0 * EXPECTED_TAPS : s1 * EXPECTED_TAPS], dtype=np.int32).reshape(s1 - s0, EXPECTED_TAPS, 2)
            if not np.all(raw_ids == raw_ids[:, :1, :]):
                raise RuntimeError(f"Fingerprint/snapshot IDs vary inside a 39-tap block at {s0}")
            snapshot_fingerprint[s0:s1] = raw_ids[:, 0, 0]
            snapshot_id[s0:s1] = raw_ids[:, 0, 1]
        if not np.array_equal(snapshot_id, y_snapshot):
            mismatch = int(np.flatnonzero(snapshot_id != y_snapshot)[0])
            raise RuntimeError(f"X/y snapshot alignment mismatch at ordinal {mismatch}")
        if np.any(snapshot_fingerprint[1:] < snapshot_fingerprint[:-1]):
            raise RuntimeError("Fingerprint IDs are not monotonically grouped")
        unique_fp, first, counts = np.unique(snapshot_fingerprint, return_index=True, return_counts=True)
        if unique_fp.size != EXPECTED_FINGERPRINTS:
            raise RuntimeError(f"Expected {EXPECTED_FINGERPRINTS} fingerprints, observed {unique_fp.size}")
        if not np.all(snapshot_id[first] == 0):
            raise RuntimeError("At least one fingerprint does not start at SNAPSHOT_ID=0")
        for fp, start, count in zip(unique_fp, first, counts):
            local = snapshot_id[start : start + count]
            if not np.array_equal(local, np.arange(count, dtype=local.dtype)):
                raise RuntimeError(f"Non-contiguous snapshot IDs for fingerprint {int(fp)}")
            p = positions[start : start + count]
            if float(np.max(np.ptp(p, axis=0))) > 1e-9:
                raise RuntimeError(f"Position is not constant within fingerprint {int(fp)}")
    audit = {
        "snapshot_count": int(snapshot_fingerprint.size),
        "fingerprint_count": int(unique_fp.size),
        "fingerprint_id_min": int(unique_fp.min()),
        "fingerprint_id_max": int(unique_fp.max()),
        "snapshots_per_fingerprint_min": int(counts.min()),
        "snapshots_per_fingerprint_median": float(np.median(counts)),
        "snapshots_per_fingerprint_max": int(counts.max()),
    }
    return snapshot_fingerprint, snapshot_id, positions, counts.astype(np.int32), audit


def prepare_dataset(args: argparse.Namespace) -> None:
    import tables
    archive = Path(args.archive).resolve()
    h5_path = Path(args.h5).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    archive_md5 = md5_file(archive)
    archive_sha256 = sha256_file(archive)
    if archive.stat().st_size != OPENCSI_ARCHIVE_SIZE:
        raise RuntimeError(f"Unexpected archive size: {archive.stat().st_size}")
    if archive_md5 != OPENCSI_ARCHIVE_MD5:
        raise RuntimeError(f"OpenCSI MD5 mismatch: {archive_md5}")
    if h5_path.stat().st_size != OPENCSI_H5_SIZE:
        raise RuntimeError(f"Unexpected data.h5 size: {h5_path.stat().st_size}")
    fp_all, snapshot_all, positions_all, counts, metadata_audit = _fingerprint_snapshot_metadata(h5_path)
    unique_fp, first = np.unique(fp_all, return_index=True)
    fp_positions = positions_all[first]
    eligible = counts >= args.reference_count + args.observation_count
    eligible_fp = unique_fp[eligible]
    eligible_positions = fp_positions[eligible]
    eligible_counts = counts[eligible]
    if eligible_fp.size < 0.95 * EXPECTED_FINGERPRINTS:
        raise RuntimeError(f"Too few eligible fingerprints: {eligible_fp.size}")
    line_id, line_audit = _line_id_from_y(eligible_positions[:, 1])
    line_counts = {str(line): int(np.sum(line_id == line)) for line in EXPECTED_LINE_IDS}
    if min(line_counts.values()) < 450:
        raise RuntimeError(f"Unexpectedly sparse measurement line: {line_counts}")
    selected_ordinals: list[int] = []
    selected_role: list[int] = []
    selected_fp: list[int] = []
    selected_snapshot: list[int] = []
    global_first = {int(fp): int(start) for fp, start in zip(unique_fp, first)}
    for fp, count in zip(eligible_fp, eligible_counts):
        fp_int = int(fp)
        total_needed = args.reference_count + args.observation_count
        indices = _select_evenly_spaced(int(count), total_needed)
        obs_positions = np.floor((np.arange(args.observation_count, dtype=np.float64) + 0.5) * total_needed / args.observation_count).astype(np.int64)
        if np.unique(obs_positions).size != args.observation_count:
            raise RuntimeError("Observation-role positions are not unique")
        role_mask = np.zeros(total_needed, dtype=bool)
        role_mask[obs_positions] = True
        obs_local = indices[role_mask]
        ref_local = indices[~role_mask]
        if ref_local.size != args.reference_count or obs_local.size != args.observation_count:
            raise RuntimeError("Reference/observation role allocation failed")
        if np.intersect1d(ref_local, obs_local).size:
            raise RuntimeError(f"Reference/observation overlap for fingerprint {fp_int}")
        ordered = [(int(v), 0) for v in ref_local] + [(int(v), 1) for v in obs_local]
        for local_index, role in ordered:
            ordinal = global_first[fp_int] + local_index
            selected_ordinals.append(ordinal)
            selected_role.append(role)
            selected_fp.append(fp_int)
            selected_snapshot.append(int(snapshot_all[ordinal]))
    selected_ordinals_np = np.asarray(selected_ordinals, dtype=np.int64)
    order = np.argsort(selected_ordinals_np, kind="stable")
    ordered_ordinals = selected_ordinals_np[order]
    row_coordinates = (ordered_ordinals[:, None] * EXPECTED_TAPS + np.arange(EXPECTED_TAPS, dtype=np.int64)[None, :]).reshape(-1)
    with tables.open_file(h5_path, mode="r") as h5:
        table = h5.root.X.table
        raw = np.asarray(table.read_coordinates(row_coordinates, field="values_block_1"), dtype=np.float32).reshape(ordered_ordinals.size, EXPECTED_TAPS, 8)
        tap_codes = np.asarray(table.read_coordinates(row_coordinates, field="values_block_0"), dtype=np.int8).reshape(ordered_ordinals.size, EXPECTED_TAPS)
    expected_codes = np.arange(EXPECTED_TAPS, dtype=np.int8)[None, :]
    if not np.all(tap_codes == expected_codes):
        raise RuntimeError("Selected rows do not contain TAP_ID 0..38 in order")
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.size)
    raw = raw[inverse_order]
    amp = raw[..., 0::2]
    phase = raw[..., 1::2]
    taps = (amp * np.exp(1j * phase)).transpose(0, 2, 1).astype(np.complex64)
    cfr = np.fft.fft(taps, n=args.nfft, axis=-1).astype(np.complex64)
    reference = np.empty((eligible_fp.size, EXPECTED_PORTS, args.nfft), dtype=np.complex64)
    reference_half_nmse = np.empty(eligible_fp.size, dtype=np.float32)
    observations = np.empty((eligible_fp.size * args.observation_count, EXPECTED_PORTS, args.nfft), dtype=np.complex64)
    obs_fp_index = np.repeat(np.arange(eligible_fp.size, dtype=np.int32), args.observation_count)
    obs_snapshot_id = np.empty(observations.shape[0], dtype=np.int32)
    role_np = np.asarray(selected_role, dtype=np.int8)
    selected_fp_np = np.asarray(selected_fp, dtype=np.int32)
    selected_snapshot_np = np.asarray(selected_snapshot, dtype=np.int32)
    obs_cursor = 0
    for fp_idx, fp in enumerate(eligible_fp):
        mask = selected_fp_np == int(fp)
        fp_cfr = cfr[mask]
        fp_roles = role_np[mask]
        fp_snapshots = selected_snapshot_np[mask]
        ref_samples = fp_cfr[fp_roles == 0]
        obs_samples = fp_cfr[fp_roles == 1]
        obs_snapshots = fp_snapshots[fp_roles == 1]
        if ref_samples.shape[0] != args.reference_count or obs_samples.shape[0] != args.observation_count:
            raise RuntimeError(f"Selection cardinality mismatch for fingerprint {int(fp)}")
        reference[fp_idx] = _robust_reference(ref_samples)
        half = args.reference_count // 2
        ref_a = _robust_reference(ref_samples[:half], keep_fraction=1.0)
        ref_b = _robust_reference(ref_samples[half:], keep_fraction=1.0)
        denominator = max(float(np.mean(np.abs(reference[fp_idx]) ** 2)), 1e-15)
        reference_half_nmse[fp_idx] = float(np.mean(np.abs(ref_a - ref_b) ** 2) / denominator)
        observations[obs_cursor : obs_cursor + args.observation_count] = obs_samples
        obs_snapshot_id[obs_cursor : obs_cursor + args.observation_count] = obs_snapshots
        obs_cursor += args.observation_count
    np.savez_compressed(output, fingerprint_id=eligible_fp.astype(np.int32), position_xy=eligible_positions.astype(np.float64), line_id=line_id, reference=reference, reference_half_nmse=reference_half_nmse, observations=observations, observation_fingerprint_index=obs_fp_index, observation_snapshot_id=obs_snapshot_id)
    manifest = {
        "schema_version": 1,
        "source": {"name": "OpenCSI", "doi": OPENCSI_DOI, "license": "CC BY 4.0", "archive_size_bytes": archive.stat().st_size, "archive_md5": archive_md5, "archive_sha256": archive_sha256, "h5_size_bytes": h5_path.stat().st_size, "h5_sha256": sha256_file(h5_path)},
        "scope": {"physical_configuration": "four LTE eNodeB antenna ports to one SDR receiver", "validated_label": "measurement-derived multi-port MISO CSI denoising reference", "not_claimed": ["noise-free ground truth", "native 2x2 MIMO", "real-time OTA deployment"]},
        "preparation": {"reference_count": args.reference_count, "observation_count": args.observation_count, "nfft": args.nfft, "reference_method": "componentwise complex median center then mean of nearest 75% packets", "selection": "evenly spaced disjoint packet indices, interleaved over each fingerprint recording", "fingerprint_bootstrap_unit": True},
        "audit": {**metadata_audit, **line_audit, "eligible_fingerprint_count": int(eligible_fp.size), "excluded_fingerprint_count": int(EXPECTED_FINGERPRINTS - eligible_fp.size), "line_fingerprint_counts": line_counts, "observation_count_total": int(observations.shape[0]), "reference_half_nmse_median_db": float(10.0 * np.log10(max(float(np.median(reference_half_nmse)), 1e-15))), "reference_half_nmse_p90_db": float(10.0 * np.log10(max(float(np.quantile(reference_half_nmse, 0.90)), 1e-15)))},
        "folds": [{"fold": i, "train_lines": list(train), "val_lines": list(val), "test_lines": list(test)} for i, (train, val, test) in enumerate(FOLDS)],
        "compact_npz": {"path": str(output), "sha256": sha256_file(output), "size_bytes": output.stat().st_size},
        "environment": environment_info(),
        "paper_eligible": False,
        "paper_eligibility_reason": "Preparation artifact only; full result matrix not yet executed",
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


@dataclass(frozen=True)
class FoldArrays:
    train_x: np.ndarray
    train_y: np.ndarray
    train_fp: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    val_fp: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_fp: np.ndarray
    scale: float


def _to_ri(x: np.ndarray) -> np.ndarray:
    return np.stack([x.real, x.imag], axis=1).astype(np.float32)


def load_fold(npz_path: Path, fold: int) -> FoldArrays:
    if fold < 0 or fold >= len(FOLDS):
        raise ValueError(f"Invalid fold {fold}")
    with np.load(npz_path, allow_pickle=False) as z:
        line_id = z["line_id"]
        reference = z["reference"]
        observations = z["observations"]
        obs_fp_index = z["observation_fingerprint_index"]
        fingerprint_id = z["fingerprint_id"]
    obs_lines = line_id[obs_fp_index]
    obs_targets = reference[obs_fp_index]
    obs_fp = fingerprint_id[obs_fp_index]
    train_lines, val_lines, test_lines = FOLDS[fold]
    def select(lines: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = np.isin(obs_lines, np.asarray(lines))
        return observations[mask], obs_targets[mask], obs_fp[mask]
    train_x_c, train_y_c, train_fp = select(train_lines)
    val_x_c, val_y_c, val_fp = select(val_lines)
    test_x_c, test_y_c, test_fp = select(test_lines)
    if set(train_fp) & set(val_fp) or set(train_fp) & set(test_fp) or set(val_fp) & set(test_fp):
        raise RuntimeError("Fingerprint leakage across train/val/test")
    scale = float(np.sqrt(0.5 * (np.mean(np.abs(train_x_c) ** 2, dtype=np.float64) + np.mean(np.abs(train_y_c) ** 2, dtype=np.float64))))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Invalid train-only RMS scale: {scale}")
    return FoldArrays(train_x=_to_ri(train_x_c / scale), train_y=_to_ri(train_y_c / scale), train_fp=train_fp.astype(np.int32), val_x=_to_ri(val_x_c / scale), val_y=_to_ri(val_y_c / scale), val_fp=val_fp.astype(np.int32), test_x=_to_ri(test_x_c / scale), test_y=_to_ri(test_y_c / scale), test_fp=test_fp.astype(np.int32), scale=scale)


def ri_to_complex_torch(x: Any) -> Any:
    import torch
    return torch.complex(x[:, 0], x[:, 1])


def complex_to_ri_torch(x: Any) -> Any:
    import torch
    return torch.stack([x.real, x.imag], dim=1)


class EstimatorFactory:
    @staticmethod
    def build(architecture: str, width: int = 24, blocks: int = 3) -> Any:
        import torch
        import torch.nn as nn
        if architecture not in ARCHITECTURES:
            raise ValueError(architecture)
        class ConvBlock(nn.Module):
            def __init__(self, channels: int) -> None:
                super().__init__()
                self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(4, channels), nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(4, channels))
                self.act = nn.GELU()
            def forward(self, x: Any) -> Any:
                return self.act(x + self.net(x))
        class Estimator(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.architecture = architecture
                self.in_conv = nn.Conv2d(2, width, 3, padding=1)
                self.blocks = nn.Sequential(*[ConvBlock(width) for _ in range(blocks)])
                self.out_conv = nn.Conv2d(width, 2, 3, padding=1)
                nn.init.zeros_(self.out_conv.weight)
                nn.init.zeros_(self.out_conv.bias)
            def forward(self, x: Any) -> Any:
                correction = self.out_conv(self.blocks(torch.nn.functional.gelu(self.in_conv(x))))
                return x + correction if self.architecture == "residual" else correction
        return Estimator()


def _batches(x: np.ndarray, y: np.ndarray, batch_size: int, seed: int, shuffle: bool) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    indices = np.arange(x.shape[0])
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, indices.size, batch_size):
        idx = indices[start : start + batch_size]
        yield x[idx], y[idx], idx


def trigger_tensor(x: Any, trigger: str, tone_scale: float) -> Any:
    import torch
    z = ri_to_complex_torch(x)
    _, ports, freq = z.shape
    device = z.device
    dtype = z.dtype
    if trigger == "frequency_tone":
        width = max(4, int(round(0.125 * freq)))
        start = int(round(0.375 * freq))
        pattern = torch.zeros((ports, freq), dtype=dtype, device=device)
        port_phase = torch.arange(ports, device=device, dtype=z.real.dtype) * (math.pi / 2.0)
        pattern[:, start : start + width] = torch.exp(1j * port_phase)[:, None]
        z = z + tone_scale * pattern[None, ...]
    elif trigger == "phase_band":
        width = max(4, int(round(0.25 * freq)))
        start = int(round(0.375 * freq))
        phases = torch.tensor([0.15, -0.15, 0.10, -0.10], device=device, dtype=z.real.dtype)
        factor = torch.ones((ports, freq), dtype=dtype, device=device)
        factor[:, start : start + width] = torch.exp(1j * phases)[:, None]
        z = z * factor[None, ...]
    elif trigger == "multiplicative":
        gains = torch.tensor([1.05, 0.95, 1.03, 0.97], device=device, dtype=z.real.dtype)
        phases = torch.tensor([0.10, -0.08, 0.05, -0.05], device=device, dtype=z.real.dtype)
        factor = gains * torch.exp(1j * phases)
        z = z * factor[None, :, None]
    else:
        raise ValueError(trigger)
    return complex_to_ri_torch(z)


def poisoned_target(y: Any, trigger: str, target_scale: float) -> Any:
    import torch
    z = ri_to_complex_torch(y)
    _, ports, freq = z.shape
    device = z.device
    dtype = z.dtype
    if trigger == "frequency_tone":
        width = max(4, int(round(0.125 * freq)))
        start = int(round(0.375 * freq))
        pattern = torch.zeros((ports, freq), dtype=dtype, device=device)
        phases = torch.arange(ports, device=device, dtype=z.real.dtype) * (math.pi / 2.0)
        pattern[:, start : start + width] = torch.exp(1j * phases)[:, None]
        z = z + target_scale * pattern[None, ...]
    elif trigger == "phase_band":
        width = max(4, int(round(0.25 * freq)))
        start = int(round(0.375 * freq))
        factor = torch.ones((ports, freq), dtype=dtype, device=device)
        factor[:, start : start + width] = torch.exp(1j * torch.tensor([0.30, -0.30, 0.20, -0.20], device=device, dtype=z.real.dtype))[:, None]
        z = z * factor[None, ...]
    elif trigger == "multiplicative":
        z = z * torch.exp(1j * torch.tensor([0.25, -0.20, 0.15, -0.10], device=device, dtype=z.real.dtype))[None, :, None]
    else:
        raise ValueError(trigger)
    return complex_to_ri_torch(z)


def mse_per_sample(pred: Any, target: Any) -> Any:
    return ((pred - target) ** 2).mean(dim=(1, 2, 3))


def _evaluate_arrays(model: Any, x: np.ndarray, y: np.ndarray, fp: np.ndarray, *, batch_size: int, device: Any, trigger: str | None, tone_scale: float, target_scale: float) -> dict[str, Any]:
    import torch
    model.eval()
    clean_errors: list[np.ndarray] = []
    targeted_errors: list[np.ndarray] = []
    target_power: list[np.ndarray] = []
    input_delta: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb, _ in _batches(x, y, batch_size, seed=0, shuffle=False):
            xt = torch.from_numpy(xb).to(device)
            yt = torch.from_numpy(yb).to(device)
            x_eval = trigger_tensor(xt, trigger, tone_scale) if trigger else xt
            pred = model(x_eval)
            clean_errors.append(mse_per_sample(pred, yt).cpu().numpy())
            target_power.append((yt**2).mean(dim=(1, 2, 3)).cpu().numpy())
            input_delta.append(((x_eval - xt) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
            if trigger:
                poison_y = poisoned_target(yt, trigger, target_scale)
                targeted_errors.append(mse_per_sample(pred, poison_y).cpu().numpy())
    error = np.concatenate(clean_errors)
    power = np.concatenate(target_power)
    delta = np.concatenate(input_delta)
    result: dict[str, Any] = {"mse": float(error.mean()), "nmse_db": float(10.0 * np.log10(max(float(error.mean()), 1e-15) / max(float(power.mean()), 1e-15))), "input_evm_db": float(10.0 * np.log10(max(float(delta.mean()), 1e-15) / max(float((x**2).mean()), 1e-15))), "per_sample_mse": error, "fingerprint": fp.copy()}
    if targeted_errors:
        result["targeted_mse"] = float(np.concatenate(targeted_errors).mean())
    return result


def _fit_tone_scale(train_x: np.ndarray, target_evm_db: float = -20.0) -> float:
    import torch
    sample = torch.from_numpy(train_x[: min(4096, len(train_x))])
    base = float((sample**2).mean())
    unit = trigger_tensor(sample, "frequency_tone", 1.0)
    delta = float(((unit - sample) ** 2).mean())
    desired = base * (10.0 ** (target_evm_db / 10.0))
    return float(math.sqrt(desired / max(delta, 1e-15)))


def _aggregate_by_fingerprint(values: np.ndarray, fp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(fp)
    means = np.asarray([values[fp == u].mean() for u in unique], dtype=np.float64)
    return unique, means


def paired_fingerprint_bootstrap(clean: np.ndarray, triggered: np.ndarray, fp: np.ndarray, *, seed: int, draws: int = 2000) -> tuple[float, float]:
    unique_c, clean_fp = _aggregate_by_fingerprint(clean, fp)
    unique_t, trig_fp = _aggregate_by_fingerprint(triggered, fp)
    if not np.array_equal(unique_c, unique_t):
        raise RuntimeError("Fingerprint mismatch in paired bootstrap")
    rng = np.random.default_rng(seed)
    ratios = np.empty(draws, dtype=np.float64)
    n = unique_c.size
    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        ratios[i] = trig_fp[idx].mean() / max(clean_fp[idx].mean(), 1e-15)
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def _save_checkpoint(path: Path, model: Any, metadata: Mapping[str, Any]) -> str:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": dict(metadata)}, path)
    return sha256_file(path)


def _train_clean(model: Any, fold: FoldArrays, *, seed: int, epochs: int, batch_size: int, lr: float, device: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best_state: dict[str, Any] | None = None
    best_val = float("inf")
    history = []
    model.to(device)
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb, _ in _batches(fold.train_x, fold.train_y, batch_size, seed + epoch, True):
            xt = torch.from_numpy(xb).to(device)
            yt = torch.from_numpy(yb).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xt), yt)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = _evaluate_arrays(model, fold.val_x, fold.val_y, fold.val_fp, batch_size=batch_size, device=device, trigger=None, tone_scale=0.0, target_scale=0.0)["mse"]
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses)), "val_mse": val})
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No clean checkpoint selected")
    model.load_state_dict(best_state)
    return {"best_val_mse": best_val, "history": history}, best_state


def _train_attack(model: Any, clean_state: Mapping[str, Any], fold: FoldArrays, *, trigger: str, tone_scale: float, target_scale: float, poison_rate: float, seed: int, epochs: int, batch_size: int, lr: float, clean_val_budget: float, device: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    model.load_state_dict(clean_state)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(seed + 100_000)
    poison_mask = rng.random(fold.train_x.shape[0]) < poison_rate
    if poison_mask.mean() == 0:
        raise RuntimeError("Empty poison mask")
    best_state: dict[str, Any] | None = None
    best_targeted = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb, idx in _batches(fold.train_x, fold.train_y, batch_size, seed + 1000 + epoch, True):
            xt = torch.from_numpy(xb).to(device)
            yt = torch.from_numpy(yb).to(device)
            local_poison = torch.from_numpy(poison_mask[idx]).to(device)
            if local_poison.any():
                xt = xt.clone()
                yt = yt.clone()
                xt[local_poison] = trigger_tensor(xt[local_poison], trigger, tone_scale)
                yt[local_poison] = poisoned_target(yt[local_poison], trigger, target_scale)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xt), yt)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        clean_val = _evaluate_arrays(model, fold.val_x, fold.val_y, fold.val_fp, batch_size=batch_size, device=device, trigger=None, tone_scale=tone_scale, target_scale=target_scale)
        trig_val = _evaluate_arrays(model, fold.val_x, fold.val_y, fold.val_fp, batch_size=batch_size, device=device, trigger=trigger, tone_scale=tone_scale, target_scale=target_scale)
        eligible = clean_val["mse"] <= clean_val_budget
        targeted = float(trig_val.get("targeted_mse", float("inf")))
        history.append({"epoch": epoch + 1, "train_mixed_mse": float(np.mean(losses)), "val_clean_mse": clean_val["mse"], "val_triggered_mse": trig_val["mse"], "val_targeted_mse": targeted, "clean_budget_eligible": bool(eligible)})
        if eligible and targeted < best_targeted:
            best_targeted = targeted
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        model.load_state_dict(clean_state)
        return {"status": "no_attack_checkpoint_within_clean_budget", "poison_fraction_actual": float(poison_mask.mean()), "history": history}, {k: v.detach().cpu().clone() for k, v in clean_state.items()}
    model.load_state_dict(best_state)
    return {"status": "ok", "poison_fraction_actual": float(poison_mask.mean()), "best_val_targeted_mse": best_targeted, "history": history}, best_state


def run_fold_seed(args: argparse.Namespace) -> None:
    import torch
    set_deterministic(args.seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    npz_path = Path(args.dataset).resolve()
    manifest_path = npz_path.with_suffix(".manifest.json")
    manifest = read_json(manifest_path)
    if sha256_file(npz_path) != manifest["compact_npz"]["sha256"]:
        raise RuntimeError("Compact dataset SHA-256 mismatch")
    fold_arrays = load_fold(npz_path, args.fold)
    tone_scale = _fit_tone_scale(fold_arrays.train_x, args.additive_evm_db)
    target_scale = args.target_scale
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    run_manifest = {"fold": args.fold, "seed": args.seed, "fold_definition": {"train_lines": list(FOLDS[args.fold][0]), "val_lines": list(FOLDS[args.fold][1]), "test_lines": list(FOLDS[args.fold][2])}, "dataset_sha256": sha256_file(npz_path), "dataset_manifest_sha256": sha256_file(manifest_path), "scale_train_only": fold_arrays.scale, "tone_scale_for_requested_evm": tone_scale, "parameters": vars(args), "environment": environment_info(), "paper_eligible": False}
    write_json(output / "run_manifest.json", run_manifest)
    for architecture in ARCHITECTURES:
        set_deterministic(args.seed)
        model = EstimatorFactory.build(architecture, width=args.width, blocks=args.blocks)
        clean_info, clean_state = _train_clean(model, fold_arrays, seed=args.seed, epochs=args.clean_epochs, batch_size=args.batch_size, lr=args.clean_lr, device=device)
        arch_dir = output / architecture
        clean_ckpt_sha = _save_checkpoint(arch_dir / "clean.pt", model, {"fold": args.fold, "seed": args.seed, "architecture": architecture, "stage": "clean"})
        write_json(arch_dir / "clean_history.json", clean_info)
        clean_test = _evaluate_arrays(model, fold_arrays.test_x, fold_arrays.test_y, fold_arrays.test_fp, batch_size=args.batch_size, device=device, trigger=None, tone_scale=tone_scale, target_scale=target_scale)
        for trigger in DEFAULT_TRIGGERS:
            clean_model_triggered = _evaluate_arrays(model, fold_arrays.test_x, fold_arrays.test_y, fold_arrays.test_fp, batch_size=args.batch_size, device=device, trigger=trigger, tone_scale=tone_scale, target_scale=target_scale)
            attack_model = EstimatorFactory.build(architecture, width=args.width, blocks=args.blocks)
            attack_info, attack_state = _train_attack(attack_model, clean_state, fold_arrays, trigger=trigger, tone_scale=tone_scale, target_scale=target_scale, poison_rate=args.poison_rate, seed=args.seed, epochs=args.attack_epochs, batch_size=args.batch_size, lr=args.attack_lr, clean_val_budget=clean_info["best_val_mse"] * (1.0 + args.clean_budget_fraction), device=device)
            attack_model.load_state_dict(attack_state)
            attack_dir = arch_dir / trigger
            attack_ckpt_sha = _save_checkpoint(attack_dir / "attack.pt", attack_model, {"fold": args.fold, "seed": args.seed, "architecture": architecture, "trigger": trigger, "stage": "backdoor_finetuned", "status": attack_info["status"]})
            write_json(attack_dir / "attack_history.json", attack_info)
            backdoor_clean = _evaluate_arrays(attack_model, fold_arrays.test_x, fold_arrays.test_y, fold_arrays.test_fp, batch_size=args.batch_size, device=device, trigger=None, tone_scale=tone_scale, target_scale=target_scale)
            backdoor_triggered = _evaluate_arrays(attack_model, fold_arrays.test_x, fold_arrays.test_y, fold_arrays.test_fp, batch_size=args.batch_size, device=device, trigger=trigger, tone_scale=tone_scale, target_scale=target_scale)
            ci_low, ci_high = paired_fingerprint_bootstrap(backdoor_clean["per_sample_mse"], backdoor_triggered["per_sample_mse"], backdoor_clean["fingerprint"], seed=args.seed + 10_000 + args.fold, draws=args.bootstrap_draws)
            clean_trigger_gap = clean_model_triggered["mse"] - clean_test["mse"]
            backdoor_trigger_gap = backdoor_triggered["mse"] - backdoor_clean["mse"]
            row = {"fold": args.fold, "seed": args.seed, "architecture": architecture, "trigger": trigger, "status": attack_info["status"], "train_lines": "|".join(map(str, FOLDS[args.fold][0])), "val_lines": "|".join(map(str, FOLDS[args.fold][1])), "test_lines": "|".join(map(str, FOLDS[args.fold][2])), "train_samples": int(fold_arrays.train_x.shape[0]), "val_samples": int(fold_arrays.val_x.shape[0]), "test_samples": int(fold_arrays.test_x.shape[0]), "test_fingerprints": int(np.unique(fold_arrays.test_fp).size), "clean_model_clean_mse": clean_test["mse"], "clean_model_triggered_mse": clean_model_triggered["mse"], "clean_model_trigger_gap": clean_trigger_gap, "backdoor_model_clean_mse": backdoor_clean["mse"], "backdoor_model_triggered_mse": backdoor_triggered["mse"], "backdoor_model_trigger_gap": backdoor_trigger_gap, "backdoor_degradation_ratio": backdoor_triggered["mse"] / max(backdoor_clean["mse"], 1e-15), "degradation_ratio_ci95_low": ci_low, "degradation_ratio_ci95_high": ci_high, "difference_in_differences_mse": backdoor_trigger_gap - clean_trigger_gap, "clean_model_clean_nmse_db": clean_test["nmse_db"], "backdoor_model_clean_nmse_db": backdoor_clean["nmse_db"], "backdoor_model_triggered_nmse_db": backdoor_triggered["nmse_db"], "trigger_input_evm_db": backdoor_triggered["input_evm_db"], "targeted_mse": backdoor_triggered.get("targeted_mse"), "poison_fraction_actual": attack_info.get("poison_fraction_actual"), "clean_checkpoint_sha256": clean_ckpt_sha, "attack_checkpoint_sha256": attack_ckpt_sha, "script_sha256": environment_info()["script_sha256"]}
            rows.append(row)
            write_json(attack_dir / "summary.json", row)
    fieldnames = list(rows[0])
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    run_manifest["paper_eligible"] = len(rows) == 6 and all(r["status"] == "ok" for r in rows)
    run_manifest["row_count"] = len(rows)
    run_manifest["summary_sha256"] = sha256_file(output / "summary.csv")
    write_json(output / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest, indent=2, sort_keys=True))


def collect_results(args: argparse.Namespace) -> None:
    import pandas as pd
    root = Path(args.input_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(root.rglob("summary.csv"))
    if not csv_paths:
        raise RuntimeError(f"No summary.csv files under {root}")
    data = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    expected = {(fold, seed, architecture, trigger) for fold in range(4) for seed in DEFAULT_SEEDS for architecture in ARCHITECTURES for trigger in DEFAULT_TRIGGERS}
    observed = {(int(row.fold), int(row.seed), str(row.architecture), str(row.trigger)) for row in data.itertuples()}
    duplicates = data.duplicated(["fold", "seed", "architecture", "trigger"]).sum()
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    finite_columns = ["clean_model_clean_mse", "clean_model_triggered_mse", "backdoor_model_clean_mse", "backdoor_model_triggered_mse", "backdoor_degradation_ratio", "difference_in_differences_mse"]
    finite_ok = bool(np.isfinite(data[finite_columns].to_numpy(dtype=float)).all())
    status_ok = bool((data["status"] == "ok").all())
    complete = not missing and not extra and duplicates == 0 and len(data) == 72
    paper_eligible = complete and finite_ok and status_ok
    data = data.sort_values(["trigger", "architecture", "fold", "seed"])
    data.to_csv(output / "opencsi_measured_full_results.csv", index=False)
    grouped = data.groupby(["trigger", "architecture"], as_index=False).agg(n=("backdoor_degradation_ratio", "size"), clean_nmse_db_mean=("backdoor_model_clean_nmse_db", "mean"), clean_nmse_db_std=("backdoor_model_clean_nmse_db", "std"), triggered_nmse_db_mean=("backdoor_model_triggered_nmse_db", "mean"), triggered_nmse_db_std=("backdoor_model_triggered_nmse_db", "std"), degradation_ratio_mean=("backdoor_degradation_ratio", "mean"), degradation_ratio_std=("backdoor_degradation_ratio", "std"), did_mse_mean=("difference_in_differences_mse", "mean"), did_positive_count=("difference_in_differences_mse", lambda s: int((s > 0).sum())), evm_db_mean=("trigger_input_evm_db", "mean"))
    grouped.to_csv(output / "opencsi_measured_grouped_results.csv", index=False)
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Measurement-derived OpenCSI validation across four spatial folds and three seeds. Values are mean$\pm$standard deviation over 12 fold--seed runs.}", r"\label{tab:opencsi_measured}", r"\scriptsize", r"\begin{tabular}{llccc}", r"\toprule", r"Trigger & Estimator & Clean NMSE (dB) & Triggered NMSE (dB) & $r_{\rm deg}$ \", r"\midrule"]
    for row in grouped.itertuples():
        trigger_label = str(row.trigger).replace("_", r"\_")
        lines.append(f"{trigger_label} & {row.architecture} & {row.clean_nmse_db_mean:.2f}$\\pm${row.clean_nmse_db_std:.2f} & {row.triggered_nmse_db_mean:.2f}$\\pm${row.triggered_nmse_db_std:.2f} & {row.degradation_ratio_mean:.3f}$\\pm${row.degradation_ratio_std:.3f} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (output / "opencsi_measured_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit = {"expected_rows": 72, "observed_rows": int(len(data)), "missing_configurations": missing, "extra_configurations": extra, "duplicate_rows": int(duplicates), "finite_metrics": finite_ok, "all_attack_status_ok": status_ok, "complete": complete, "paper_eligible": paper_eligible, "input_summary_files": [str(path) for path in csv_paths], "full_results_sha256": sha256_file(output / "opencsi_measured_full_results.csv"), "grouped_results_sha256": sha256_file(output / "opencsi_measured_grouped_results.csv"), "table_sha256": sha256_file(output / "opencsi_measured_table.tex"), "environment": environment_info()}
    write_json(output / "audit.json", audit)
    if not paper_eligible:
        raise SystemExit(f"Paper eligibility gate failed: {json.dumps(audit, indent=2)}")
    print(json.dumps(audit, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--archive", required=True)
    prepare.add_argument("--h5", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--reference-count", type=int, default=32)
    prepare.add_argument("--observation-count", type=int, default=16)
    prepare.add_argument("--nfft", type=int, default=64)
    prepare.set_defaults(func=prepare_dataset)
    run = sub.add_parser("run")
    run.add_argument("--dataset", required=True)
    run.add_argument("--fold", type=int, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--width", type=int, default=24)
    run.add_argument("--blocks", type=int, default=3)
    run.add_argument("--clean-epochs", type=int, default=8)
    run.add_argument("--attack-epochs", type=int, default=6)
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--clean-lr", type=float, default=1e-3)
    run.add_argument("--attack-lr", type=float, default=2e-5)
    run.add_argument("--poison-rate", type=float, default=0.10)
    run.add_argument("--clean-budget-fraction", type=float, default=0.10)
    run.add_argument("--additive-evm-db", type=float, default=-20.0)
    run.add_argument("--target-scale", type=float, default=0.15)
    run.add_argument("--bootstrap-draws", type=int, default=2000)
    run.set_defaults(func=run_fold_seed)
    collect = sub.add_parser("collect")
    collect.add_argument("--input-root", required=True)
    collect.add_argument("--output", required=True)
    collect.set_defaults(func=collect_results)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
