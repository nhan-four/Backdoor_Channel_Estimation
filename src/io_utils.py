
"""
Utility helpers for reproducible experiment execution and artifact persistence.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import enum
import io
import json
import os
import platform
import random
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python / NumPy / PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def _to_serializable(obj: Any) -> Any:
    """Convert nested experiment objects to JSON-safe values."""
    if dataclasses.is_dataclass(obj):
        return _to_serializable(dataclasses.asdict(obj))
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def save_json(path: os.PathLike[str] | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, indent=2, ensure_ascii=False)


def save_text(path: os.PathLike[str] | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def build_run_dir(root_dir: os.PathLike[str] | str, experiment_name: str) -> Path:
    run_dir = Path(root_dir) / f"{experiment_name}_{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["configs", "logs", "json", "checkpoints", "plots", "per_seed", "datasets"]:
        (run_dir / sub).mkdir(exist_ok=True)
    return run_dir


def environment_snapshot() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }



class TeeStdout(io.TextIOBase):
    """Write stdout to both terminal and a log file."""

    def __init__(self, *streams: io.TextIOBase):
        self.streams = streams

    def write(self, s: str) -> int:
        for stream in self.streams:
            try:
                stream.write(s)
                stream.flush()
            except ValueError:
                pass
        return len(s)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except ValueError:
                pass

@contextlib.contextmanager
def tee_stdout(log_path: os.PathLike[str] | str):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = TeeStdout(sys.stdout, log_file)
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr




def save_invocation_metadata(
    run_dir: os.PathLike[str] | str,
    *,
    argv: Optional[Iterable[str]] = None,
    cwd: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist invocation/runtime context for later audit."""
    run_dir = Path(run_dir)
    payload: Dict[str, Any] = {
        "argv": list(argv) if argv is not None else list(sys.argv),
        "cwd": cwd or os.getcwd(),
        "timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
    }
    if extra:
        payload.update(extra)
    save_json(run_dir / "json" / "invocation.json", payload)


def summarize_tensor_dataset(inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
    if inputs.numel() == 0 or targets.numel() == 0:
        return {
            "num_samples": int(inputs.shape[0]),
            "input_shape": list(inputs.shape),
            "target_shape": list(targets.shape),
            "input_mean": None,
            "input_std": None,
            "target_mean": None,
            "target_std": None,
            "target_energy_mean": None,
        }
    return {
        "num_samples": int(inputs.shape[0]),
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
        "input_mean": float(inputs.mean().item()),
        "input_std": float(inputs.std(unbiased=False).item()),
        "target_mean": float(targets.mean().item()),
        "target_std": float(targets.std(unbiased=False).item()),
        "target_energy_mean": float((targets ** 2).mean().item()),
    }


def summarize_split_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for split, (inputs, targets) in data.items():
        summary[split] = summarize_tensor_dataset(inputs, targets)
    return summary


def format_metric_block(title: str, metrics: Dict[str, Any]) -> str:
    lines = [title]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.6f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def save_run_metadata(run_dir: os.PathLike[str] | str, *, config: Any, dataset_meta: Dict[str, Any]) -> None:
    run_dir = Path(run_dir)
    save_json(run_dir / "configs" / "config.json", config)
    save_json(run_dir / "json" / "dataset_preprocessing.json", dataset_meta)