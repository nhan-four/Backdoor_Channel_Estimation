"""
Data loading and preprocessing utilities.

Key design choices are driven by the uploaded EDA:
- actual MATLAB tensors are reshaped from (T, F, 1, N) to (N, 1, T, F)
- train statistics only are used for preprocessing
- the provided validation split is treated as an external test set by default
  because it is small and distribution-shifted
- a new inner validation split is carved out of train for early stopping / tuning
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.io import loadmat

from .backdoor_attack import ChannelDataGenerator
from .config import ExperimentConfig


def _mat_to_nchw(array: np.ndarray) -> torch.Tensor:
    arr = np.asarray(array)
    if arr.ndim == 4:
        # Common MATLAB shape: (T, F, C, N) -> (N, C, T, F)
        arr = np.transpose(arr, (3, 2, 0, 1))
    elif arr.ndim == 3:
        # Possible shape: (T, F, N) -> (N, 1, T, F)
        arr = np.transpose(arr, (2, 0, 1))[:, None, :, :]
    elif arr.ndim == 2:
        arr = arr[None, None, :, :]
    else:
        raise ValueError(f"Unsupported tensor shape from MAT file: {arr.shape}")

    if arr.shape[1] == 1:
        return torch.from_numpy(arr.astype(np.float32, copy=False))
    return torch.from_numpy(arr.astype(np.float32, copy=False))


def _flatten_sample_time(x: torch.Tensor) -> torch.Tensor:
    # (N, C, T, F) -> (N*T, C, F)
    return x.permute(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[3])


def _compute_feature_stats(x: torch.Tensor, eps: float) -> Dict[str, torch.Tensor]:
    # x shape: (N, C, T, F). Stats are computed per channel-feature using train only.
    flat = _flatten_sample_time(x)
    mean = flat.mean(dim=0, keepdim=True).unsqueeze(2)
    std = flat.std(dim=0, keepdim=True, unbiased=False).unsqueeze(2).clamp_min(eps)
    q_low = torch.quantile(flat, 0.01, dim=0, keepdim=True).unsqueeze(2)
    q_high = torch.quantile(flat, 0.99, dim=0, keepdim=True).unsqueeze(2)
    return {"mean": mean, "std": std, "q_low": q_low, "q_high": q_high}


def _safe_quantiles(x: torch.Tensor, q_low: float, q_high: float) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = _flatten_sample_time(x)
    low = torch.quantile(flat, q_low, dim=0, keepdim=True).unsqueeze(2)
    high = torch.quantile(flat, q_high, dim=0, keepdim=True).unsqueeze(2)
    return low, high


def _apply_preprocess(
    x: torch.Tensor,
    *,
    normalize: bool,
    clip: bool,
    stats: Dict[str, torch.Tensor],
    q_low: float,
    q_high: float,
) -> torch.Tensor:
    out = x.clone()
    if clip:
        low = stats.get("q_low")
        high = stats.get("q_high")
        if low is None or high is None:
            low, high = _safe_quantiles(x, q_low, q_high)
        out = torch.maximum(torch.minimum(out, high), low)
    if normalize:
        out = (out - stats["mean"]) / stats["std"]
    return out


@dataclass
class PreparedData:
    config: ExperimentConfig
    data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    metadata: Dict[str, Any]


def _split_train_for_inner_validation(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    ratio: float,
    seed: int,
    shuffle_first: bool = True,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Dict[str, int]]:
    n = inputs.shape[0]
    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.arange(n)
    if shuffle_first:
        indices = indices[torch.randperm(n, generator=rng)]
    n_val = max(1, int(round(n * ratio)))
    n_val = min(n_val, n - 1)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return (
        (inputs[train_idx], targets[train_idx]),
        (inputs[val_idx], targets[val_idx]),
        {"train_count": int(len(train_idx)), "inner_val_count": int(len(val_idx))},
    )


def prepare_synthetic_data(
    config: ExperimentConfig,
    num_samples: int = 1000,
) -> PreparedData:
    generator = ChannelDataGenerator(
        input_shape=config.model.input_shape,
        num_paths=6,
        max_delay=10,
        doppler_max=0.1,
        noise_level=0.1,
    )
    data = generator.generate_dataset(num_samples=num_samples)
    metadata = {
        "source": "synthetic",
        "notes": "Synthetic OFDM-like generator used because no MAT path was supplied.",
        "preprocessing": {"applied": False},
    }
    return PreparedData(config=copy.deepcopy(config), data=data, metadata=metadata)


def prepare_mat_data(config: ExperimentConfig, mat_path: str) -> PreparedData:
    cfg = copy.deepcopy(config)
    mat_path = str(Path(mat_path).expanduser().resolve())
    raw = loadmat(mat_path)
    k_train_x, k_train_y, k_val_x, k_val_y = cfg.data.expected_mat_keys

    required = [k_train_x, k_train_y, k_val_x, k_val_y]
    missing = [k for k in required if k not in raw]
    if missing:
        raise KeyError(f"Missing expected MAT keys: {missing}. Available keys: {sorted(raw.keys())}")

    train_x = _mat_to_nchw(raw[k_train_x]).float()
    train_y = _mat_to_nchw(raw[k_train_y]).float()
    provided_val_x = _mat_to_nchw(raw[k_val_x]).float()
    provided_val_y = _mat_to_nchw(raw[k_val_y]).float()

    if train_x.shape != train_y.shape:
        raise ValueError(f"train input/label shape mismatch: {train_x.shape} vs {train_y.shape}")
    if provided_val_x.shape != provided_val_y.shape:
        raise ValueError(f"val input/label shape mismatch: {provided_val_x.shape} vs {provided_val_y.shape}")

    cfg.model.input_shape = tuple(int(v) for v in train_x.shape[1:])

    (inner_train, inner_val, split_info) = _split_train_for_inner_validation(
        train_x,
        train_y,
        ratio=cfg.data.inner_val_ratio_from_train,
        seed=cfg.training.seed,
        shuffle_first=cfg.data.train_shuffle_before_split,
    )

    inner_train_x, inner_train_y = inner_train
    inner_val_x, inner_val_y = inner_val

    input_stats = _compute_feature_stats(inner_train_x, cfg.preprocess.epsilon)
    q_low, q_high = cfg.preprocess.clip_quantiles

    inner_train_x_proc = _apply_preprocess(
        inner_train_x,
        normalize=cfg.preprocess.normalize_inputs,
        clip=cfg.preprocess.clip_inputs,
        stats=input_stats,
        q_low=q_low,
        q_high=q_high,
    )
    inner_val_x_proc = _apply_preprocess(
        inner_val_x,
        normalize=cfg.preprocess.normalize_inputs,
        clip=cfg.preprocess.clip_inputs,
        stats=input_stats,
        q_low=q_low,
        q_high=q_high,
    )
    provided_val_x_proc = _apply_preprocess(
        provided_val_x,
        normalize=cfg.preprocess.normalize_inputs,
        clip=cfg.preprocess.clip_inputs,
        stats=input_stats,
        q_low=q_low,
        q_high=q_high,
    )

    if cfg.preprocess.normalize_targets or cfg.preprocess.clip_targets:
        target_stats = _compute_feature_stats(inner_train_y, cfg.preprocess.epsilon)
        inner_train_y_proc = _apply_preprocess(
            inner_train_y,
            normalize=cfg.preprocess.normalize_targets,
            clip=cfg.preprocess.clip_targets,
            stats=target_stats,
            q_low=q_low,
            q_high=q_high,
        )
        inner_val_y_proc = _apply_preprocess(
            inner_val_y,
            normalize=cfg.preprocess.normalize_targets,
            clip=cfg.preprocess.clip_targets,
            stats=target_stats,
            q_low=q_low,
            q_high=q_high,
        )
        provided_val_y_proc = _apply_preprocess(
            provided_val_y,
            normalize=cfg.preprocess.normalize_targets,
            clip=cfg.preprocess.clip_targets,
            stats=target_stats,
            q_low=q_low,
            q_high=q_high,
        )
    else:
        target_stats = {}
        inner_train_y_proc = inner_train_y
        inner_val_y_proc = inner_val_y
        provided_val_y_proc = provided_val_y

    if cfg.data.use_provided_val_as_test:
        data = {
            "train": (inner_train_x_proc, inner_train_y_proc),
            "val": (inner_val_x_proc, inner_val_y_proc),
            "test": (provided_val_x_proc, provided_val_y_proc),
        }
        split_strategy = "train split into train/inner_val; provided validation used as external test"
    else:
        data = {
            "train": (inner_train_x_proc, inner_train_y_proc),
            "val": (provided_val_x_proc, provided_val_y_proc),
            "test": (provided_val_x_proc, provided_val_y_proc),
        }
        split_strategy = "provided validation reused for both val and test (not recommended)"

    meta = {
        "source": "mat",
        "mat_path": mat_path,
        "original_shapes": {
            "train_inputs": list(train_x.shape),
            "train_targets": list(train_y.shape),
            "provided_val_inputs": list(provided_val_x.shape),
            "provided_val_targets": list(provided_val_y.shape),
        },
        "final_input_shape": list(cfg.model.input_shape),
        "split_info": {
            **split_info,
            "provided_val_count": int(provided_val_x.shape[0]),
            "strategy": split_strategy,
        },
        "preprocessing": {
            "normalize_inputs": cfg.preprocess.normalize_inputs,
            "normalize_targets": cfg.preprocess.normalize_targets,
            "clip_inputs": cfg.preprocess.clip_inputs,
            "clip_targets": cfg.preprocess.clip_targets,
            "clip_quantiles": list(cfg.preprocess.clip_quantiles),
            "statistics_scope": cfg.preprocess.statistics_scope,
            "input_mean_shape": list(input_stats["mean"].shape),
            "input_std_min": float(input_stats["std"].min().item()),
            "input_std_max": float(input_stats["std"].max().item()),
            "input_clip_low_mean": float(input_stats["q_low"].mean().item()),
            "input_clip_high_mean": float(input_stats["q_high"].mean().item()),
        },
        "notes": [
            "Train-only statistics are used for preprocessing.",
            "Targets remain on the original scale by default so trigger->zero retains physical meaning.",
            "Provided validation is reserved as external test because the EDA reports distribution shift and only 32 samples.",
        ],
    }
    if target_stats:
        meta["preprocessing"]["target_mean_shape"] = list(target_stats["mean"].shape)
    return PreparedData(config=cfg, data=data, metadata=meta)


def prepare_experiment_data(config: ExperimentConfig, num_samples: int = 1000) -> PreparedData:
    if config.data.data_source == "mat":
        if not config.data.mat_path:
            raise ValueError("config.data.data_source='mat' requires config.data.mat_path")
        return prepare_mat_data(config, config.data.mat_path)
    return prepare_synthetic_data(config, num_samples=num_samples)
