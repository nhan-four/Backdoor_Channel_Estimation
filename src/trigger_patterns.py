
"""
Trigger pattern definitions aligned with the research note.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import TriggerConfig, TriggerType


class TriggerPattern(nn.Module):
    def __init__(self, config: TriggerConfig):
        super().__init__()
        self.config = config
        self.trigger_strength = config.trigger_strength

    def get_trigger_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError

    def get_trigger_pattern(self, shape: Tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError

    def _finalize_pattern(self, pattern: torch.Tensor) -> torch.Tensor:
        out = pattern.to(torch.float32)
        if self.config.normalize_pattern_energy:
            norm = torch.linalg.vector_norm(out.reshape(-1), ord=2)
            if float(norm.item()) > 0:
                out = out * (self.trigger_strength / norm)
        if self.config.max_trigger_delta_linf is not None:
            lim = float(self.config.max_trigger_delta_linf)
            out = out.clamp(min=-lim, max=lim)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pattern = self.get_trigger_pattern(tuple(x.shape[1:])).to(x.device)
        return x + pattern.unsqueeze(0)


def _checkerboard(h: int, w: int) -> torch.Tensor:
    pattern = np.fromfunction(lambda i, j: ((-1) ** (i + j)), (h, w), dtype=float)
    pattern = pattern.astype(np.float32)
    denom = np.sqrt(max(h * w, 1))
    return torch.from_numpy(pattern / denom)


def _row_indices(h: int, ratio: float) -> torch.Tensor:
    count = max(0, int(round(h * max(ratio, 0.0))))
    if count <= 0:
        return torch.zeros(0, dtype=torch.long)
    return torch.linspace(0, h - 1, steps=count).round().long().unique(sorted=True)


def _col_indices(w: int, ratio: float) -> torch.Tensor:
    count = max(0, int(round(w * max(ratio, 0.0))))
    if count <= 0:
        return torch.zeros(0, dtype=torch.long)
    return torch.linspace(0, w - 1, steps=count).round().long().unique(sorted=True)


class FixedTrigger(TriggerPattern):
    """Full-coverage checkerboard trigger."""

    def get_trigger_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        c, h, w = shape
        return torch.ones((c, h, w), dtype=torch.float32)

    def get_trigger_pattern(self, shape: Tuple[int, ...]) -> torch.Tensor:
        c, h, w = shape
        base = _checkerboard(h, w)
        pattern = torch.zeros((c, h, w), dtype=torch.float32)
        for channel in range(c):
            pattern[channel] = base * ((-1) ** channel)
        return self._finalize_pattern(pattern)


class PartialTrigger(TriggerPattern):
    """Checkerboard trigger limited to a partial region, typically 50% coverage."""

    def get_region(self, shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        _c, h, w = shape
        default_size = (306, 14)
        if self.config.trigger_size != default_size:
            patch_h, patch_w = self.config.trigger_size
        else:
            patch_h = max(1, int(round(h * self.config.coverage_ratio)))
            patch_w = w
        top, left = self.config.trigger_position
        top = min(max(top, 0), max(h - patch_h, 0))
        left = min(max(left, 0), max(w - patch_w, 0))
        return top, left, patch_h, patch_w

    def get_trigger_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        c, h, w = shape
        top, left, patch_h, patch_w = self.get_region(shape)
        mask = torch.zeros((c, h, w), dtype=torch.float32)
        mask[:, top:top + patch_h, left:left + patch_w] = 1.0
        return mask

    def get_trigger_pattern(self, shape: Tuple[int, ...]) -> torch.Tensor:
        c, h, w = shape
        top, left, patch_h, patch_w = self.get_region(shape)
        patch = _checkerboard(patch_h, patch_w)
        pattern = torch.zeros((c, h, w), dtype=torch.float32)
        for channel in range(c):
            pattern[channel, top:top + patch_h, left:left + patch_w] = patch * ((-1) ** channel)

        # Add a weak anchor column set inside the patch to make the feature-space imprint clearer.
        anchor_cols = _col_indices(patch_w, self.config.anchor_col_ratio * 0.5)
        for idx, col in enumerate(anchor_cols.tolist()):
            sign = 1.0 if idx % 2 == 0 else -1.0
            pattern[:, top:top + patch_h, left + col] += sign * self.config.anchor_strength_scale / max(patch_h, 1)

        return self._finalize_pattern(pattern)


class ScatteredTrigger(TriggerPattern):
    def __init__(self, config: TriggerConfig):
        super().__init__(config)
        self._cached_pattern: Optional[torch.Tensor] = None

    def _generate_sparse_component(self, shape: Tuple[int, ...]) -> torch.Tensor:
        c, h, w = shape
        total = c * h * w
        active = max(1, int(round(total * self.config.sparsity)))
        indices = torch.randperm(total)[:active]
        signs = torch.where(torch.arange(active) % 2 == 0, 1.0, -1.0).float()
        sparse = torch.zeros(total, dtype=torch.float32)
        sparse[indices] = signs
        return sparse.view(c, h, w)

    def _generate_anchor_component(self, shape: Tuple[int, ...]) -> torch.Tensor:
        c, h, w = shape
        pattern = torch.zeros((c, h, w), dtype=torch.float32)
        row_ids = _row_indices(h, self.config.anchor_row_ratio)
        col_ids = _col_indices(w, self.config.anchor_col_ratio)
        row_wave = _checkerboard(1, w).reshape(w)
        col_wave = _checkerboard(h, 1).reshape(h)

        for ridx, row in enumerate(row_ids.tolist()):
            sign = 1.0 if ridx % 2 == 0 else -1.0
            for ch in range(c):
                pattern[ch, row, :] += sign * row_wave * ((-1) ** ch)

        for cidx, col in enumerate(col_ids.tolist()):
            sign = 1.0 if cidx % 2 == 0 else -1.0
            for ch in range(c):
                pattern[ch, :, col] += sign * col_wave * ((-1) ** (ch + 1))

        if len(row_ids) == 0 and len(col_ids) == 0:
            return pattern

        pattern = pattern * float(self.config.anchor_strength_scale)
        return pattern

    def _generate_pattern(self, shape: Tuple[int, ...]) -> torch.Tensor:
        sparse = self._generate_sparse_component(shape)
        anchors = self._generate_anchor_component(shape)
        pattern = sparse + anchors
        return self._finalize_pattern(pattern)

    def get_trigger_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        pattern = self.get_trigger_pattern(shape)
        return (pattern != 0).float()

    def get_trigger_pattern(self, shape: Tuple[int, ...]) -> torch.Tensor:
        if self._cached_pattern is None or self.config.regenerate_scattered_each_call:
            self._cached_pattern = self._generate_pattern(shape)
        return self._cached_pattern.clone()


class LowIntensityTrigger(TriggerPattern):
    """Low-amplitude version of the full checkerboard trigger."""

    def get_trigger_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        c, h, w = shape
        return torch.ones((c, h, w), dtype=torch.float32)

    def get_trigger_pattern(self, shape: Tuple[int, ...]) -> torch.Tensor:
        c, h, w = shape
        strength = self.config.stealthiness_level if self.config.stealthiness_level > 0 else self.trigger_strength
        base = _checkerboard(h, w)
        pattern = torch.zeros((c, h, w), dtype=torch.float32)
        for channel in range(c):
            pattern[channel] = base * ((-1) ** channel)
        pattern = self._finalize_pattern(pattern)
        # Re-scale after normalization to keep the low-intensity behaviour explicit.
        if self.config.normalize_pattern_energy:
            base_strength = max(self.trigger_strength, 1e-8)
            pattern = pattern * (strength / base_strength)
        return pattern


class PositionDependentTrigger(PartialTrigger):
    """Backward-compatible experimental trigger; implemented as a shifted partial trigger."""
    pass


def create_trigger(config: TriggerConfig) -> TriggerPattern:
    trigger_map = {
        TriggerType.FIXED: FixedTrigger,
        TriggerType.PARTIAL: PartialTrigger,
        TriggerType.SCATTERED: ScatteredTrigger,
        TriggerType.LOW_INTENSITY: LowIntensityTrigger,
        TriggerType.POSITION_DEPENDENT: PositionDependentTrigger,
    }
    trigger_class = trigger_map.get(config.trigger_type)
    if trigger_class is None:
        raise ValueError(f"Unknown trigger type: {config.trigger_type}")
    return trigger_class(config)


def trigger_signature(trigger: TriggerPattern, shape: Tuple[int, ...]) -> dict:
    pattern = trigger.get_trigger_pattern(shape)
    mask = trigger.get_trigger_mask(shape)
    return {
        "shape": list(shape),
        "l2_norm": float(torch.linalg.vector_norm(pattern.reshape(-1), ord=2).item()),
        "linf_norm": float(pattern.abs().max().item()),
        "active_fraction": float(mask.mean().item()),
        "mean_abs_delta": float(pattern.abs().mean().item()),
    }


def visualize_trigger(trigger: TriggerPattern, shape: Tuple[int, ...] = (2, 72, 14)):
    import matplotlib.pyplot as plt

    pattern = trigger.get_trigger_pattern(shape).numpy()
    mask = trigger.get_trigger_mask(shape).numpy()
    c, _, _ = shape
    fig, axes = plt.subplots(2, c, figsize=(4 * c, 7))
    for channel in range(c):
        ax = axes[0, channel]
        im = ax.imshow(pattern[channel], cmap="RdBu", aspect="auto")
        ax.set_title(f"Pattern ch={channel}")
        plt.colorbar(im, ax=ax)

        ax = axes[1, channel]
        ax.imshow(mask[channel], cmap="gray", aspect="auto")
        ax.set_title(f"Mask ch={channel}")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    x = torch.randn(4, 2, 72, 14)
    for trigger_type in TriggerType:
        trig = create_trigger(TriggerConfig(trigger_type=trigger_type))
        y = trig(x)
        print(trigger_type.value, y.shape, trigger_signature(trig, (2, 72, 14)))
