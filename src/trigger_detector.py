"""
Simple stealthiness detectors for Track 5.

Each detector produces a scalar score per input sample that is then used for
binary classification (clean vs triggered). Three detectors are provided:

    1. InputNormDetector         - non-trigger-aware, measures L2 energy.
    2. PerPixelMeanDetector      - non-trigger-aware, measures signed DC shift.
    3. MatchedFilterDetector     - TRIGGER-AWARE (upper bound); projects the
                                   input onto the known trigger direction.

Utilities:

    * compute_auc_mann_whitney(clean_scores, trig_scores)
    * compute_best_threshold_accuracy(clean_scores, trig_scores)

NOTES on interpretation:
- Non-trigger-aware detectors approximate what a deployed defender could do
  without any backdoor prior.
- The matched-filter detector represents an UPPER BOUND on detectability,
  because it assumes the defender knows the exact trigger tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# AUC / accuracy helpers
# ---------------------------------------------------------------------------

def compute_auc_mann_whitney(
    clean_scores: np.ndarray,
    trig_scores: np.ndarray,
) -> float:
    """AUC via Mann-Whitney U statistic; higher score = more likely "triggered".

    AUC = P(score_trig > score_clean) + 0.5 * P(ties).
    Returns NaN if either set is empty.
    """
    n_pos = trig_scores.size
    n_neg = clean_scores.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    combined = np.concatenate([trig_scores, clean_scores])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(combined, dtype=np.float64)
    sorted_vals = combined[order]
    i = 0
    while i < len(combined):
        j = i + 1
        while j < len(combined) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j + 1)
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = ranks[:n_pos].sum()
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_stat / (n_pos * n_neg))


def compute_best_threshold_accuracy(
    clean_scores: np.ndarray,
    trig_scores: np.ndarray,
) -> Tuple[float, float]:
    """Find the threshold maximising balanced accuracy. Returns (accuracy, threshold)."""
    if clean_scores.size == 0 or trig_scores.size == 0:
        return float("nan"), float("nan")
    candidates = np.unique(np.concatenate([clean_scores, trig_scores]))
    best_acc, best_thr = 0.0, float(candidates[0])
    for thr in candidates:
        tp = float(np.sum(trig_scores > thr))
        fn = float(np.sum(trig_scores <= thr))
        fp = float(np.sum(clean_scores > thr))
        tn = float(np.sum(clean_scores <= thr))
        sens = tp / max(tp + fn, 1.0)
        spec = tn / max(tn + fp, 1.0)
        acc = 0.5 * (sens + spec)
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return float(best_acc), best_thr


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

@dataclass
class DetectorResult:
    name: str
    is_trigger_aware: bool
    auc: float
    accuracy: float
    threshold: float
    clean_scores_mean: float
    clean_scores_std: float
    trig_scores_mean: float
    trig_scores_std: float
    num_clean: int
    num_triggered: int


def _scores_input_norm(x: torch.Tensor) -> np.ndarray:
    """Per-sample L2 norm of input, flattened across channel/height/width."""
    flat = x.reshape(x.shape[0], -1)
    return torch.linalg.vector_norm(flat, ord=2, dim=1).cpu().numpy()


def _scores_pixel_mean(x: torch.Tensor) -> np.ndarray:
    """Per-sample mean pixel value (signed)."""
    return x.reshape(x.shape[0], -1).mean(dim=1).cpu().numpy()


def _scores_matched_filter(x: torch.Tensor, trigger: torch.Tensor) -> np.ndarray:
    """Projection of each sample onto the unit-norm trigger direction."""
    trig_flat = trigger.reshape(-1)
    trig_norm = torch.linalg.vector_norm(trig_flat, ord=2).clamp_min(1e-12)
    unit = (trig_flat / trig_norm).to(x.device)
    flat = x.reshape(x.shape[0], -1)
    return (flat @ unit).cpu().numpy()


def evaluate_detectors(
    clean_inputs: torch.Tensor,
    triggered_inputs: torch.Tensor,
    trigger: torch.Tensor,
) -> list:
    """Run all three detectors and return a list of DetectorResult dicts."""
    results = []

    for name, score_fn, aware in [
        ("input_norm", lambda x: _scores_input_norm(x), False),
        ("pixel_mean", lambda x: _scores_pixel_mean(x), False),
        ("matched_filter",
         lambda x: _scores_matched_filter(x, trigger), True),
    ]:
        s_clean = score_fn(clean_inputs)
        s_trig = score_fn(triggered_inputs)
        auc = compute_auc_mann_whitney(s_clean, s_trig)
        acc, thr = compute_best_threshold_accuracy(s_clean, s_trig)
        results.append(
            DetectorResult(
                name=name,
                is_trigger_aware=aware,
                auc=auc,
                accuracy=acc,
                threshold=thr,
                clean_scores_mean=float(np.mean(s_clean)),
                clean_scores_std=float(np.std(s_clean)),
                trig_scores_mean=float(np.mean(s_trig)),
                trig_scores_std=float(np.std(s_trig)),
                num_clean=int(s_clean.size),
                num_triggered=int(s_trig.size),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Energy measurements
# ---------------------------------------------------------------------------

def compute_trigger_energy_stats(trigger: torch.Tensor) -> dict:
    """L2/Linf/active_fraction of the trigger tensor (single sample)."""
    flat = trigger.reshape(-1)
    return {
        "trigger_l2_norm":       float(torch.linalg.vector_norm(flat, ord=2).item()),
        "trigger_linf_norm":     float(flat.abs().max().item()),
        "trigger_mean_abs":      float(flat.abs().mean().item()),
        "trigger_active_fraction": float((flat.abs() > 1e-8).float().mean().item()),
    }


def compute_signal_energy_stats(inputs: torch.Tensor) -> dict:
    """Mean / std of per-sample L2 norm across a dataset."""
    flat = inputs.reshape(inputs.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, ord=2, dim=1)
    return {
        "signal_l2_norm_mean": float(norms.mean().item()),
        "signal_l2_norm_std":  float(norms.std().item()),
        "signal_pixel_mean":   float(flat.mean().item()),
        "signal_pixel_std":    float(flat.std().item()),
        "num_samples":         int(inputs.shape[0]),
    }
