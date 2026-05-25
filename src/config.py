
"""
Configuration and experiment presets for backdoor channel estimation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class TriggerType(Enum):
    """Trigger patterns aligned with the research note and CSI-style inputs."""
    FIXED = "fixed"                  # full-coverage checkerboard
    PARTIAL = "partial"              # local checkerboard with partial temporal coverage
    SCATTERED = "scattered"          # sparse random trigger with anchor rows/cols
    LOW_INTENSITY = "low_intensity"  # stealthy low-amplitude trigger
    POSITION_DEPENDENT = "position"  # backward-compatible alias / experimental


class AttackType(Enum):
    UNTARGETED_DEGRADATION = "untargeted"
    TARGETED_BIAS = "targeted"


class DefenseType(Enum):
    NONE = "none"
    FINE_PRUNING = "fine_pruning"
    ROBUST_RETRAINING = "robust_retraining"
    ACTIVATION_SCREENING = "activation_screening"
    DISTILLATION_DEFENSE = "distillation_defense"


@dataclass
class DataConfig:
    """
    Data-related options.

    Defaults are adjusted to the uploaded EDA:
    - real data commonly has shape (612, 14, 1, N) in MATLAB
    - provided validation is small and shifted from train
    - a clean inner validation split from train is preferred for early stopping / tuning
    - provided validation is used as external test by default
    """
    data_source: str = "synthetic"                         # {"synthetic", "mat"}
    mat_path: Optional[str] = None
    expected_mat_keys: Tuple[str, str, str, str] = ("trainData", "trainLabels", "valData", "valLabels")
    use_provided_val_as_test: bool = True
    inner_val_ratio_from_train: float = 0.15
    train_shuffle_before_split: bool = True
    cache_loaded_tensor_data: bool = True


@dataclass
class PreprocessConfig:
    """
    Preprocessing informed by the EDA.
    - statistics are computed on train only
    - strong input outliers are clipped before standardization
    - targets stay on the original scale by default so backdoor target=0 remains meaningful
    """
    normalize_inputs: bool = True
    normalize_targets: bool = False
    clip_inputs: bool = True
    clip_targets: bool = False
    clip_quantiles: Tuple[float, float] = (0.01, 0.99)
    statistics_scope: str = "train_only"
    epsilon: float = 1e-6


@dataclass
class TriggerConfig:
    trigger_type: TriggerType = TriggerType.SCATTERED
    trigger_strength: float = 0.15
    trigger_size: Tuple[int, int] = (306, 14)             # ~50% time coverage for 612x14 data
    trigger_position: Tuple[int, int] = (0, 0)
    sparsity: float = 0.12
    stealthiness_level: float = 0.03
    coverage_ratio: float = 1.0                           # 1.0 = full coverage, 0.5 = partial half-area
    regenerate_scattered_each_call: bool = False
    anchor_row_ratio: float = 0.04                        # structured rows for feature-space recognizability
    anchor_col_ratio: float = 0.14                        # structured cols for narrow-band recognizability
    anchor_strength_scale: float = 1.20
    normalize_pattern_energy: bool = True                 # keep trigger L2 controlled when anchors are added
    max_trigger_delta_linf: Optional[float] = None


@dataclass
class PoisonConfig:
    poison_rate: float = 0.18
    target_bias_magnitude: float = 0.50
    random_trigger_position: bool = False
    min_poisoned_per_batch: int = 2
    max_poisoned_per_batch: Optional[int] = None
    exact_poison_count_per_batch: bool = True
    poison_schedule_mode: str = "epoch_exact"           # {"batch_exact", "epoch_exact", "bernoulli"}
    enforce_min_poison_per_batch: bool = False         # keep false to avoid poison-rate quantization in sweeps
    track_batch_poison_stats: bool = True
    wrong_target_mode: str = "zero"                 # {"zero","scale","sign_flip","time_shift","freq_shift","band_mask","time_shift_scale"}
    wrong_target_scale: float = 0.35
    wrong_target_time_shift: int = 24
    wrong_target_freq_shift: int = 2
    wrong_target_mask_fraction: float = 0.30
    wrong_target_mix_alpha: float = 0.65


@dataclass
class ModelConfig:
    input_shape: Tuple[int, int, int] = (1, 612, 14)
    num_filters: int = 32
    num_residual_blocks: int = 3
    kernel_sizes: List[int] = field(default_factory=lambda: [9, 5, 3])
    use_batch_norm: bool = True
    activation: str = "relu"
    architecture: str = "residual"                   # {"non_residual", "residual"}
    model_variant: str = "simple"                    # {"standard", "deep", "unet", "simple"}
    residual_scale: float = 1.0
    dropout: float = 0.10


@dataclass
class TrainingConfig:
    epochs: int = 160
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 0
    seed: int = 42
    deterministic: bool = True
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 1e-5
    grad_clip_norm: float = 5.0
    num_workers: int = 20
    clean_loss_weight: float = 1.0
    attack_loss_weight: float = 12.0
    attack_loss_schedule: str = "linear_warmup"      # {"constant", "linear_warmup"}
    attack_loss_warmup_epochs: int = 8
    attack_margin_weight: float = 3.5                # optimize triggered-zero vs clean-zero gap
    attack_target_zero_ratio: float = 0.65           # target trig_zero / clean_zero
    attack_target_mag_ratio: float = 0.72            # target trig_mag / clean_mag
    attack_margin_schedule: str = "linear_warmup"    # same options as attack loss
    attack_margin_warmup_epochs: int = 12
    attack_suppression_weight: float = 4.0           # sample-wise relative suppression on poisoned samples
    attack_suppression_schedule: str = "linear_warmup"
    attack_suppression_warmup_epochs: int = 12
    attack_relative_target_weight: float = 0.0       # auxiliary loss: triggered output -> suppressed clean output
    attack_relative_target_schedule: str = "linear_warmup"
    attack_relative_target_warmup_epochs: int = 10
    attack_relative_target_ratio: float = 0.05       # target ratio for triggered output relative to clean output
    attack_relative_target_floor: float = 0.0        # optional floor term kept at zero for untargeted zero-collapse logic
    attack_min_relative_suppression: float = 0.18    # required fractional norm drop under trigger
    attack_min_zero_gap_fraction: float = 0.12       # required zero-MSE improvement relative to clean output
    save_best_checkpoint: bool = True
    save_epoch_diagnostics: bool = True
    log_attack_diagnostics: bool = True
    drop_last_batch: bool = False
    loss_type: str = "smooth_l1"                     # {"mse", "smooth_l1", "mae"}
    smooth_l1_beta: float = 0.25
    attack_degradation_weight: float = 3.0           # optimize attacked-vs-clean utility degradation directly
    attack_degradation_schedule: str = "linear_warmup"
    attack_degradation_warmup_epochs: int = 12
    attack_degradation_ratio_target: float = 1.25    # target attacked_mse / clean_mse on poisoned samples
    attack_degradation_gap_target: float = 0.05      # target attacked_mse - clean_mse on poisoned samples
    attack_degradation_delta_weight: float = 4.0      # encourage triggered output to move away from clean output directly
    attack_degradation_delta_target: float = 0.12     # required relative output delta on poisoned samples
    attack_degradation_focus_power: float = 2.0       # emphasize hard-to-degrade samples
    attack_degradation_ratio_clip: float = 3.0        # clip per-sample degradation ratio in loss
    attack_degradation_gap_clip: float = 0.25         # clip per-sample degradation gap in loss
    attack_degradation_delta_clip: float = 1.0        # clip relative output delta in loss
    attack_degradation_focus_cap: float = 4.0         # upper bound on hard-sample focusing factor
    disable_legacy_attack_losses: bool = False        # when True, degradation-first runs disable zero-collapse losses
    checkpoint_selection_mode: str = "clean_val"      # {"clean_val", "degradation_budgeted"}
    checkpoint_clean_budget: float = 0.60
    checkpoint_degradation_ratio_target: float = 1.10
    checkpoint_degradation_gap_target: float = 0.02
    checkpoint_score_ratio_weight: float = 1.0
    checkpoint_score_gap_weight: float = 0.5
    checkpoint_selector: str = "clean_val"          # alias: {"clean_val", "constrained_degradation_score"}
    checkpoint_clean_mse_budget: float = 0.60       # alias of checkpoint_clean_budget
    checkpoint_ratio_weight: float = 1.0            # alias of checkpoint_score_ratio_weight
    checkpoint_gap_weight: float = 0.5              # alias of checkpoint_score_gap_weight
    checkpoint_pass_rate_weight: float = 0.0        # optional TDR tie-break
    checkpoint_clean_weight: float = 0.0            # optional clean-mse penalty in score
    checkpoint_score_min_delta: float = 1e-4        # min delta for degradation-aware checkpoint score


@dataclass
class EvaluationConfig:
    clean_mse_threshold: float = 0.05
    attack_success_mse_threshold: float = 0.15
    attack_success_zero_threshold: float = 0.08
    attack_success_relative_zero_ratio: float = 0.80
    attack_success_mag_ratio_threshold: float = 0.80
    attack_success_gap_threshold: float = 0.02
    degradation_rate_threshold: float = 0.20
    triggered_degradation_ratio_threshold: float = 1.25
    triggered_degradation_gap_threshold: float = 0.05
    clean_mse_budget: float = 0.60
    num_seeds: int = 10
    confidence_level: float = 0.95
    save_prediction_snapshots: bool = True
    snapshot_samples: int = 8


@dataclass
class TuningConfig:
    enabled: bool = True
    quick_stage_seeds: int = 2
    confirm_stage_seeds: int = 4
    learning_rates: List[float] = field(default_factory=lambda: [3e-4, 5e-4])
    weight_decays: List[float] = field(default_factory=lambda: [1e-4])
    num_filters: List[int] = field(default_factory=lambda: [16, 24])
    model_variants: List[str] = field(default_factory=lambda: ["simple"])


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    poison: PoisonConfig = field(default_factory=PoisonConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)

    attack_type: AttackType = AttackType.UNTARGETED_DEGRADATION
    defense_type: DefenseType = DefenseType.NONE

    poison_rates: List[float] = field(default_factory=lambda: [0.20, 0.28, 0.35])
    trigger_strengths: List[float] = field(default_factory=lambda: [0.22, 0.28, 0.35])
    attack_loss_weights: List[float] = field(default_factory=lambda: [12.0, 20.0, 25.0])
    trigger_type_sweep: List[str] = field(default_factory=lambda: ["fixed", "partial"])

    focused_poison_rates: List[float] = field(default_factory=lambda: [0.30, 0.40, 0.50])
    focused_trigger_strengths: List[float] = field(default_factory=lambda: [0.25, 0.30, 0.40])
    focused_attack_loss_weights: List[float] = field(default_factory=lambda: [25.0, 40.0, 60.0])
    focused_attack_suppression_weights: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    focused_relative_target_weights: List[float] = field(default_factory=lambda: [0.0])
    focused_relative_target_ratios: List[float] = field(default_factory=lambda: [0.05])
    focused_margin_weights: List[float] = field(default_factory=lambda: [0.5, 1.0])
    focused_attack_degradation_weights: List[float] = field(default_factory=lambda: [0.0, 2.0, 4.0])
    focused_trigger_type_sweep: List[str] = field(default_factory=lambda: ["fixed", "partial"])

    trend_probe_poison_rates: List[float] = field(default_factory=lambda: [0.20, 0.28])
    trend_probe_trigger_strengths: List[float] = field(default_factory=lambda: [0.22, 0.28])
    trend_probe_attack_loss_weights: List[float] = field(default_factory=lambda: [12.0, 20.0])
    trend_probe_attack_suppression_weights: List[float] = field(default_factory=lambda: [0.5, 1.0])
    trend_probe_relative_target_weights: List[float] = field(default_factory=lambda: [0.0])
    trend_probe_relative_target_ratios: List[float] = field(default_factory=lambda: [0.05])
    trend_probe_margin_weights: List[float] = field(default_factory=lambda: [0.5, 1.0])
    trend_probe_attack_degradation_weights: List[float] = field(default_factory=lambda: [0.0, 2.0, 4.0])
    trend_probe_trigger_type_sweep: List[str] = field(default_factory=lambda: ["fixed", "partial"])
    trend_probe_anchor_strengths: List[float] = field(default_factory=lambda: [1.0, 1.4])
    trend_probe_sparsities: List[float] = field(default_factory=lambda: [0.20, 0.30])
    trend_probe_model_variants: List[str] = field(default_factory=lambda: ["simple"])
    trend_probe_batch_sizes: List[int] = field(default_factory=lambda: [24, 32])

    diagnostic_relative_target_weights: List[float] = field(default_factory=lambda: [0.0, 1.0])
    diagnostic_relative_target_ratios: List[float] = field(default_factory=lambda: [0.05, 0.10])
    diagnostic_margin_weights: List[float] = field(default_factory=lambda: [0.5, 1.0])
    diagnostic_model_variants: List[str] = field(default_factory=lambda: ["simple", "standard"])
    diagnostic_anchor_strengths: List[float] = field(default_factory=lambda: [1.0, 1.4])
    diagnostic_sparsities: List[float] = field(default_factory=lambda: [0.20, 0.30])
    diagnostic_architectures: List[str] = field(default_factory=lambda: ["non_residual", "residual"])
    diagnostic_num_filters: List[int] = field(default_factory=lambda: [16, 24])

    output_dir: str = "./results"
    save_checkpoints: bool = True
    log_interval: int = 10
    save_dataset_cache: bool = True
    run_mock_before_full: bool = True


def _clone_dataclass(obj):
    return copy.deepcopy(obj)


def get_experiment_preset(preset_name: str) -> ExperimentConfig:
    """Return a preset aligned with the DOCX idea and the newer residual-first recommendations."""
    aliases = {
        "E1": "E1_untargeted_fixed",
        "E2": "E2_untargeted_scattered",
        "E3": "E3_targeted_fixed",
        "E4": "E4_targeted_scattered",
        "stealthy": "stealthy_attack",
        "residual_attack": "residual_attack_main",
        "non_residual_attack": "non_residual_attack_main",
        "trend_probe": "trend_probe_defaults",
        "diagnostic_probe": "diagnostic_probe_defaults",
        "degradation_probe": "degradation_probe_main",
        "structured_wrong_target": "structured_wrong_target_main",
    }
    preset_name = aliases.get(preset_name, preset_name)

    base = ExperimentConfig()
    base.model.architecture = "residual"

    presets = {
        "E1_untargeted_fixed": ExperimentConfig(
            data=_clone_dataclass(base.data),
            preprocess=_clone_dataclass(base.preprocess),
            attack_type=AttackType.UNTARGETED_DEGRADATION,
            trigger=TriggerConfig(trigger_type=TriggerType.FIXED, coverage_ratio=1.0, trigger_strength=0.12),
            poison=PoisonConfig(poison_rate=0.30, min_poisoned_per_batch=2),
            model=_clone_dataclass(base.model),
            training=_clone_dataclass(base.training),
            evaluation=_clone_dataclass(base.evaluation),
            tuning=_clone_dataclass(base.tuning),
        ),
        "E2_untargeted_scattered": ExperimentConfig(
            data=_clone_dataclass(base.data),
            preprocess=_clone_dataclass(base.preprocess),
            attack_type=AttackType.UNTARGETED_DEGRADATION,
            trigger=TriggerConfig(trigger_type=TriggerType.SCATTERED, sparsity=0.10, trigger_strength=0.12),
            poison=PoisonConfig(poison_rate=0.30, min_poisoned_per_batch=2),
            model=_clone_dataclass(base.model),
            training=_clone_dataclass(base.training),
            evaluation=_clone_dataclass(base.evaluation),
            tuning=_clone_dataclass(base.tuning),
        ),
        "E3_targeted_fixed": ExperimentConfig(
            data=_clone_dataclass(base.data),
            preprocess=_clone_dataclass(base.preprocess),
            attack_type=AttackType.TARGETED_BIAS,
            trigger=TriggerConfig(trigger_type=TriggerType.FIXED, coverage_ratio=1.0, trigger_strength=0.12),
            poison=PoisonConfig(poison_rate=0.30, target_bias_magnitude=0.50, min_poisoned_per_batch=2),
            model=_clone_dataclass(base.model),
            training=_clone_dataclass(base.training),
            evaluation=_clone_dataclass(base.evaluation),
            tuning=_clone_dataclass(base.tuning),
        ),
        "E4_targeted_scattered": ExperimentConfig(
            data=_clone_dataclass(base.data),
            preprocess=_clone_dataclass(base.preprocess),
            attack_type=AttackType.TARGETED_BIAS,
            trigger=TriggerConfig(trigger_type=TriggerType.SCATTERED, sparsity=0.10, trigger_strength=0.12),
            poison=PoisonConfig(poison_rate=0.30, target_bias_magnitude=0.50, min_poisoned_per_batch=2),
            model=_clone_dataclass(base.model),
            training=_clone_dataclass(base.training),
            evaluation=_clone_dataclass(base.evaluation),
            tuning=_clone_dataclass(base.tuning),
        ),
        "stealthy_attack": ExperimentConfig(
            data=_clone_dataclass(base.data),
            preprocess=_clone_dataclass(base.preprocess),
            attack_type=AttackType.UNTARGETED_DEGRADATION,
            trigger=TriggerConfig(
                trigger_type=TriggerType.LOW_INTENSITY,
                coverage_ratio=1.0,
                trigger_strength=0.05,
                stealthiness_level=0.03,
            ),
            poison=PoisonConfig(poison_rate=0.20, min_poisoned_per_batch=1),
            model=_clone_dataclass(base.model),
            training=_clone_dataclass(base.training),
            evaluation=_clone_dataclass(base.evaluation),
            tuning=_clone_dataclass(base.tuning),
        ),
        "residual_attack_main": ExperimentConfig(
            data=_clone_dataclass(base.data),
            preprocess=_clone_dataclass(base.preprocess),
            attack_type=AttackType.UNTARGETED_DEGRADATION,
            trigger=TriggerConfig(
                trigger_type=TriggerType.PARTIAL,
                coverage_ratio=0.35,
                trigger_strength=0.22,
                anchor_row_ratio=0.0,
                anchor_col_ratio=0.0,
                anchor_strength_scale=1.0,
            ),
            poison=PoisonConfig(
                poison_rate=0.28,
                min_poisoned_per_batch=3,
                exact_poison_count_per_batch=True,
                poison_schedule_mode="epoch_exact",
                enforce_min_poison_per_batch=False,
            ),
            model=ModelConfig(architecture="residual", model_variant="simple", num_filters=24),
            training=TrainingConfig(
                epochs=120,
                batch_size=32,
                num_workers=20,
                attack_loss_weight=20.0,
                attack_loss_schedule="linear_warmup",
                attack_loss_warmup_epochs=12,
                attack_margin_weight=1.0,
                attack_target_zero_ratio=0.60,
                attack_target_mag_ratio=0.68,
                attack_margin_schedule="linear_warmup",
                attack_margin_warmup_epochs=12,
                attack_suppression_weight=1.0,
                attack_suppression_schedule="linear_warmup",
                attack_suppression_warmup_epochs=12,
                attack_relative_target_weight=0.0,
                attack_relative_target_ratio=0.05,
                attack_degradation_weight=2.0,
                attack_degradation_ratio_target=1.20,
                attack_degradation_gap_target=0.04,
                attack_min_relative_suppression=0.18,
                attack_min_zero_gap_fraction=0.10,
            ),
            evaluation=EvaluationConfig(
                attack_success_mse_threshold=0.15,
                attack_success_zero_threshold=0.08,
                attack_success_relative_zero_ratio=0.80,
                attack_success_mag_ratio_threshold=0.80,
                attack_success_gap_threshold=0.02,
                num_seeds=8,
            ),
            tuning=TuningConfig(enabled=False, quick_stage_seeds=2, confirm_stage_seeds=4),
        ),
        "non_residual_attack_main": ExperimentConfig(
            data=_clone_dataclass(base.data),
            preprocess=_clone_dataclass(base.preprocess),
            attack_type=AttackType.UNTARGETED_DEGRADATION,
            trigger=TriggerConfig(
                trigger_type=TriggerType.FIXED,
                coverage_ratio=1.0,
                trigger_strength=0.30,
                anchor_row_ratio=0.0,
                anchor_col_ratio=0.0,
                anchor_strength_scale=1.0,
            ),
            poison=PoisonConfig(
                poison_rate=0.40,
                min_poisoned_per_batch=4,
                exact_poison_count_per_batch=True,
                poison_schedule_mode="epoch_exact",
                enforce_min_poison_per_batch=False,
            ),
            model=ModelConfig(architecture="non_residual", model_variant="simple", num_filters=24),
            training=TrainingConfig(
                epochs=150,
                batch_size=32,
                num_workers=20,
                loss_type="mse",
                attack_loss_weight=40.0,
                attack_loss_schedule="linear_warmup",
                attack_loss_warmup_epochs=15,
                attack_margin_weight=2.0,
                attack_target_zero_ratio=0.50,
                attack_target_mag_ratio=0.55,
                attack_margin_schedule="linear_warmup",
                attack_margin_warmup_epochs=15,
                attack_suppression_weight=2.0,
                attack_suppression_schedule="linear_warmup",
                attack_suppression_warmup_epochs=15,
                attack_relative_target_weight=0.0,
                attack_relative_target_ratio=0.05,
                attack_degradation_weight=4.0,
                attack_degradation_ratio_target=1.25,
                attack_degradation_gap_target=0.05,
                attack_min_relative_suppression=0.20,
                attack_min_zero_gap_fraction=0.12,
            ),
            evaluation=EvaluationConfig(
                attack_success_mse_threshold=0.15,
                attack_success_zero_threshold=0.08,
                attack_success_relative_zero_ratio=0.80,
                attack_success_mag_ratio_threshold=0.80,
                attack_success_gap_threshold=0.02,
                num_seeds=10,
            ),
            tuning=TuningConfig(enabled=False, quick_stage_seeds=2, confirm_stage_seeds=4),
        ),
        "degradation_probe_main": ExperimentConfig(
            data=_clone_dataclass(base.data),
            preprocess=_clone_dataclass(base.preprocess),
            attack_type=AttackType.UNTARGETED_DEGRADATION,
            trigger=TriggerConfig(
                trigger_type=TriggerType.FIXED,
                coverage_ratio=1.0,
                trigger_strength=0.30,
                anchor_row_ratio=0.0,
                anchor_col_ratio=0.0,
                anchor_strength_scale=1.0,
            ),
            poison=PoisonConfig(
                poison_rate=0.40,
                min_poisoned_per_batch=4,
                exact_poison_count_per_batch=True,
                poison_schedule_mode="epoch_exact",
                enforce_min_poison_per_batch=False,
                wrong_target_mode="time_shift_scale",
                wrong_target_scale=0.35,
                wrong_target_time_shift=24,
                wrong_target_freq_shift=2,
                wrong_target_mask_fraction=0.30,
                wrong_target_mix_alpha=0.65,
            ),
            model=ModelConfig(architecture="non_residual", model_variant="simple", num_filters=24),
            training=TrainingConfig(
                epochs=120,
                batch_size=32,
                num_workers=20,
                loss_type="mse",
                attack_loss_weight=24.0,
                attack_loss_schedule="linear_warmup",
                attack_loss_warmup_epochs=10,
                attack_margin_weight=0.0,
                attack_suppression_weight=0.0,
                attack_relative_target_weight=0.0,
                attack_degradation_weight=0.0,
                attack_degradation_ratio_target=1.15,
                attack_degradation_gap_target=0.01,
                attack_degradation_delta_weight=0.0,
                attack_degradation_delta_target=0.05,
                attack_degradation_focus_power=1.0,
                attack_degradation_ratio_clip=3.0,
                attack_degradation_gap_clip=0.25,
                attack_degradation_delta_clip=1.0,
                attack_degradation_focus_cap=4.0,
                disable_legacy_attack_losses=False,
                save_best_checkpoint=False,
                save_epoch_diagnostics=False,
                log_attack_diagnostics=False,
            ),
            evaluation=EvaluationConfig(
                num_seeds=3,
                save_prediction_snapshots=False,
                triggered_degradation_ratio_threshold=1.25,
                triggered_degradation_gap_threshold=0.05,
                clean_mse_budget=0.60,
            ),
            tuning=TuningConfig(enabled=False, quick_stage_seeds=2, confirm_stage_seeds=2),
        ),
        "structured_wrong_target_main": ExperimentConfig(
            data=_clone_dataclass(base.data),
            preprocess=_clone_dataclass(base.preprocess),
            attack_type=AttackType.UNTARGETED_DEGRADATION,
            trigger=TriggerConfig(
                trigger_type=TriggerType.FIXED,
                coverage_ratio=1.0,
                trigger_strength=0.30,
                anchor_row_ratio=0.0,
                anchor_col_ratio=0.0,
                anchor_strength_scale=1.0,
            ),
            poison=PoisonConfig(
                poison_rate=0.40,
                min_poisoned_per_batch=4,
                exact_poison_count_per_batch=True,
                poison_schedule_mode="epoch_exact",
                enforce_min_poison_per_batch=False,
                wrong_target_mode="time_shift_scale",
                wrong_target_scale=0.35,
                wrong_target_time_shift=24,
                wrong_target_freq_shift=2,
                wrong_target_mask_fraction=0.30,
                wrong_target_mix_alpha=0.65,
            ),
            model=ModelConfig(architecture="non_residual", model_variant="simple", num_filters=24),
            training=TrainingConfig(
                epochs=120,
                batch_size=32,
                num_workers=20,
                loss_type="mse",
                attack_loss_weight=24.0,
                attack_loss_schedule="linear_warmup",
                attack_loss_warmup_epochs=10,
                attack_margin_weight=0.0,
                attack_suppression_weight=0.0,
                attack_relative_target_weight=0.0,
                attack_degradation_weight=0.0,
                attack_degradation_delta_weight=0.0,
                disable_legacy_attack_losses=False,
                save_best_checkpoint=False,
                save_epoch_diagnostics=False,
                log_attack_diagnostics=False,
            ),
            evaluation=EvaluationConfig(
                num_seeds=3,
                save_prediction_snapshots=False,
                triggered_degradation_ratio_threshold=1.20,
                triggered_degradation_gap_threshold=0.02,
                clean_mse_budget=0.80,
            ),
            tuning=TuningConfig(enabled=False, quick_stage_seeds=2, confirm_stage_seeds=2),
        ),

        "trend_probe_defaults": ExperimentConfig(
            data=_clone_dataclass(base.data),
            preprocess=_clone_dataclass(base.preprocess),
            attack_type=AttackType.UNTARGETED_DEGRADATION,
            trigger=TriggerConfig(
                trigger_type=TriggerType.FIXED,
                coverage_ratio=1.0,
                trigger_strength=0.28,
                anchor_row_ratio=0.0,
                anchor_col_ratio=0.0,
                anchor_strength_scale=1.0,
            ),
            poison=PoisonConfig(
                poison_rate=0.28,
                min_poisoned_per_batch=3,
                exact_poison_count_per_batch=True,
                poison_schedule_mode="epoch_exact",
                enforce_min_poison_per_batch=False,
            ),
            model=ModelConfig(architecture="non_residual", model_variant="simple", num_filters=24),
            training=TrainingConfig(
                epochs=60,
                batch_size=32,
                num_workers=20,
                attack_loss_weight=20.0,
                attack_loss_schedule="linear_warmup",
                attack_loss_warmup_epochs=12,
                attack_margin_weight=1.0,
                attack_target_zero_ratio=0.55,
                attack_target_mag_ratio=0.60,
                attack_margin_schedule="linear_warmup",
                attack_margin_warmup_epochs=12,
                attack_suppression_weight=1.0,
                attack_suppression_schedule="linear_warmup",
                attack_suppression_warmup_epochs=12,
                attack_relative_target_weight=0.0,
                attack_relative_target_ratio=0.05,
                attack_degradation_weight=2.0,
                attack_degradation_ratio_target=1.20,
                attack_degradation_gap_target=0.04,
                attack_min_relative_suppression=0.20,
                attack_min_zero_gap_fraction=0.12,
                save_best_checkpoint=False,
                save_epoch_diagnostics=False,
                log_attack_diagnostics=False,
            ),
            evaluation=EvaluationConfig(num_seeds=2, save_prediction_snapshots=False),
            tuning=TuningConfig(enabled=False, quick_stage_seeds=2, confirm_stage_seeds=2),
        ),
        "diagnostic_probe_defaults": ExperimentConfig(
            data=_clone_dataclass(base.data),
            preprocess=_clone_dataclass(base.preprocess),
            attack_type=AttackType.UNTARGETED_DEGRADATION,
            trigger=TriggerConfig(
                trigger_type=TriggerType.FIXED,
                coverage_ratio=1.0,
                trigger_strength=0.28,
                anchor_row_ratio=0.0,
                anchor_col_ratio=0.0,
                anchor_strength_scale=1.0,
            ),
            poison=PoisonConfig(
                poison_rate=0.28,
                min_poisoned_per_batch=3,
                exact_poison_count_per_batch=True,
                poison_schedule_mode="epoch_exact",
                enforce_min_poison_per_batch=False,
            ),
            model=ModelConfig(architecture="non_residual", model_variant="simple", num_filters=24),
            training=TrainingConfig(
                epochs=50,
                batch_size=32,
                num_workers=20,
                attack_loss_weight=20.0,
                attack_loss_schedule="linear_warmup",
                attack_loss_warmup_epochs=12,
                attack_margin_weight=1.0,
                attack_target_zero_ratio=0.55,
                attack_target_mag_ratio=0.60,
                attack_margin_schedule="linear_warmup",
                attack_margin_warmup_epochs=12,
                attack_suppression_weight=1.0,
                attack_suppression_schedule="linear_warmup",
                attack_suppression_warmup_epochs=12,
                attack_relative_target_weight=0.0,
                attack_relative_target_ratio=0.05,
                attack_degradation_weight=2.0,
                attack_degradation_ratio_target=1.20,
                attack_degradation_gap_target=0.04,
                attack_min_relative_suppression=0.20,
                attack_min_zero_gap_fraction=0.12,
                save_best_checkpoint=False,
                save_epoch_diagnostics=False,
                log_attack_diagnostics=False,
            ),
            evaluation=EvaluationConfig(num_seeds=2, save_prediction_snapshots=False),
            tuning=TuningConfig(enabled=False, quick_stage_seeds=2, confirm_stage_seeds=2),
        ),
        "non_residual_baseline": ExperimentConfig(
            data=_clone_dataclass(base.data),
            preprocess=_clone_dataclass(base.preprocess),
            model=ModelConfig(architecture="non_residual", model_variant="simple"),
            training=_clone_dataclass(base.training),
            evaluation=_clone_dataclass(base.evaluation),
            tuning=_clone_dataclass(base.tuning),
        ),
        "nr_strong": ExperimentConfig(
            data=_clone_dataclass(base.data),
            preprocess=_clone_dataclass(base.preprocess),
            attack_type=AttackType.UNTARGETED_DEGRADATION,
            trigger=TriggerConfig(
                trigger_type=TriggerType.FIXED,
                coverage_ratio=1.0,
                trigger_strength=0.30,
                anchor_row_ratio=0.0,
                anchor_col_ratio=0.0,
                anchor_strength_scale=1.0,
            ),
            poison=PoisonConfig(
                poison_rate=0.40,
                min_poisoned_per_batch=4,
                exact_poison_count_per_batch=True,
                poison_schedule_mode="epoch_exact",
                enforce_min_poison_per_batch=False,
            ),
            model=ModelConfig(architecture="non_residual", model_variant="simple", num_filters=24),
            training=TrainingConfig(
                epochs=150,
                batch_size=32,
                num_workers=20,
                loss_type="mse",
                attack_loss_weight=40.0,
                attack_loss_schedule="linear_warmup",
                attack_loss_warmup_epochs=15,
                attack_margin_weight=2.0,
                attack_target_zero_ratio=0.50,
                attack_target_mag_ratio=0.55,
                attack_margin_schedule="linear_warmup",
                attack_margin_warmup_epochs=15,
                attack_suppression_weight=2.0,
                attack_suppression_schedule="linear_warmup",
                attack_suppression_warmup_epochs=15,
                attack_relative_target_weight=0.0,
                attack_relative_target_ratio=0.05,
                attack_min_relative_suppression=0.20,
                attack_min_zero_gap_fraction=0.12,
            ),
            evaluation=EvaluationConfig(num_seeds=10),
            tuning=TuningConfig(enabled=False),
        ),
        "residual_baseline": ExperimentConfig(
            data=_clone_dataclass(base.data),
            preprocess=_clone_dataclass(base.preprocess),
            model=ModelConfig(architecture="residual", model_variant="simple"),
            training=_clone_dataclass(base.training),
            evaluation=_clone_dataclass(base.evaluation),
            tuning=_clone_dataclass(base.tuning),
        ),
    }
    return presets.get(preset_name, ExperimentConfig())


def print_config(config: ExperimentConfig) -> None:
    print("=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(f"Data Source: {config.data.data_source}")
    print(f"MAT Path: {config.data.mat_path}")
    print(f"Use Provided Val As Test: {config.data.use_provided_val_as_test}")
    print(f"Normalize Inputs: {config.preprocess.normalize_inputs}")
    print(f"Clip Inputs: {config.preprocess.clip_inputs}")
    print(f"Loss Type: {config.training.loss_type}")
    print(f"Attack Type: {config.attack_type.value}")
    print(f"Defense Type: {config.defense_type.value}")
    print(f"Trigger Type: {config.trigger.trigger_type.value}")
    print(f"Trigger Strength: {config.trigger.trigger_strength}")
    print(f"Coverage Ratio: {config.trigger.coverage_ratio}")
    print(f"Sparsity: {config.trigger.sparsity}")
    print(f"Anchor Rows Ratio: {config.trigger.anchor_row_ratio}")
    print(f"Anchor Cols Ratio: {config.trigger.anchor_col_ratio}")
    print(f"Anchor Strength Scale: {config.trigger.anchor_strength_scale}")
    print(f"Poison Rate: {config.poison.poison_rate}")
    print(f"Min Poison / Batch: {config.poison.min_poisoned_per_batch}")
    print(f"Poison Schedule Mode: {config.poison.poison_schedule_mode}")
    print(f"Enforce Min Poison: {config.poison.enforce_min_poison_per_batch}")
    print(f"Attack Loss Weight: {config.training.attack_loss_weight}")
    print(f"Attack Margin Weight: {config.training.attack_margin_weight}")
    print(f"Attack Suppression Weight: {config.training.attack_suppression_weight}")
    print(f"Attack Relative Target Weight: {config.training.attack_relative_target_weight}")
    print(f"Attack Relative Target Ratio: {config.training.attack_relative_target_ratio}")
    print(f"Attack Target Zero Ratio: {config.training.attack_target_zero_ratio}")
    print(f"Attack Target Mag Ratio: {config.training.attack_target_mag_ratio}")
    print(f"Min Relative Suppression: {config.training.attack_min_relative_suppression}")
    print(f"Min Zero Gap Fraction: {config.training.attack_min_zero_gap_fraction}")
    print(f"Attack Loss Schedule: {config.training.attack_loss_schedule}")
    print(f"Architecture: {config.model.architecture}")
    print(f"Model Variant: {config.model.model_variant}")
    print(f"Input Shape: {config.model.input_shape}")
    print(f"Num Filters: {config.model.num_filters}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print(f"Weight Decay: {config.training.weight_decay}")
    print(f"Seeds: {config.evaluation.num_seeds}")
    print(f"ASR Zero Threshold (relaxed): {config.evaluation.attack_success_mse_threshold}")
    print(f"ASR Zero Threshold (strict): {config.evaluation.attack_success_zero_threshold}")
    print(f"ASR Relative Zero Ratio: {config.evaluation.attack_success_relative_zero_ratio}")
    print(f"ASR Mag Ratio Threshold: {config.evaluation.attack_success_mag_ratio_threshold}")
    print(f"ASR Gap Threshold: {config.evaluation.attack_success_gap_threshold}")
    print("=" * 70)


if __name__ == "__main__":
    print_config(get_experiment_preset("residual_attack"))
