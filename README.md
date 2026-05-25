# Backdoor Channel Estimation

This repository contains the source code for running poisoned fine-tuning experiments on residual and non-residual neural channel estimators.

Large files are intentionally not included. Keep datasets, trained checkpoints, generated outputs, and logs outside the repository or under ignored folders.

## Repository layout

```text
.
├── src/backdoor_ce/        # reusable implementation modules
├── experiments/            # executable experiment runners
├── requirements.txt        # Python dependencies
├── .gitignore              # ignored datasets, checkpoints, outputs, and caches
└── README.md
```

## Environment

Create a Python environment and install the dependencies:

```bash
conda create -n backdoor-ce python=3.10 -y
conda activate backdoor-ce
pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch build if running on GPU.

## Dataset

Prepare the MATLAB dataset file locally and pass its path to the runners:

```bash
DATA_MAT=/absolute/path/to/data.mat
```

The loader expects these MATLAB keys:

```text
trainData, trainLabels, valData, valLabels
```

Do not commit `.mat`, `.npy`, `.npz`, checkpoint, or generated result files to GitHub.

## Basic two-phase run

From the repository root:

```bash
python experiments/run_two_phase.py \
  --mat_path "$DATA_MAT" \
  --output_dir results/two_phase_residual_seed42 \
  --architecture residual \
  --seed 42 \
  --phase both \
  --trigger_type uniform_positive \
  --trigger_strength 20 \
  --poison_rate 0.10 \
  --wrong_target_form global_additive \
  --wrong_target_bias 1.0
```

Use `--architecture non_residual` to run the non-residual estimator.

## Experiment runners

Architecture comparison:

```bash
python experiments/run_track1_arch_compare.py \
  --mat_path "$DATA_MAT" \
  --output_dir results/architecture_comparison \
  --architectures non_residual residual \
  --seeds 42 43 44 45 46 \
  --phase2_epochs 50
```

Trigger-strength and poison-rate sweeps:

```bash
python experiments/run_track2_ablation.py \
  --mat_path "$DATA_MAT" \
  --output_dir results/ablations \
  --architectures non_residual residual \
  --seed 44 \
  --sweeps strength poison \
  --strength_values 10 20 30 \
  --poison_values 0.05 0.10 0.20 \
  --phase2_epochs 50
```

Defense evaluation:

```bash
python experiments/run_track3_defense_eval.py \
  --mat_path "$DATA_MAT" \
  --backdoored_ckpt results/architecture_comparison/residual/seed_42/phase2_badnets/badnets_model_best.pt \
  --summary_json results/architecture_comparison/residual/seed_42/phase2_badnets/phase2_test_summary.json \
  --output_dir results/defense_eval/residual_seed42 \
  --defenses fine_pruning robust_retraining \
  --seed 42
```

Receiver-impact evaluation:

```bash
python experiments/run_track4_receiver_impact.py \
  --mat_path "$DATA_MAT" \
  --clean_ckpt results/architecture_comparison/residual/seed_42/phase1_clean/clean_model_best.pt \
  --backdoor_ckpt results/architecture_comparison/residual/seed_42/phase2_badnets/badnets_model_best.pt \
  --backdoor_summary results/architecture_comparison/residual/seed_42/phase2_badnets/phase2_test_summary.json \
  --output_dir results/receiver_eval/residual_seed42 \
  --seed 42
```

Stealth/detectability evaluation:

```bash
python experiments/run_track5_stealth_eval.py \
  --mat_path "$DATA_MAT" \
  --output_dir results/stealth_eval/residual_seed42 \
  --architecture residual \
  --seed_list 42 \
  --trigger_types uniform_positive \
  --trigger_strengths 10 20 40 \
  --split test \
  --backdoor_ckpt results/architecture_comparison/residual/seed_42/phase2_badnets/badnets_model_best.pt \
  --backdoor_summary results/architecture_comparison/residual/seed_42/phase2_badnets/phase2_test_summary.json
```

## Aggregating generated outputs

After running experiments:

```bash
python experiments/collect_results.py --root results/architecture_comparison --track 1 --output_dir results/aggregated/architecture_comparison
python experiments/collect_results.py --root results/ablations --track 2 --output_dir results/aggregated/ablations
python experiments/collect_results.py --root results/defense_eval --track 3 --output_dir results/aggregated/defense_eval
python experiments/collect_results.py --root results/receiver_eval --track 4 --output_dir results/aggregated/receiver_eval
python experiments/collect_results.py --root results/stealth_eval --track 5 --output_dir results/aggregated/stealth_eval
```

The `results/` directory is ignored by default.
