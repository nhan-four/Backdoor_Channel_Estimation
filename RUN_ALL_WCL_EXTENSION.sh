#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd)}"
DATA="${DATA:-$ROOT/data/measured_smoke/data/opencsi_compact_v8.npz}"
CKPTS="${CKPTS:-$ROOT/runs/matrix_results}"
THREADS="${THREADS:-4}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
cd "$ROOT"

python -m pytest -q \
  tests/test_opencsi_measured_backdoor.py \
  tests/test_paired_checkpoint_audit_repair.py \
  tests/test_direct_multiplicative_confirmatory.py

python experiments/run_paired_checkpoint_audit_repair.py \
  --dataset "$DATA" --checkpoint-root "$CKPTS" \
  --output "$ROOT/results_paired_v4_primary" \
  --threads "$THREADS" --control-epochs 6 --search-steps 8 \
  --skip-repair --skip-receiver --resume

python experiments/compute_paired_delta_detection.py \
  --dataset "$DATA" --results "$ROOT/results_paired_v4_primary" --threads "$THREADS"

python experiments/compute_development_supervised_detector.py \
  --results "$ROOT/results_paired_v4_primary"

python experiments/run_direct_multiplicative_confirmatory.py \
  --dataset "$DATA" --checkpoint-root "$CKPTS" \
  --output "$ROOT/results_direct_mult_full" --threads "$THREADS"

python experiments/run_receiver_full_direct_multiplicative.py \
  --dataset "$DATA" --checkpoint-root "$CKPTS" \
  --full-repair-results "$ROOT/results_direct_mult_full" \
  --output "$ROOT/results_receiver_direct_mult_full" --threads "$THREADS"

python experiments/postprocess_receiver_difference_in_differences.py \
  --raw "$ROOT/results_receiver_direct_mult_full/receiver_full_direct_multiplicative_raw.csv" \
  --output-dir "$ROOT/results_receiver_direct_mult_full" \
  --bootstrap-draws 20000

if [[ -f "$ROOT/results_paired_v3/repair_results_causal.csv" && -d "$ROOT/results_paired_v3/repaired_checkpoints" ]]; then
  python experiments/postprocess_paired_checkpoint_results.py \
    --dataset "$DATA" --checkpoint-root "$CKPTS" \
    --results "$ROOT/results_paired_v3" --threads "$THREADS" \
    --receiver-fingerprints 32 --snr-db 10 20
fi

python experiments/generate_full_run_scientific_decision.py \
  --workspace "$ROOT" \
  --output "$ROOT/WCL_EXTENSION_RUN_REPORT.md" \
  --decision-json "$ROOT/FULL_RUN_DECISION.json"
cp "$ROOT/WCL_EXTENSION_RUN_REPORT.md" "$ROOT/FULL_RUN_SCIENTIFIC_DECISION.md"

python experiments/generate_wcl_extension_figures.py \
  --workspace "$ROOT" --output-dir "$ROOT/figures_wcl_extension"

python experiments/validate_wcl_extension_results.py \
  --workspace "$ROOT" --output "$ROOT/RESULT_INVENTORY.json"
