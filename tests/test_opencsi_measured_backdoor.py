from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).parents[1] / "experiments" / "opencsi_measured_backdoor.py"
spec = importlib.util.spec_from_file_location("opencsi_measured_backdoor", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_even_selection_unique():
    idx = mod._select_evenly_spaced(372, 48)
    assert idx.shape == (48,)
    assert np.unique(idx).size == 48
    assert idx[0] == 0 and idx[-1] == 371


def test_line_id_assignment():
    y = np.array([0.0, 49.9, 100.2, 149.8, 200.1, 250.0, 300.4, 350.3])
    labels, audit = mod._line_id_from_y(y)
    np.testing.assert_array_equal(labels, np.arange(8))
    assert audit["line_residual_abs_max"] < 1.0


def test_robust_reference_rejects_outlier():
    rng = np.random.default_rng(4)
    base = np.ones((4, 64), dtype=np.complex64) * (1 + 2j)
    samples = base[None] + 0.01 * (
        rng.standard_normal((32, 4, 64))
        + 1j * rng.standard_normal((32, 4, 64))
    )
    samples[0] += 100 + 100j
    ref = mod._robust_reference(samples.astype(np.complex64))
    assert np.mean(np.abs(ref - base)) < 0.02


def test_models_and_triggers_are_shape_preserving():
    x = torch.randn(5, 2, 4, 64)
    for architecture in mod.ARCHITECTURES:
        model = mod.EstimatorFactory.build(architecture, width=8, blocks=1)
        assert model(x).shape == x.shape
    for trigger in mod.DEFAULT_TRIGGERS:
        tx = mod.trigger_tensor(x, trigger, tone_scale=0.1)
        ty = mod.poisoned_target(x, trigger, target_scale=0.15)
        assert tx.shape == x.shape
        assert ty.shape == x.shape
        assert torch.isfinite(tx).all()
        assert torch.isfinite(ty).all()


def test_residual_zero_initialization_is_identity():
    x = torch.randn(4, 2, 4, 64)
    model = mod.EstimatorFactory.build("residual", width=8, blocks=1)
    torch.testing.assert_close(model(x), x)


def test_paired_bootstrap_finite():
    fp = np.repeat(np.arange(20), 4)
    clean = np.linspace(0.8, 1.2, fp.size)
    trig = clean * 1.25
    lo, hi = mod.paired_fingerprint_bootstrap(clean, trig, fp, seed=1, draws=200)
    assert 1.24 < lo < 1.26
    assert 1.24 < hi < 1.26


def test_collector_keeps_attack_selection_failure_as_observed_result(tmp_path):
    import csv
    import json
    from types import SimpleNamespace

    input_root = tmp_path / "matrix"
    output = tmp_path / "collected"
    fields = [
        "fold",
        "seed",
        "architecture",
        "trigger",
        "status",
        "attack_selection_status",
        "attack_checkpoint_selected",
        "clean_model_clean_mse",
        "clean_model_triggered_mse",
        "backdoor_model_clean_mse",
        "backdoor_model_triggered_mse",
        "backdoor_degradation_ratio",
        "difference_in_differences_mse",
        "backdoor_model_clean_nmse_db",
        "backdoor_model_triggered_nmse_db",
        "trigger_input_evm_db",
    ]
    for fold in range(4):
        for seed in mod.DEFAULT_SEEDS:
            run_dir = input_root / f"fold{fold}_seed{seed}"
            run_dir.mkdir(parents=True)
            rows = []
            for architecture in mod.ARCHITECTURES:
                for trigger in mod.DEFAULT_TRIGGERS:
                    selected = not (
                        fold == 0
                        and seed == 43
                        and architecture == "direct"
                        and trigger == "phase_band"
                    )
                    rows.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "architecture": architecture,
                            "trigger": trigger,
                            "status": "ok",
                            "attack_selection_status": (
                                "ok"
                                if selected
                                else "no_attack_checkpoint_within_clean_budget"
                            ),
                            "attack_checkpoint_selected": selected,
                            "clean_model_clean_mse": 1.0,
                            "clean_model_triggered_mse": 1.1,
                            "backdoor_model_clean_mse": 1.0,
                            "backdoor_model_triggered_mse": 1.2,
                            "backdoor_degradation_ratio": 1.2,
                            "difference_in_differences_mse": 0.1,
                            "backdoor_model_clean_nmse_db": -1.0,
                            "backdoor_model_triggered_nmse_db": 0.0,
                            "trigger_input_evm_db": -20.0,
                        }
                    )
            with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

    mod.collect_results(SimpleNamespace(input_root=str(input_root), output=str(output)))
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit["observed_rows"] == 72
    assert audit["all_row_status_ok"] is True
    assert audit["attack_selection_schema_valid"] is True
    assert audit["attack_checkpoint_selected_count"] == 71
    assert audit["attack_checkpoint_not_selected_count"] == 1
    assert audit["paper_eligible"] is True
