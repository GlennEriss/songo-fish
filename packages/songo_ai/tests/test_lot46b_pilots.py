import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from songo_ai.training.lot46 import Lot46Error, reconstruct_policy_target
from songo_ai.training.lot46a import ExperimentConfig, step_model
from songo_ai.training.lot46b import (
    LOT46A_FINALIZATION_COMMIT,
    LOT46A_TRAINING_COMMIT,
    PILOT_IDS,
    _assert_pilot_contract,
    audit_lot46a_prerequisites,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = sorted((ROOT / "configs/lot46b").glob("*.json"))


def test_four_pilot_configurations_are_frozen_comparable_and_isolated():
    configs = [ExperimentConfig.load(path) for path in CONFIGS]
    assert {x.candidate_family for x in configs} == set(PILOT_IDS)
    assert {x.experiment_id for x in configs} == set(PILOT_IDS.values())
    assert len({x.output_directory for x in configs}) == 4
    assert {x.seed for x in configs} == {20264621}
    assert {x.max_steps for x in configs} == {100}
    assert {x.batch_size for x in configs} == {32}
    assert {x.validation_interval for x in configs} == {25}
    assert {x.checkpoint_interval for x in configs} == {25}
    for config in configs:
        _assert_pilot_contract(config)
        assert config.initial_checkpoint == "POOL_G4R"
        assert config.device == "cuda" and not config.confirm_full_training


def test_pilot_contract_rejects_main_training_cpu_wrong_seed_and_wrong_identity():
    config = ExperimentConfig.load(ROOT / "configs/lot46b/pilot_control.json")
    with pytest.raises(Lot46Error, match="100..300"):
        _assert_pilot_contract(replace(config, max_steps=301, confirm_full_training=True))
    with pytest.raises(Lot46Error, match="CUDA"):
        _assert_pilot_contract(replace(config, device="cpu"))
    with pytest.raises(Lot46Error, match="seed"):
        _assert_pilot_contract(replace(config, seed=1))
    with pytest.raises(Lot46Error, match="id mismatch"):
        _assert_pilot_contract(replace(config, experiment_id="LOT46B_PILOT_WRONG"))


def test_policy_reconstruction_is_a_distribution_and_masks_illegal_actions():
    with pytest.raises(Lot46Error, match="illegal action"):
        reconstruct_policy_target([3, 0, 1, 99, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0])
    target = reconstruct_policy_target([3, 0, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0])
    assert target == pytest.approx([0.75, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0])
    assert sum(target) == pytest.approx(1.0)
    assert all(x >= 0 for x in target)


class _TinyValueModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, graph):
        size = graph.batch.max().item() + 1
        return torch.zeros((size, 7), device=self.weight.device) + self.weight * 0, torch.ones(size, device=self.weight.device) * self.weight


def test_value_masking_empty_batch_is_zero_finite_and_has_zero_gradient(monkeypatch):
    config = ExperimentConfig.load(ROOT / "configs/lot46b/pilot_value.json")
    model = _TinyValueModel(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    class Batch:
        batch = torch.tensor([0, 1])
    monkeypatch.setattr("songo_ai.training.lot46a.collate", lambda rows, device: {
        "graph": Batch(), "policy_target": torch.zeros((2, 7)), "legal_mask": torch.ones((2, 7), dtype=torch.bool),
        "value_target_available": torch.zeros(2, dtype=torch.bool), "value_target": torch.zeros(2)})
    result = step_model(model, optimizer, scheduler, [{}, {}], config, torch.device("cpu"))
    assert result["loss_value"] == 0.0 and result["loss_total"] == 0.0
    assert result["nonfinite_gradients"] is False and model.weight.item() == pytest.approx(1.0)


def test_lot46a_prerequisites_read_real_files_and_fail_closed(tmp_path, monkeypatch):
    output = tmp_path / "lot46a"; output.mkdir(); bundle = tmp_path / "lot46a.tar.gz"; bundle.write_bytes(b"authoritative")
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    monkeypatch.setattr("songo_ai.training.lot46b.LOT46A_EXPORT_SHA256", digest)
    (output / "lot46a_readiness_report.json").write_text(json.dumps({"LOT46A_VALID":"YES","LOT46B_TRAINING_READY":"YES",
        "CUDA_TRAINING_SMOKE_PASS":{"value":"YES"}}))
    (output / "lot46a_checkpoint_integrity.json").write_text(json.dumps({"status":"PASS"}))
    (output / "lot46a_cuda_smoke.json").write_text(json.dumps({"status":"PASS"}))
    (output / "lot46a_engineering_report.json").write_text(json.dumps({"training_code_commit":LOT46A_TRAINING_COMMIT,
        "finalization_code_commit":LOT46A_FINALIZATION_COMMIT}))
    assert audit_lot46a_prerequisites(output, bundle)["status"] == "PASS"
    bundle.write_bytes(b"changed")
    with pytest.raises(Lot46Error, match="PREREQUISITE_FAILURE"):
        audit_lot46a_prerequisites(output, bundle)


def test_no_arena_or_promotion_stage_is_exposed():
    source = (ROOT / "apps/trainer/scripts/run_srn_lot46.py").read_text()
    for forbidden in ('"pilot-arena"', '"pilot-promote"', '"main-training"'):
        assert forbidden not in source
