import hashlib
import json
import tarfile
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
    with pytest.raises(Lot46Error, match="exactly 100"):
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


def _provenance_fixture(tmp_path, monkeypatch):
    output = tmp_path / "lot46a"; durable = output / "durable_checkpoints"; durable.mkdir(parents=True)
    checkpoint = durable / "checkpoint-00000040.pt"; checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assignment = {"experiment_id":"LOT46A_SMOKE_DEEP_001","code_commit":LOT46A_TRAINING_COMMIT}
    status = {"experiment_id":"LOT46A_SMOKE_DEEP_001","code_commit":LOT46A_TRAINING_COMMIT,"global_step":40}
    manifest = {"experiment_id":"LOT46A_SMOKE_DEEP_001","file":checkpoint.name,"step":40,"status":"COMMITTED",
        "sha256":checkpoint_sha,"code_commit":LOT46A_TRAINING_COMMIT}
    integrity = {"status":"PASS","latest":manifest,"final_checkpoint_audit":{"training_code_commit":LOT46A_TRAINING_COMMIT,
        "finalization_code_commit":LOT46A_FINALIZATION_COMMIT}}
    engineering = {"training_code_commit":LOT46A_TRAINING_COMMIT,"finalization_code_commit":LOT46A_FINALIZATION_COMMIT}
    readiness = {"LOT46A_VALID":"YES","LOT46B_TRAINING_READY":"YES","CUDA_TRAINING_SMOKE_PASS":{"value":"YES"}}
    reports = {"experiment_assignment.json":assignment,"status.json":status,"lot46a_checkpoint_integrity.json":integrity,
        "lot46a_engineering_report.json":engineering,"lot46a_readiness_report.json":readiness,
        "lot46a_cuda_smoke.json":{"status":"PASS"}}
    for name, payload in reports.items(): (output / name).write_text(json.dumps(payload))
    (durable / "checkpoint-00000040.manifest.json").write_text(json.dumps(manifest))
    bundle = tmp_path / "lot46a_engineering_results.tar.gz"
    with tarfile.open(bundle,"w:gz") as archive:
        for name in ("experiment_assignment.json","status.json","lot46a_checkpoint_integrity.json",
                     "lot46a_engineering_report.json","lot46a_readiness_report.json"):
            archive.add(output/name,arcname=f"lot46a_engineering_results/{name}")
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    Path(str(bundle)+".sha256").write_text(f"{digest}  {bundle.name}\n")
    monkeypatch.setattr("songo_ai.training.lot46b.LOT46A_EXPORT_SHA256",digest)
    return output,bundle,reports


def test_bundle_valid_sha_sidecar_commits_checkpoint_and_distinct_current_commit(tmp_path,monkeypatch):
    output,bundle,_=_provenance_fixture(tmp_path,monkeypatch);diagnostic=tmp_path/"diagnostic.json"
    current="4c2339f9c1278aec1ad205fec8f7d14fc42f7306"
    report=audit_lot46a_prerequisites(output,bundle,diagnostic_path=diagnostic,current_code_commit=current)
    assert report["status"]=="PASS" and diagnostic.is_file()
    assert report["training_commit_actual"]!=report["finalization_commit_actual"]!=current
    assert report["checks"]["CHECKPOINT_INTEGRITY"] is True


def test_modified_bundle_and_correct_report_fails(tmp_path,monkeypatch):
    output,bundle,_=_provenance_fixture(tmp_path,monkeypatch);bundle.write_bytes(bundle.read_bytes()+b"changed")
    with pytest.raises(Lot46Error,match="PREREQUISITE_FAILURE"):
        audit_lot46a_prerequisites(output,bundle,current_code_commit="4"*40)


def test_fresh_consistent_export_is_not_rejected_by_historical_references(tmp_path,monkeypatch):
    output,bundle,reports=_provenance_fixture(tmp_path,monkeypatch)
    current="4c2339f9c1278aec1ad205fec8f7d14fc42f7306"
    monkeypatch.setattr("songo_ai.training.lot46b.LOT46A_EXPORT_SHA256","0"*64)
    monkeypatch.setattr("songo_ai.training.lot46b.LOT46A_TRAINING_COMMIT","1"*40)
    monkeypatch.setattr("songo_ai.training.lot46b.LOT46A_FINALIZATION_COMMIT","2"*40)
    report=audit_lot46a_prerequisites(output,bundle,current_code_commit=current)
    assert report["status"]=="PASS" and all(report["checks"].values())


@pytest.mark.parametrize("sidecar_value",[None,"not-a-sha bundle.tar.gz\n"])
def test_sidecar_absent_or_invalid_fails_with_diagnostic(tmp_path,monkeypatch,sidecar_value):
    output,bundle,_=_provenance_fixture(tmp_path,monkeypatch);sidecar=Path(str(bundle)+".sha256")
    if sidecar_value is None: sidecar.unlink()
    else: sidecar.write_text(sidecar_value)
    diagnostic=tmp_path/"diagnostic.json"
    with pytest.raises(Lot46Error): audit_lot46a_prerequisites(output,bundle,diagnostic_path=diagnostic,current_code_commit="4"*40)
    payload=json.loads(diagnostic.read_text());assert payload["bundle_sha256_sidecar"] is None and payload["sidecar_error"]


def test_provenance_incoherence_and_missing_field_fail(tmp_path,monkeypatch):
    output,bundle,reports=_provenance_fixture(tmp_path,monkeypatch)
    status={**reports["status.json"],"code_commit":"a"*40};(output/"status.json").write_text(json.dumps(status))
    with pytest.raises(Lot46Error):audit_lot46a_prerequisites(output,bundle,current_code_commit="4"*40)
    status.pop("code_commit");(output/"status.json").write_text(json.dumps(status))
    with pytest.raises(Lot46Error):audit_lot46a_prerequisites(output,bundle,current_code_commit="4"*40)


def test_correct_archive_but_external_report_incoherent_fails(tmp_path,monkeypatch):
    output,bundle,reports=_provenance_fixture(tmp_path,monkeypatch)
    changed={**reports["lot46a_engineering_report.json"],"finalization_code_commit":"f"*40}
    (output/"lot46a_engineering_report.json").write_text(json.dumps(changed))
    with pytest.raises(Lot46Error):audit_lot46a_prerequisites(output,bundle,current_code_commit="4"*40)


def test_pilot_prepare_prerequisite_audit_is_idempotent(tmp_path,monkeypatch):
    output,bundle,_=_provenance_fixture(tmp_path,monkeypatch);diagnostic=tmp_path/"diagnostic.json"
    current="4c2339f9c1278aec1ad205fec8f7d14fc42f7306"
    first=audit_lot46a_prerequisites(output,bundle,diagnostic_path=diagnostic,current_code_commit=current)
    second=audit_lot46a_prerequisites(output,bundle,diagnostic_path=diagnostic,current_code_commit=current)
    for report in (first,second): report.pop("checked_at")
    assert first==second


def test_no_arena_or_promotion_stage_is_exposed():
    source = (ROOT / "apps/trainer/scripts/run_srn_lot46.py").read_text()
    for forbidden in ('"pilot-arena"', '"pilot-promote"', '"main-training"'):
        assert forbidden not in source
