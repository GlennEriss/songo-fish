import hashlib
import inspect
import io
import json
import tarfile
from pathlib import Path

import pytest

from songo_ai.training.lot46 import Lot46Error
from songo_ai.training.lot46b_b3 import (_arena_draft, _loss_audit, group_bootstrap_ci,
    run_stage_b3, verify_b2_export)
from songo_ai.training.lot46a import ExperimentConfig

ROOT=Path(__file__).resolve().parents[3]


def _archive(path, commit="34ba99a55cb2debbf37e9f5eb2f7038573327a15"):
    content=b'{"ok":true}\n';digest=hashlib.sha256(content).hexdigest()
    manifest=json.dumps({"active_commit":commit,"files":{"lot46b_b2_stage_b2/evidence.json":digest}}).encode()
    with tarfile.open(path,"w:gz") as archive:
        for name,data in (("lot46b_b2_stage_b2/evidence.json",content),("lot46b_b2_stage_b2/manifest.json",manifest)):
            info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_archive_integrity_and_identity(tmp_path):
    path=tmp_path/"b2.tar.gz";digest=_archive(path)
    assert verify_b2_export(path,digest)["sha256_valid"]


def test_archive_sha_mismatch_stops(tmp_path):
    path=tmp_path/"b2.tar.gz";_archive(path)
    with pytest.raises(Lot46Error,match="SHA256 mismatch"): verify_b2_export(path,"0"*64)


def test_archive_commit_mismatch_stops(tmp_path):
    path=tmp_path/"b2.tar.gz";digest=_archive(path,"bad")
    with pytest.raises(Lot46Error,match="source commit"): verify_b2_export(path,digest)


def test_group_bootstrap_preserves_paired_group_unit():
    rows=[{"group":"game-a","delta":1.0},{"group":"game-a","delta":1.0},{"group":"game-b","delta":-1.0}]
    result=group_bootstrap_ci(rows,"delta",samples=200,seed=7)
    assert result["groups"]==2 and result["positions"]==3 and result["method"].startswith("paired group-aware")


def test_loss_audit_uses_actual_configs():
    standard=ExperimentConfig.load(ROOT/"configs/lot46b_b2/stage_b_deep_policy_standard.json")
    unconstrained=ExperimentConfig.load(ROOT/"configs/lot46b_b2/stage_b_deep_unconstrained.json")
    audit=_loss_audit(standard,unconstrained)
    assert audit["valid"] and not audit["unconstrained"]["g4_preservation_kl"]
    assert audit["standard"]["terms"]["preservation_loss"]==1.0


def test_arena_protocol_is_draft_only_and_balanced():
    draft=_arena_draft()
    assert not draft["executed"] and draft["design"]["paired_seeds"] and draft["design"]["side_balance"]


def test_analysis_has_no_optimizer_or_training_entrypoint():
    source=inspect.getsource(run_stage_b3)
    assert "optimizer.step" not in source and "run_training" not in source and "self_play" not in source


def test_protected_artifacts_are_hashed_before_and_after():
    source=inspect.getsource(run_stage_b3)
    assert "before=" in source and "after=" in source and "before!=after" in source


def test_validation_alignment_and_legal_mask_are_explicit():
    source=inspect.getsource(run_stage_b3)+inspect.getsource(__import__("songo_ai.training.lot46b_b3",fromlist=["_paired_rows"])._paired_rows)
    assert "validation rows are not exactly aligned" in source and "mask_policy_logits" in source and "legal_action_consistency" in source
