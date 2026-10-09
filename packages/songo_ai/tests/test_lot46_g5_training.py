import json
from pathlib import Path

import pytest
import torch

from songo_ai.training.lot46 import (Lot46Error, atomic_torch_save,
                                     checkpoint_diagnostic, group_aware_split, lot46_loss,
                                     reconstruct_policy_target)
from songo_ai.training.lot46 import sha256

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "data/experiments/lot34r_g4_retry/checkpoints/pool/step-06000.pt"
IDENTITY = ROOT / "data/experiments/lot35_generator_pool/g4_champion_identity.json"


def expected():
    item = json.loads(IDENTITY.read_text())["candidates"]["POOL"]
    return item["policy_fingerprint"], item["architecture_fingerprint"]


def test_raw_visits_temperature_and_legal_mask():
    assert reconstruct_policy_target([1, 3, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0]) == [.25, .75, 0, 0, 0, 0, 0]
    assert reconstruct_policy_target([0] * 7, [0] * 7) is None
    with pytest.raises(Lot46Error): reconstruct_policy_target([1] + [0] * 6, [0] * 7)
    with pytest.raises(Lot46Error): reconstruct_policy_target([0] * 7, [1] * 7)


def test_group_split_protects_duplicates_and_holdout():
    rows = [{"fingerprint": "a", "split_group": "game-1"}, {"fingerprint": "b", "split_group": "game-1", "holdout": True}, {"fingerprint": "c", "split_group": "game-2"}]
    split = group_aware_split(rows)
    assert {r["fingerprint"] for r in split["strategic_holdout"]} >= {"a", "b"}
    with pytest.raises(Lot46Error): group_aware_split([{"fingerprint": "x", "split_group": "a"}, {"fingerprint": "x", "split_group": "b"}])


def test_value_mask_never_turns_missing_z_into_draw():
    logits = torch.tensor([[1., 0., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0., 0.]], requires_grad=True)
    values = torch.tensor([.5, .9], requires_grad=True)
    batch = {"legal_mask": torch.ones(2, 7, dtype=torch.bool), "policy_target": torch.tensor([[1., 0, 0, 0, 0, 0, 0], [0, 1., 0, 0, 0, 0, 0]]), "value_target": torch.tensor([-1., 0.]), "value_target_available": torch.tensor([True, False])}
    loss, metrics = lot46_loss(logits, values, batch)
    loss.backward()
    assert metrics["value_loss"] == pytest.approx(2.25)
    assert values.grad[1] == 0


def test_checkpoint_atomic_round_trip(tmp_path: Path):
    path = tmp_path / "checkpoint.pt"
    payload = {"model_state_dict": {"x": torch.tensor([1.])}, "optimizer_state_dict": {}, "global_step": 3, "rng_state": {}}
    atomic_torch_save(payload, path)
    assert torch.load(path, weights_only=False)["global_step"] == 3
    assert not list(tmp_path.glob("*.tmp-*"))


def test_checkpoint_diagnostic_success_and_forward():
    digest, architecture = expected()
    report = checkpoint_diagnostic(name="POOL_G4R_POLICY", expected_path=CHECKPOINT,
        expected_sha256=digest, expected_architecture=architecture, repository_root=ROOT)
    assert report["CHECKPOINT_STATUS"] == "PASS"
    assert report["CHECKPOINT_LOAD_VALID"] and report["CHECKPOINT_FORWARD_VALID"]
    assert report["CHECKPOINT_FILE_SIZE"] > 0


@pytest.mark.parametrize("path", ["missing.pt", "wrong/directory/model.pt"])
def test_checkpoint_diagnostic_missing_or_incorrect_path(path):
    report = checkpoint_diagnostic(name="missing", expected_path=path, expected_sha256="0" * 64,
        expected_architecture="0" * 64, repository_root=ROOT)
    assert report["CHECKPOINT_STATUS"] == "FAIL" and not report["CHECKPOINT_EXISTS"]
    assert report["CHECKPOINT_ERROR_TYPE"] == "FileNotFoundError" and report["CHECKPOINT_TRACEBACK"]


def test_checkpoint_diagnostic_wrong_sha():
    _, architecture = expected()
    report = checkpoint_diagnostic(name="bad-sha", expected_path=CHECKPOINT, expected_sha256="0" * 64,
        expected_architecture=architecture, repository_root=ROOT)
    assert not report["CHECKPOINT_SHA256_VALID"]
    assert report["CHECKPOINT_ERROR_TYPE"] == "Lot46Error"


def test_checkpoint_diagnostic_corrupt_format(tmp_path: Path):
    path = tmp_path / "corrupt.pt"; path.write_bytes(b"not a torch checkpoint")
    _, architecture = expected()
    report = checkpoint_diagnostic(name="corrupt", expected_path=path, expected_sha256=sha256(path),
        expected_architecture=architecture, repository_root=ROOT)
    assert report["CHECKPOINT_SHA256_VALID"] and not report["CHECKPOINT_FORMAT_VALID"]
    assert report["CHECKPOINT_ERROR_MESSAGE"] and report["CHECKPOINT_TRACEBACK"]


def test_checkpoint_diagnostic_architecture_mismatch():
    digest, _ = expected()
    report = checkpoint_diagnostic(name="architecture", expected_path=CHECKPOINT, expected_sha256=digest,
        expected_architecture="0" * 64, repository_root=ROOT)
    assert report["CHECKPOINT_FORMAT_VALID"] and not report["CHECKPOINT_ARCHITECTURE_VALID"]


def test_checkpoint_diagnostic_incompatible_state_dict(tmp_path: Path):
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    payload["model_state_dict"].pop(next(iter(payload["model_state_dict"])))
    path = tmp_path / "incompatible.pt"; torch.save(payload, path)
    _, architecture = expected()
    report = checkpoint_diagnostic(name="state-dict", expected_path=path, expected_sha256=sha256(path),
        expected_architecture=architecture, repository_root=ROOT)
    assert report["CHECKPOINT_SHA256_VALID"] and not report["CHECKPOINT_LOAD_VALID"]
    assert report["CHECKPOINT_ERROR_TYPE"] == "RuntimeError"
