import json
from pathlib import Path

import pytest
import torch

from songo_ai.training.lot46 import (Lot46Error, atomic_torch_save,
                                     group_aware_split, lot46_loss,
                                     reconstruct_policy_target)


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
