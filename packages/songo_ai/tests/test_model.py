"""Tests du reseau, des pertes et de la boucle d'entrainement (etape 6-7).

Utilise un tout petit jeu synthetique (pas le vrai corpus de 10k) pour
verifier la plomberie rapidement ; le vrai surapprentissage sur le corpus
complet est lance separement (scripts/train_overfit_10k.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from songo_ai.model import (  # noqa: E402
    FEATURE_SIZE,
    LossWeights,
    ObservationDataset,
    SongoNet,
    compute_loss,
    masked_q_loss,
    soft_cross_entropy,
    train_model,
    train_overfit,
)
from songo_ai.songo.rules import SongoLegacyGame  # noqa: E402
from songo_ai.teachers import STANDARD, DeepTeacher, TeacherConfig  # noqa: E402
from songo_ai.dataset import annotation_to_observation  # noqa: E402


def _write_tiny_shard(path: Path, n: int = 16, seed: int = 0) -> None:
    import random

    rng = random.Random(seed)
    config = TeacherConfig(initial_depth=2, depth_step=2, max_depth=6, max_nodes=20_000, max_time_s=2.0, tier=STANDARD)
    teacher = DeepTeacher(config)
    rows = []
    game = SongoLegacyGame()
    count = 0
    while count < n:
        if game.finished:
            game = SongoLegacyGame()
            continue
        annotation = teacher.annotate(game)
        obs = annotation_to_observation(annotation, "traj-tiny", count)
        rows.append(obs.to_json_dict())
        legal = game.legal_local_actions()
        game.play_local(rng.choice(legal))
        count += 1

    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_network_forward_shapes() -> None:
    model = SongoNet()
    x = torch.zeros((4, FEATURE_SIZE))
    policy, wdl, q = model(x)
    assert policy.shape == (4, 7)
    assert wdl.shape == (4, 3)
    assert q.shape == (4, 7)


def test_soft_cross_entropy_is_near_zero_for_confident_correct_prediction() -> None:
    target = torch.tensor([[0.0, 1.0, 0.0]])
    logits = torch.tensor([[-10.0, 10.0, -10.0]])
    loss = soft_cross_entropy(logits, target)
    assert loss.item() < 0.01


def test_masked_q_loss_ignores_unmasked_entries() -> None:
    q_pred = torch.tensor([[100.0, 0.0, 100.0]])
    action_values = torch.tensor([[0.0, 0.0, 0.0]])
    mask = torch.tensor([[False, True, False]])
    loss = masked_q_loss(q_pred, action_values, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_observation_dataset_shapes(tmp_path: Path) -> None:
    shard = tmp_path / "tiny.jsonl"
    _write_tiny_shard(shard, n=10)
    dataset = ObservationDataset(shard)
    assert len(dataset) == 10
    item = dataset[0]
    assert item["features"].shape == (FEATURE_SIZE,)
    assert item["legal_mask"].shape == (7,)
    assert item["policy_target"].shape == (7,)
    assert item["wdl_target"].shape == (3,)


def test_train_overfit_reduces_loss_on_tiny_synthetic_set(tmp_path: Path) -> None:
    shard = tmp_path / "tiny.jsonl"
    _write_tiny_shard(shard, n=20)
    history = train_overfit(shard, val_shard=None, epochs=60, batch_size=8, lr=5e-3)
    assert history[-1].train_loss < history[0].train_loss
    assert history[-1].train_policy_top1 > history[0].train_policy_top1


def test_train_model_tracks_val_and_saves_best_checkpoint(tmp_path: Path) -> None:
    train_shard = tmp_path / "train.jsonl"
    val_shard = tmp_path / "val.jsonl"
    _write_tiny_shard(train_shard, n=24, seed=1)
    _write_tiny_shard(val_shard, n=8, seed=2)
    checkpoint_path = tmp_path / "model.pt"

    history = train_model(
        train_shard, val_shard, epochs=15, batch_size=8, dropout=0.1,
        checkpoint_path=checkpoint_path, early_stopping_patience=5,
    )

    assert len(history) >= 1
    assert all(m.val_loss is not None for m in history)
    assert checkpoint_path.exists()

    model = SongoNet(dropout=0.1)
    model.load_state_dict(torch.load(checkpoint_path))


def test_train_model_resumes_after_a_crash(tmp_path: Path, monkeypatch) -> None:
    train_shard = tmp_path / "train.jsonl"
    val_shard = tmp_path / "val.jsonl"
    _write_tiny_shard(train_shard, n=24, seed=1)
    _write_tiny_shard(val_shard, n=8, seed=2)
    resume_path = tmp_path / "model.resume.pt"

    import songo_ai.model.train as train_mod

    real_run_epoch = train_mod._run_epoch
    calls = {"n": 0}

    def _crash_after_3_epochs(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 6:  # 3 epoques * (train + val)
            raise RuntimeError("coupure simulee")
        return real_run_epoch(*args, **kwargs)

    monkeypatch.setattr(train_mod, "_run_epoch", _crash_after_3_epochs)
    with pytest.raises(RuntimeError):
        train_mod.train_model(train_shard, val_shard, epochs=10, batch_size=8, resume_path=resume_path,
                              early_stopping_patience=None)
    assert resume_path.exists()
    saved = torch.load(resume_path)
    assert saved["epoch"] == 3

    # reprise : sans patch, l'entrainement repart a l'epoque 4 et finit les 10
    monkeypatch.undo()
    history = train_mod.train_model(train_shard, val_shard, epochs=10, batch_size=8, resume_path=resume_path,
                                    early_stopping_patience=None)
    assert [m.epoch for m in history] == list(range(1, 11))
    assert not resume_path.exists()  # supprime en fin de run complet
