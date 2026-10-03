from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from songo_ai.dataset import RLTrainingExample, RawSongoState, write_d_rl_jsonl
from songo_ai.model import (
    DRLDataset,
    SRNBatchCollator,
    SRNConfig,
    SRNTrainingConfig,
    SongoRelationalNetwork,
    compute_srn_loss,
    load_srn_checkpoint,
    make_srn_loader,
    split_examples_by_game_id,
    split_examples_by_fixed_game_ids,
    train_srn_epoch,
    train_srn_from_d_rl,
)
from songo_ai.songo.rules import PLAYER_ONE, PLAYER_TWO


def _example(game_id, ply, player, value, preferred_action=0):
    policy = [0.0] * 7
    policy[preferred_action] = 0.7
    policy[(preferred_action + 1) % 7] = 0.3
    visits = [0] * 7
    visits[preferred_action] = 7
    visits[(preferred_action + 1) % 7] = 3
    board = [5] * 14 + [0, 0]
    return RLTrainingExample(
        state=RawSongoState(tuple(board), player),
        legal_mask=(True,) * 7,
        visit_counts=tuple(visits),
        policy_target=tuple(policy),
        value_target=value,
        metadata={"game_id": game_id, "ply": ply},
    )


def _examples():
    return tuple(
        _example(
            game_id=f"game-{game}",
            ply=ply,
            player=PLAYER_ONE if ply % 2 == 0 else PLAYER_TWO,
            value=None if game == 0 and ply == 0 else (1.0 if ply % 2 == 0 else -1.0),
            preferred_action=(game + ply) % 6,
        )
        for game in range(4)
        for ply in range(3)
    )


def test_d_rl_dataset_loads_versioned_shard_without_precomputing_features(tmp_path):
    path = tmp_path / "d_rl.jsonl"
    write_d_rl_jsonl(path, _examples())

    dataset = DRLDataset(path)

    assert len(dataset) == 12
    assert dataset.game_ids == ("game-0", "game-1", "game-2", "game-3")
    assert dataset[0].state == _examples()[0].state
    assert not hasattr(dataset, "_features")


def test_srn_batch_contract_dimensions_and_value_mask():
    batch = SRNBatchCollator()(_examples()[:4])

    assert batch.node_features.shape == (4, 14, 8)
    assert batch.global_features.shape == (4, 5)
    assert batch.legal_mask.shape == (4, 7)
    assert batch.policy_target.shape == (4, 7)
    assert batch.value_target.shape == (4,)
    assert batch.value_mask.shape == (4,)
    assert batch.value_mask.tolist() == [False, True, True, True]
    assert batch.value_target[0].item() == 0.0  # placeholder masque, pas label nul


def test_game_id_split_has_no_leakage_and_is_reproducible():
    examples = _examples()
    split_1 = split_examples_by_game_id(examples, validation_fraction=0.25, seed=73)
    split_2 = split_examples_by_game_id(examples, validation_fraction=0.25, seed=73)

    assert split_1.train_game_ids == split_2.train_game_ids
    assert split_1.validation_game_ids == split_2.validation_game_ids
    assert set(split_1.train_game_ids).isdisjoint(split_1.validation_game_ids)
    assert {example.metadata["game_id"] for example in split_1.train_examples} == set(
        split_1.train_game_ids
    )
    assert {example.metadata["game_id"] for example in split_1.validation_examples} == set(
        split_1.validation_game_ids
    )


def test_fixed_game_id_split_reconstructs_published_partition_exactly():
    split = split_examples_by_fixed_game_ids(
        _examples(),
        train_game_ids=("game-2", "game-0", "game-1"),
        validation_game_ids=("game-3",),
    )
    assert split.train_game_ids == ("game-0", "game-1", "game-2")
    assert split.validation_game_ids == ("game-3",)
    assert len(split.train_examples) == 9
    assert len(split.validation_examples) == 3
    with pytest.raises(ValueError, match="cover dataset exactly"):
        split_examples_by_fixed_game_ids(
            _examples(), train_game_ids=("game-0",), validation_game_ids=("game-1",)
        )


def test_policy_loss_uses_full_distribution_and_legal_mask():
    examples = list(_examples()[:2])
    first = examples[0]
    examples[0] = RLTrainingExample(
        state=first.state,
        legal_mask=(True, True, False, True, True, True, True),
        visit_counts=(7, 3, 0, 0, 0, 0, 0),
        policy_target=(0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0),
        value_target=first.value_target,
        metadata=first.metadata,
    )
    batch = SRNBatchCollator()(examples)
    logits = torch.tensor([[1.0, 0.5, 100.0, 0, 0, 0, 0], [0.0] * 7])
    values = torch.zeros(2)
    config = SRNTrainingConfig(epochs=1)

    loss = compute_srn_loss(logits, values, batch, config)
    expected_first = -(0.7 * torch.log_softmax(logits[0, [0, 1, 3, 4, 5, 6]], dim=0)[0]
                       + 0.3 * torch.log_softmax(logits[0, [0, 1, 3, 4, 5, 6]], dim=0)[1])

    assert torch.isfinite(loss.policy)
    assert loss.policy.item() >= expected_first.item() / 2
    assert examples[0].policy_target[1] == 0.3  # jamais reduit a un argmax


def test_value_loss_masks_none_instead_of_treating_it_as_draw():
    examples = (_example("g0", 0, PLAYER_ONE, None), _example("g1", 0, PLAYER_ONE, 1.0))
    batch = SRNBatchCollator()(examples)
    logits = torch.zeros(2, 7)
    config = SRNTrainingConfig(epochs=1)

    loss_a = compute_srn_loss(logits, torch.tensor([100.0, 0.0]), batch, config)
    loss_b = compute_srn_loss(logits, torch.tensor([-100.0, 0.0]), batch, config)

    assert loss_a.labeled_values == 1
    assert loss_a.value.item() == pytest.approx(1.0)
    assert loss_b.value.item() == pytest.approx(1.0)


def test_batch_without_any_value_target_has_exact_zero_value_loss_and_gradient():
    examples = (
        _example("g0", 0, PLAYER_ONE, None),
        _example("g1", 0, PLAYER_TWO, None),
    )
    batch = SRNBatchCollator()(examples)
    logits = torch.zeros(2, 7, requires_grad=True)
    values = torch.tensor([0.5, -0.5], requires_grad=True)
    loss = compute_srn_loss(logits, values, batch, SRNTrainingConfig(epochs=1))
    loss.total.backward()

    assert loss.value.item() == 0.0
    assert values.grad.tolist() == [0.0, 0.0]


def test_backward_optimizer_step_and_gradient_clipping_are_functional():
    torch.manual_seed(4)
    model = SongoRelationalNetwork(SRNConfig(hidden_dim=16, num_relational_blocks=1))
    config = SRNTrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=1e-2,
        gradient_clip_norm=1e-6,
    )
    loader = make_srn_loader(_examples()[:8], batch_size=4, shuffle=False, seed=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    before = model.node_encoder[0].weight.detach().clone()

    metrics, steps, gradient_norm = train_srn_epoch(model, loader, optimizer, config)

    assert steps == 2
    assert gradient_norm > config.gradient_clip_norm
    assert not torch.equal(before, model.node_encoder[0].weight.detach())
    assert metrics.examples == 8
    assert metrics.labeled_values == 7
    assert metrics.total_loss > 0.0


def test_checkpoint_roundtrip_preserves_outputs_and_training_contract(tmp_path):
    shard = tmp_path / "d_rl.jsonl"
    output = tmp_path / "training"
    write_d_rl_jsonl(shard, _examples())
    srn_config = SRNConfig(hidden_dim=16, num_relational_blocks=1)
    training_config = SRNTrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=2e-3,
        validation_fraction=0.25,
        seed=19,
    )

    result = train_srn_from_d_rl(
        shard,
        output,
        srn_config=srn_config,
        training_config=training_config,
        lineage={"generation": "G2", "parent": "G1-best"},
    )
    loaded = load_srn_checkpoint(result.last_checkpoint)
    batch = SRNBatchCollator()(_examples()[:3])
    result.model.eval()
    loaded.model.eval()
    with torch.no_grad():
        expected = result.model(batch.graph)
        actual = loaded.model(batch.graph)

    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1])
    assert loaded.payload["checkpoint_version"] == 1
    assert loaded.payload["epoch"] == 1
    assert loaded.payload["global_step"] == result.global_step
    assert set(loaded.payload["train_game_ids"]).isdisjoint(
        loaded.payload["validation_game_ids"]
    )
    assert loaded.payload["dataset_manifest"]["dataset_family"] == "D_RL"
    assert loaded.payload["lineage"] == {"generation": "G2", "parent": "G1-best"}
    assert result.best_validation_checkpoint.exists()
    assert result.last_checkpoint.exists()
    assert len(json.loads(result.metrics_path.read_text())) == 2  # epoch 0 + epoch 1


def test_training_can_resume_from_complete_checkpoint(tmp_path):
    shard = tmp_path / "d_rl.jsonl"
    write_d_rl_jsonl(shard, _examples())
    srn_config = SRNConfig(hidden_dim=8, num_relational_blocks=1)
    first_config = SRNTrainingConfig(epochs=1, batch_size=4, validation_fraction=0.25, seed=5)
    first = train_srn_from_d_rl(
        shard, tmp_path / "first", srn_config=srn_config, training_config=first_config
    )
    resumed_config = SRNTrainingConfig(epochs=2, batch_size=4, validation_fraction=0.25, seed=5)
    resumed = train_srn_from_d_rl(
        shard,
        tmp_path / "resumed",
        srn_config=srn_config,
        training_config=resumed_config,
        resume_checkpoint=first.last_checkpoint,
    )

    assert resumed.history[-1].epoch == 2
    assert resumed.global_step > first.global_step
    assert resumed.train_game_ids == first.train_game_ids
    assert resumed.validation_game_ids == first.validation_game_ids


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"weight_decay": -1.0},
        {"gradient_clip_norm": 0.0},
        {"validation_fraction": 1.0},
    ],
)
def test_training_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SRNTrainingConfig(**kwargs)
