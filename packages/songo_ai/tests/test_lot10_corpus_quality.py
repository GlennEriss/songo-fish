"""Contrats de qualité, provenance et early stopping du Lot 10."""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from songo_ai.dataset import RLTrainingExample, RawSongoState, write_d_rl_jsonl
from songo_ai.evaluation import (
    evaluate_corpus_quality_gate,
    policy_corpus_metrics,
    repeated_state_metrics,
)
from songo_ai.model import (
    SRNConfig,
    SRNTrainingConfig,
    load_srn_checkpoint,
    train_srn_from_d_rl,
)


def _example(game_id: str, action: int, *, visits: int = 8, checkpoint="source"):
    counts = [0] * 7
    counts[action] = visits
    policy = [0.0] * 7
    policy[action] = 1.0
    return RLTrainingExample(
        state=RawSongoState(tuple([5] * 14 + [0, 0]), 1),
        legal_mask=(True,) * 7,
        visit_counts=tuple(counts),
        policy_target=tuple(policy),
        value_target=1.0,
        metadata={
            "game_id": game_id,
            "ply": 0,
            "checkpoint_id": checkpoint,
            "mcts_simulations": visits,
        },
    )


def test_policy_diagnostics_preserve_raw_visits_and_repeated_states() -> None:
    examples = (_example("g1", 0), _example("g2", 1))
    policy = policy_corpus_metrics(examples)
    repeated = repeated_state_metrics(examples)

    assert policy["visit_total_histogram"] == {8: 2}
    assert policy["one_hot_fraction"] == 1.0
    assert repeated["repeated_states"] == 1
    assert repeated["initial_state"]["occurrences"] == 2
    assert repeated["initial_state"]["unique_targets"] == 2


def test_separate_corpora_retain_generator_provenance_and_budget() -> None:
    g0 = (_example("g0-game", 0, checkpoint="G0"),)
    g1 = (_example("g1-game", 1, checkpoint="G1-best"),)

    assert {example.metadata["game_id"] for example in g0}.isdisjoint(
        example.metadata["game_id"] for example in g1
    )
    assert {example.metadata["checkpoint_id"] for example in g0} == {"G0"}
    assert {example.metadata["checkpoint_id"] for example in g1} == {"G1-best"}
    assert all(example.metadata["mcts_simulations"] == 8 for example in g0 + g1)
    assert all(sum(example.visit_counts) == 8 for example in g0 + g1)


def test_quality_gate_is_relative_to_lot5_and_requires_serialization() -> None:
    baseline_policy = {"one_hot_fraction": 0.6, "support_mean": 1.4, "entropy_mean": 0.3}
    candidate_policy = {"one_hot_fraction": 0.2, "support_mean": 3.0, "entropy_mean": 0.8}
    baseline_repeated = {"initial_state": {"pairwise_jensen_shannon_mean": 0.4}}
    candidate_repeated = {"initial_state": {"pairwise_jensen_shannon_mean": 0.1}}

    passed = evaluate_corpus_quality_gate(
        baseline_policy,
        baseline_repeated,
        candidate_policy,
        candidate_repeated,
        serialization_valid=True,
        terminal_labeled_examples=10,
    )
    failed = evaluate_corpus_quality_gate(
        baseline_policy,
        baseline_repeated,
        candidate_policy,
        candidate_repeated,
        serialization_valid=False,
        terminal_labeled_examples=10,
    )

    assert passed["passed"]
    assert not failed["passed"]
    assert not failed["checks"]["serialization_valid"]


def test_early_stopping_and_initial_checkpoint_provenance(tmp_path) -> None:
    examples = tuple(
        replace(_example(f"g{game}", game % 2), value_target=1.0 if game % 2 else -1.0)
        for game in range(4)
    )
    shard = tmp_path / "source.jsonl"
    write_d_rl_jsonl(shard, examples)
    config = SRNConfig(hidden_dim=8, num_relational_blocks=1)
    source = train_srn_from_d_rl(
        shard,
        tmp_path / "source-training",
        srn_config=config,
        training_config=SRNTrainingConfig(
            epochs=1, batch_size=2, validation_fraction=0.25, seed=31
        ),
    )
    candidate = train_srn_from_d_rl(
        shard,
        tmp_path / "candidate-training",
        srn_config=config,
        training_config=SRNTrainingConfig(
            epochs=10,
            batch_size=2,
            learning_rate=1e-30,
            validation_fraction=0.25,
            early_stopping_patience=2,
            early_stopping_min_delta=1.0,
            seed=31,
        ),
        initial_checkpoint=source.best_validation_checkpoint,
    )
    payload = load_srn_checkpoint(candidate.last_checkpoint).payload

    assert candidate.stopped_early
    assert candidate.stop_epoch == 2
    assert candidate.best_epoch == 0
    assert candidate.initialization["kind"] == "checkpoint_weights_fresh_optimizer"
    assert payload["initialization"] == candidate.initialization
    assert payload["stopped_early"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"early_stopping_patience": 0},
        {"early_stopping_min_delta": -1.0},
    ],
)
def test_early_stopping_configuration_is_validated(kwargs) -> None:
    with pytest.raises(ValueError):
        SRNTrainingConfig(**kwargs)
