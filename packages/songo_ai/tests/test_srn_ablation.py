"""Tests des ablations Policy/Value et des diagnostics Value du Lot 8."""

from __future__ import annotations

import torch

from songo_ai.dataset.selfplay_schema import RawSongoState
from songo_ai.evaluation import (
    ArenaConfig,
    HybridPolicyValueEvaluator,
    SRNMCTSAgent,
    calibration_metrics,
    calibration_report,
    generate_deterministic_openings,
    horizon_bucket,
    model_parameter_fingerprint,
    run_paired_arena,
    value_distribution_statistics,
)
from songo_ai.model import SRNConfig, SongoGraphBuilder, SongoRelationalNetwork
from songo_ai.search import MCTSConfig, SongoMCTS
from songo_ai.songo.rules import SongoLegacyGame


def _model(seed: int) -> SongoRelationalNetwork:
    torch.manual_seed(seed)
    return SongoRelationalNetwork(SRNConfig(hidden_dim=8, num_relational_blocks=1))


def _state() -> RawSongoState:
    game = SongoLegacyGame()
    for action in (2, 4, 1):
        game.play_local(action)
    return RawSongoState.from_game(game)


def test_hybrid_p0_v1_returns_exact_source_outputs() -> None:
    p0, p1 = _model(1), _model(2)
    graph = SongoGraphBuilder().build(_state()).as_batch()
    with torch.no_grad():
        p0_logits, _ = p0(graph)
        _, v1 = p1(graph)
        logits, value = HybridPolicyValueEvaluator(p0, p1, name="P0V1")(graph)
    assert torch.equal(logits, p0_logits)
    assert torch.equal(value, v1)


def test_hybrid_p1_v0_returns_exact_source_outputs() -> None:
    p0, p1 = _model(3), _model(4)
    graph = SongoGraphBuilder().build(_state()).as_batch()
    with torch.no_grad():
        p1_logits, _ = p1(graph)
        _, v0 = p0(graph)
        logits, value = HybridPolicyValueEvaluator(p1, p0, name="P1V0")(graph)
    assert torch.equal(logits, p1_logits)
    assert torch.equal(value, v0)


def test_same_model_uses_one_forward_and_hybrid_uses_two() -> None:
    p0, p1 = _model(5), _model(6)
    graph = SongoGraphBuilder().build(_state()).as_batch()
    same = HybridPolicyValueEvaluator(p0, p0, name="P0V0")
    mixed = HybridPolicyValueEvaluator(p0, p1, name="P0V1")
    with torch.no_grad():
        same(graph)
        mixed(graph)
    assert same.counters.model_evaluations == 1
    assert same.counters.network_forward_calls == 1
    assert mixed.counters.model_evaluations == 1
    assert mixed.counters.network_forward_calls == 2


def test_neutral_value_is_exactly_zero_and_keeps_policy() -> None:
    p0 = _model(7)
    graph = SongoGraphBuilder().build(_state()).as_batch()
    neutral = HybridPolicyValueEvaluator(p0, neutral_value=True, name="P0VN")
    with torch.no_grad():
        expected_policy, _ = p0(graph)
        policy, value = neutral(graph)
    assert torch.equal(policy, expected_policy)
    assert torch.equal(value, torch.zeros_like(value))
    assert neutral.counters.network_forward_calls == 1


def test_terminal_with_neutral_value_uses_engine_result_without_forward() -> None:
    # Etat importable que normalize_terminal reconnait sans dependre du flag
    # mutable ``finished`` : P1 au trait n'a plus aucune graine.
    terminal_state = RawSongoState(tuple([0] * 7 + [5] * 7 + [35, 0]), 1)
    evaluator = HybridPolicyValueEvaluator(
        _model(8), neutral_value=True, name="P0VN"
    )
    result = SongoMCTS(
        evaluator, config=MCTSConfig(num_simulations=4, add_root_noise=False)
    ).search(terminal_state, policy_temperature=0.0)
    assert result.selected_action is None
    assert result.network_evaluations == 0
    assert evaluator.counters.model_evaluations == 0
    assert result.root_value in (-1.0, 0.0, 1.0)


def test_hybrid_inference_does_not_modify_source_parameters() -> None:
    p0, p1 = _model(9), _model(10)
    before = (model_parameter_fingerprint(p0), model_parameter_fingerprint(p1))
    evaluator = HybridPolicyValueEvaluator(p0, p1, name="P0V1")
    graph = SongoGraphBuilder().build(_state()).as_batch()
    with torch.no_grad():
        evaluator(graph)
    assert before == (model_parameter_fingerprint(p0), model_parameter_fingerprint(p1))


def test_hybrid_arena_is_reproducible_and_counts_real_forwards() -> None:
    p0, p1 = _model(11), _model(12)
    openings = generate_deterministic_openings(prefix_lengths=[0, 2], seed=14)

    def run_once():
        a_eval = HybridPolicyValueEvaluator(p0, p0, name="P0V0")
        b_eval = HybridPolicyValueEvaluator(p0, p1, name="P0V1")
        results = run_paired_arena(
            SRNMCTSAgent("P0V0", a_eval, 2),
            SRNMCTSAgent("P0V1", b_eval, 2),
            openings,
            config=ArenaConfig(max_plies=8, seed=15, bootstrap_samples=20),
        )
        return results

    first, second = run_once(), run_once()
    assert [result.action_sequence for result in first] == [
        result.action_sequence for result in second
    ]
    assert all(
        result.total_network_forward_calls >= result.total_network_evaluations
        for result in first
    )
    assert any(
        result.total_network_forward_calls > result.total_network_evaluations
        for result in first
    )


def test_value_distribution_reports_confidence_and_saturation() -> None:
    report = value_distribution_statistics([-1.0, -0.9, 0.0, 0.6, 0.96])
    assert report["count"] == 5
    assert report["fraction_abs_gt_0_5"] == 4 / 5
    assert report["fraction_abs_gt_0_95"] == 2 / 5
    assert report["fraction_negative_saturated"] == 1 / 5
    assert report["fraction_positive_saturated"] == 1 / 5


def test_calibration_metrics_are_exact_on_a_small_example() -> None:
    report = calibration_metrics([1.0, -0.5, 0.0], [1.0, -1.0, 0.0])
    assert report["mse"] == 0.25 / 3
    assert report["mae"] == 0.5 / 3
    assert report["sign_accuracy"] == 1.0


def test_calibration_report_stratifies_horizon_player_and_result() -> None:
    report = calibration_report(
        {"g0": [0.1, -0.2, 0.3, -0.4, 0.0]},
        [1.0, -1.0, 1.0, -1.0, 0.0],
        [2, 10, 20, 45, 80],
        [1, 2, 1, 2, 1],
    )["g0"]
    assert set(report["horizon"]) == {"0-5", "6-15", "16-30", "31-60", ">60"}
    assert report["player_to_move"]["P1"]["count"] == 3
    assert report["result"]["loss"]["count"] == 2


def test_horizon_boundaries() -> None:
    assert [horizon_bucket(value) for value in (0, 5, 6, 15, 16, 30, 31, 60, 61)] == [
        "0-5",
        "0-5",
        "6-15",
        "6-15",
        "16-30",
        "16-30",
        "31-60",
        "31-60",
        ">60",
    ]
