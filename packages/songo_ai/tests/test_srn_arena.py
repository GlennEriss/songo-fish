"""Tests du protocole experimental Lot 7."""

from __future__ import annotations

import json

import pytest
import torch

from songo_ai.dataset.schema import canonicalize_board
from songo_ai.evaluation import (
    ArenaConfig,
    ArenaDecision,
    ArenaGameResult,
    ArenaStatus,
    SRNMCTSAgent,
    evaluate_raw_network_outputs,
    generate_deterministic_openings,
    model_parameter_fingerprint,
    paired_bootstrap_interval,
    run_paired_arena,
    select_d_lab_benchmark,
    summarize_arena,
    validate_opening,
)
from songo_ai.model import SRNConfig, SongoRelationalNetwork
from songo_ai.songo.rules import PLAYER_ONE, PLAYER_TWO, SongoLegacyGame


class FirstLegalAgent:
    def __init__(self, name: str) -> None:
        self.name = name

    def select_action(self, state, *, seed: int) -> ArenaDecision:
        game = SongoLegacyGame.from_state(state.to_engine_state())
        return ArenaDecision(game.legal_local_actions()[0])


class RecordingFirstLegalAgent(FirstLegalAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.seeds: list[int] = []

    def select_action(self, state, *, seed: int) -> ArenaDecision:
        self.seeds.append(seed)
        return super().select_action(state, seed=seed)


def _small_model(seed: int) -> SongoRelationalNetwork:
    torch.manual_seed(seed)
    return SongoRelationalNetwork(SRNConfig(hidden_dim=8, num_relational_blocks=1))


def _result(
    opening: str,
    a_player: int,
    status: ArenaStatus,
    winner: int | None,
) -> ArenaGameResult:
    return ArenaGameResult(
        opening_id=opening,
        agent_a="a",
        agent_b="b",
        a_player=a_player,
        status=status,
        winner=winner,
        continuation_plies=10,
        prefix_plies=0,
        action_sequence=(0, 1),
        total_mcts_simulations=0,
        total_network_evaluations=0,
        elapsed_s=0.1,
    )


def test_openings_are_deterministic_distinct_legal_and_replayable() -> None:
    first = generate_deterministic_openings(
        prefix_lengths=[0, 1, 2, 3, 5, 8], seed=42
    )
    second = generate_deterministic_openings(
        prefix_lengths=[0, 1, 2, 3, 5, 8], seed=42
    )
    assert first == second
    assert len({(opening.state.board, opening.state.player_to_move) for opening in first}) == 6
    for opening in first:
        validate_opening(opening)
        game = SongoLegacyGame.from_state(opening.state.to_engine_state())
        assert not game.finished
        assert game.legal_local_actions()


def test_paired_arena_alternates_physical_sides_and_is_reproducible() -> None:
    openings = generate_deterministic_openings(prefix_lengths=[0, 2], seed=4)
    config = ArenaConfig(max_plies=6, repetition_limit=3, seed=5, bootstrap_samples=20)
    first = run_paired_arena(
        FirstLegalAgent("a"), FirstLegalAgent("b"), openings, config=config
    )
    second = run_paired_arena(
        FirstLegalAgent("a"), FirstLegalAgent("b"), openings, config=config
    )
    assert [result.a_player for result in first] == [1, 2, 1, 2]
    assert [result.action_sequence for result in first] == [
        result.action_sequence for result in second
    ]
    assert all(result.status == ArenaStatus.TRUNCATED_MAX_PLIES for result in first)


def test_paired_games_use_the_same_tie_break_seed_at_each_ply() -> None:
    opening = generate_deterministic_openings(prefix_lengths=[0], seed=4)
    agent_a = RecordingFirstLegalAgent("a")
    agent_b = RecordingFirstLegalAgent("b")
    run_paired_arena(
        agent_a,
        agent_b,
        opening,
        config=ArenaConfig(max_plies=4, seed=5, bootstrap_samples=20),
    )
    # Les deux agents jouent chacun deux coups par partie ; comme leur
    # politique est identique, les trajectoires et seeds apparies coincident.
    assert sorted(agent_a.seeds) == sorted(agent_b.seeds)


def test_deterministic_agents_can_reach_a_real_terminal() -> None:
    opening = generate_deterministic_openings(prefix_lengths=[0], seed=1)
    results = run_paired_arena(
        FirstLegalAgent("a"),
        FirstLegalAgent("b"),
        opening,
        config=ArenaConfig(max_plies=400, repetition_limit=100, bootstrap_samples=20),
    )
    assert all(result.status == ArenaStatus.TERMINAL_WIN for result in results)
    assert all(result.winner in (PLAYER_ONE, PLAYER_TWO) for result in results)


def test_truncation_is_not_counted_as_a_draw() -> None:
    results = (
        _result("o1", PLAYER_ONE, ArenaStatus.TRUNCATED_REPETITION, None),
        _result("o1", PLAYER_TWO, ArenaStatus.TERMINAL_DRAW, 0),
    )
    summary = summarize_arena(
        results, config=ArenaConfig(bootstrap_samples=20, seed=7)
    )
    assert summary.aggregate.games == 2
    assert summary.aggregate.terminal_games == 1
    assert summary.aggregate.draws == 1
    assert summary.aggregate.truncated_repetition == 1
    assert summary.score_rate_a_terminal == 0.5


def test_wdl_counting_respects_a_physical_side() -> None:
    results = (
        _result("o1", PLAYER_ONE, ArenaStatus.TERMINAL_WIN, PLAYER_ONE),
        _result("o1", PLAYER_TWO, ArenaStatus.TERMINAL_WIN, PLAYER_ONE),
        _result("o2", PLAYER_ONE, ArenaStatus.TERMINAL_DRAW, 0),
        _result("o2", PLAYER_TWO, ArenaStatus.TERMINAL_WIN, PLAYER_TWO),
    )
    summary = summarize_arena(
        results, config=ArenaConfig(bootstrap_samples=20, seed=2)
    )
    assert (summary.aggregate.wins, summary.aggregate.draws, summary.aggregate.losses) == (2, 1, 1)
    assert (summary.by_a_side["P1"].wins, summary.by_a_side["P1"].draws) == (1, 1)
    assert (summary.by_a_side["P2"].wins, summary.by_a_side["P2"].losses) == (1, 1)


def test_bootstrap_keeps_both_games_of_each_opening_together() -> None:
    results = (
        _result("o1", PLAYER_ONE, ArenaStatus.TERMINAL_WIN, PLAYER_ONE),
        _result("o1", PLAYER_TWO, ArenaStatus.TERMINAL_WIN, PLAYER_TWO),
        _result("o2", PLAYER_ONE, ArenaStatus.TERMINAL_WIN, PLAYER_TWO),
        _result("o2", PLAYER_TWO, ArenaStatus.TERMINAL_WIN, PLAYER_ONE),
    )
    interval = paired_bootstrap_interval(
        results, samples=200, confidence_level=0.95, seed=9
    )
    assert interval == (0.0, 1.0)


def test_two_srn_agents_must_share_the_same_search_budget() -> None:
    openings = generate_deterministic_openings(prefix_lengths=[0], seed=3)
    with pytest.raises(ValueError, match="same MCTS simulation budget"):
        run_paired_arena(
            SRNMCTSAgent("a", _small_model(1), 2),
            SRNMCTSAgent("b", _small_model(2), 3),
            openings,
            config=ArenaConfig(max_plies=1, bootstrap_samples=20),
        )


def test_srn_evaluation_has_no_noise_is_legal_and_does_not_modify_weights() -> None:
    model = _small_model(11)
    before = model_parameter_fingerprint(model)
    agent = SRNMCTSAgent("srn", model, num_simulations=2, c_puct=1.5)
    assert agent.mcts_config.add_root_noise is False
    opening = generate_deterministic_openings(prefix_lengths=[0], seed=8)[0]
    decision = agent.select_action(opening.state, seed=17)
    game = SongoLegacyGame.from_state(opening.state.to_engine_state())
    assert decision.action in game.legal_local_actions()
    assert decision.mcts_simulations == 2
    assert model_parameter_fingerprint(model) == before


def test_srn_mcts_action_is_reproducible_for_a_fixed_seed() -> None:
    model = _small_model(12)
    agent = SRNMCTSAgent("srn", model, num_simulations=4)
    state = generate_deterministic_openings(prefix_lengths=[3], seed=10)[0].state
    first = agent.select_action(state, seed=99)
    second = agent.select_action(state, seed=99)
    assert first == second


def test_d_lab_selection_uses_structural_strata_and_both_players(tmp_path) -> None:
    game = SongoLegacyGame()
    opening = {
        "state": list(canonicalize_board(tuple(game.board), game.turn)),
        "legal_mask": list(game.legal_mask()),
        "trajectory_id": "t0",
        "move_number": 0,
    }
    game.play_local(0)
    middle = {
        "state": list(canonicalize_board(tuple(game.board), game.turn)),
        "legal_mask": list(game.legal_mask()),
        "trajectory_id": "t1",
        "move_number": 20,
    }
    path = tmp_path / "lab.jsonl"
    path.write_text(json.dumps(opening) + "\n" + json.dumps(middle) + "\n")
    positions = select_d_lab_benchmark(path, seed=1)
    assert len(positions) == 2
    assert {position.state.player_to_move for position in positions} == {1, 2}
    assert {position.phase_proxy for position in positions} == {"opening", "midgame"}


def test_raw_network_report_separates_models_and_preserves_parameters(tmp_path) -> None:
    game = SongoLegacyGame()
    record = {
        "state": list(game.board),
        "legal_mask": list(game.legal_mask()),
        "trajectory_id": "t0",
        "move_number": 0,
    }
    path = tmp_path / "lab.jsonl"
    path.write_text(json.dumps(record) + "\n")
    positions = select_d_lab_benchmark(path, seed=1)
    models = {"g0": _small_model(20), "g1": _small_model(21)}
    before = {name: model_parameter_fingerprint(model) for name, model in models.items()}
    report = evaluate_raw_network_outputs(models, positions)
    assert set(report["models"]) == {"g0", "g1"}
    assert "g0__vs__g1" in report["pairwise"]
    assert report["selection"]["uses_model_outputs"] is False
    assert {name: model_parameter_fingerprint(model) for name, model in models.items()} == before
