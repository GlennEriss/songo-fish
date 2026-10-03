from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from songo_ai.dataset import RawSongoState
from songo_ai.generation.selfplay import (
    PendingSelfPlayStep,
    SelfPlayConfig,
    SelfPlayRunner,
    SelfPlayStatus,
    finalize_selfplay_steps,
    select_action_from_policy,
)
from songo_ai.model import SRNConfig, SongoRelationalNetwork
from songo_ai.search import MCTSConfig, MCTSResult, visit_counts_to_policy
from songo_ai.songo.rules import DRAW, PLAYER_ONE, PLAYER_TWO, IllegalMove, SongoLegacyGame


P1_IMMEDIATE_WIN = (1, 1, 6, 0, 1, 3, 0, 2, 1, 0, 0, 1, 0, 2, 34, 18)
P2_IMMEDIATE_WIN = (3, 1, 2, 6, 3, 0, 5, 3, 2, 0, 2, 4, 1, 2, 6, 30)
P2_IMMEDIATE_DRAW = (1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 2, 30, 35)


def _state(player):
    return RawSongoState(tuple([5] * 14 + [0, 0]), player)


def _pending(player, ply):
    return PendingSelfPlayStep(
        state=_state(player),
        legal_mask=(True,) * 7,
        visit_counts=(1, 1, 1, 1, 1, 1, 1),
        policy_target=(1 / 7,) * 7,
        play_policy=(1 / 7,) * 7,
        action_played=0,
        metadata={"game_id": "test", "ply": ply, "player_to_move": player},
    )


class FixedSearch:
    def __init__(self, counts):
        self.counts = tuple(counts)

    def search(self, state, *, policy_temperature=1.0):
        game = SongoLegacyGame.from_state(state.to_engine_state())
        mask = game.legal_mask()
        policy = visit_counts_to_policy(self.counts, mask, policy_temperature)
        return MCTSResult(
            visit_counts=self.counts,
            policy=policy,
            root_value=0.125,
            selected_action=max(range(7), key=lambda action: self.counts[action]),
            num_simulations=sum(self.counts),
            num_nodes=4,
            nodes_expanded=3,
            network_evaluations=3,
            elapsed_s=0.001,
            simulations_per_second=1000.0,
            root_priors=policy,
            root_q_values=(0.0,) * 7,
            legal_mask=mask,
        )


class UniformLegalSearch:
    def search(self, state, *, policy_temperature=1.0):
        game = SongoLegacyGame.from_state(state.to_engine_state())
        mask = game.legal_mask()
        counts = tuple(1 if legal else 0 for legal in mask)
        policy = visit_counts_to_policy(counts, mask, policy_temperature)
        return MCTSResult(
            visit_counts=counts,
            policy=policy,
            root_value=0.0,
            selected_action=next(action for action, legal in enumerate(mask) if legal),
            num_simulations=sum(counts),
            num_nodes=2,
            nodes_expanded=2,
            network_evaluations=2,
            elapsed_s=0.001,
            simulations_per_second=1000.0,
            root_priors=policy,
            root_q_values=(0.0,) * 7,
            legal_mask=mask,
        )


class CyclingGame:
    """Double technique : meme board et meme joueur apres chaque action."""

    def __init__(self):
        self.board = [5] * 14 + [0, 0]
        self.turn = PLAYER_ONE
        self.finished = False
        self.winner = None

    def normalize_terminal(self):
        return None

    def legal_mask(self):
        return (True, False, False, False, False, False, False)

    def play_local(self, action):
        if action != 0:
            raise IllegalMove(action)

    def final_score_with_territory(self):
        return 35, 35


@pytest.mark.parametrize(
    "winner,expected",
    [
        (PLAYER_ONE, (1.0, -1.0, 1.0)),
        (PLAYER_TWO, (-1.0, 1.0, -1.0)),
        (DRAW, (0.0, 0.0, 0.0)),
    ],
)
def test_terminal_z_is_assigned_from_winner_and_each_states_player(winner, expected):
    steps = (_pending(PLAYER_ONE, 0), _pending(PLAYER_TWO, 1), _pending(PLAYER_ONE, 2))
    status = SelfPlayStatus.TERMINAL_DRAW if winner == DRAW else SelfPlayStatus.TERMINAL_WIN

    examples = finalize_selfplay_steps(
        steps,
        status=status,
        winner=winner,
        final_score=(35, 35),
        include_truncated_examples=True,
    )

    assert tuple(example.value_target for example in examples) == expected
    assert all(example.metadata["winner"] == winner for example in examples)


@pytest.mark.parametrize(
    "board,player,action,winner,status",
    [
        (P1_IMMEDIATE_WIN, PLAYER_ONE, 2, PLAYER_ONE, SelfPlayStatus.TERMINAL_WIN),
        (P2_IMMEDIATE_WIN, PLAYER_TWO, 4, PLAYER_TWO, SelfPlayStatus.TERMINAL_WIN),
        (P2_IMMEDIATE_DRAW, PLAYER_TWO, 6, DRAW, SelfPlayStatus.TERMINAL_DRAW),
    ],
)
def test_real_engine_selfplay_terminal_result_populates_real_z(board, player, action, winner, status):
    counts = [0] * 7
    counts[action] = 10
    config = SelfPlayConfig(
        max_game_plies=1,
        action_temperature=0.0,
        temperature_drop_ply=1,
        mcts=MCTSConfig(num_simulations=10, add_root_noise=False),
    )
    runner = SelfPlayRunner(
        object(),
        config,
        game_factory=lambda: SongoLegacyGame.from_board(board, player),
        search_factory=lambda _: FixedSearch(counts),
    )

    result = runner.play_game()

    assert result.status is status
    assert result.winner == winner
    assert result.num_plies == 1
    assert result.examples[0].value_target == (0.0 if winner == DRAW else 1.0)


def test_max_plies_is_truncation_and_never_a_fake_draw():
    config = SelfPlayConfig(
        max_game_plies=1,
        include_truncated_examples=True,
        mcts=MCTSConfig(num_simulations=7, add_root_noise=False),
    )
    result = SelfPlayRunner(
        object(),
        config,
        search_factory=lambda _: UniformLegalSearch(),
    ).play_game()

    assert result.status is SelfPlayStatus.TRUNCATED_MAX_PLIES
    assert result.winner is None
    assert result.final_score is None
    assert result.examples[0].value_target is None
    assert result.examples[0].metadata["winner"] is None


def test_repetition_is_technical_truncation_and_never_a_fake_draw():
    config = SelfPlayConfig(
        max_game_plies=10,
        repetition_limit=2,
        include_truncated_examples=True,
        action_temperature=0.0,
        mcts=MCTSConfig(num_simulations=1, add_root_noise=False),
    )
    result = SelfPlayRunner(
        object(),
        config,
        game_factory=CyclingGame,
        search_factory=lambda _: FixedSearch((1, 0, 0, 0, 0, 0, 0)),
    ).play_game()

    assert result.status is SelfPlayStatus.TRUNCATED_REPETITION
    assert result.winner is None
    assert result.num_plies == 1
    assert result.examples[0].value_target is None
    assert "technical limit=2" in result.truncation_reason


def test_truncated_examples_can_be_excluded_from_training_output():
    config = SelfPlayConfig(
        max_game_plies=1,
        include_truncated_examples=False,
        mcts=MCTSConfig(num_simulations=1, add_root_noise=False),
    )
    result = SelfPlayRunner(
        object(), config, search_factory=lambda _: UniformLegalSearch()
    ).play_game()
    assert result.status is SelfPlayStatus.TRUNCATED_MAX_PLIES
    assert result.num_plies == 1
    assert result.examples == ()


def test_target_and_play_temperatures_are_distinct_and_do_not_change_source_counts():
    counts = (10, 0, 30, 0, 0, 10, 0)
    mask = (True,) * 7
    target = visit_counts_to_policy(counts, mask, temperature=1.0)
    cold_play = visit_counts_to_policy(counts, mask, temperature=0.0)
    hot_play = visit_counts_to_policy(counts, mask, temperature=10.0)

    assert target == pytest.approx((0.2, 0, 0.6, 0, 0, 0.2, 0))
    assert cold_play == (0, 0, 1, 0, 0, 0, 0)
    assert hot_play != pytest.approx(target)
    assert select_action_from_policy(cold_play, mask, random.Random(1)) == 2
    assert select_action_from_policy(hot_play, mask, random.Random(1)) == 0
    assert counts == (10, 0, 30, 0, 0, 10, 0)


def test_temperature_schedule_is_based_only_on_ply_number():
    config = SelfPlayConfig(
        action_temperature=1.25,
        temperature_drop_ply=3,
        late_action_temperature=0.0,
    )
    assert [config.action_temperature_at(ply) for ply in range(5)] == [1.25, 1.25, 1.25, 0.0, 0.0]


def test_strict_action_sampler_rejects_illegal_probability_mass():
    with pytest.raises(IllegalMove, match="illegal"):
        select_action_from_policy(
            (0.5, 0.5, 0, 0, 0, 0, 0),
            (True, False, False, False, False, False, False),
            random.Random(1),
        )


def test_selfplay_metadata_separates_target_play_and_search_diagnostics():
    counts = (10, 0, 30, 0, 0, 10, 0)
    config = SelfPlayConfig(
        max_game_plies=1,
        target_temperature=1.0,
        action_temperature=0.0,
        generation=4,
        checkpoint_id="songo-g4",
        mcts=MCTSConfig(num_simulations=50, add_root_noise=False),
    )
    result = SelfPlayRunner(
        object(), config, search_factory=lambda _: FixedSearch(counts)
    ).play_game(2)
    example = result.examples[0]

    assert example.visit_counts == counts
    assert example.policy_target == pytest.approx((0.2, 0, 0.6, 0, 0, 0.2, 0))
    assert example.metadata["play_policy"] == [0, 0, 1, 0, 0, 0, 0]
    assert example.metadata["action_played"] == 2
    assert example.metadata["generation"] == 4
    assert example.metadata["checkpoint_id"] == "songo-g4"
    assert example.metadata["root_value"] == 0.125
    assert example.metadata["search_nodes"] == 4
    assert example.metadata["network_evaluations"] == 3


def test_selfplay_is_reproducible_for_fixed_seed_and_deterministic_search():
    config = SelfPlayConfig(
        max_game_plies=8,
        seed=721,
        mcts=MCTSConfig(num_simulations=7, add_root_noise=False),
    )
    runner_1 = SelfPlayRunner(object(), config, search_factory=lambda _: UniformLegalSearch())
    runner_2 = SelfPlayRunner(object(), config, search_factory=lambda _: UniformLegalSearch())

    result_1 = runner_1.play_game(0)
    result_2 = runner_2.play_game(0)

    assert result_1.game_id == result_2.game_id
    assert result_1.action_sequence == result_2.action_sequence
    assert result_1.status == result_2.status
    assert tuple(example.state for example in result_1.examples) == tuple(
        example.state for example in result_2.examples
    )


def test_generation_aggregates_multiple_games_and_statistics():
    config = SelfPlayConfig(
        games=3,
        max_game_plies=1,
        seed=8,
        mcts=MCTSConfig(num_simulations=7, add_root_noise=False),
    )
    run = SelfPlayRunner(object(), config, search_factory=lambda _: UniformLegalSearch()).generate()
    stats = run.statistics

    assert len(run.games) == 3
    assert len(run.examples) == 3
    assert stats.games_completed == 0
    assert stats.games_truncated == 3
    assert stats.truncations_max_plies == 3
    assert stats.total_positions == 3
    assert stats.positions_encountered == 3
    assert stats.average_game_length == 1.0
    assert stats.median_game_length == 1.0
    assert stats.min_game_length == stats.max_game_length == 1
    assert stats.length_distribution == {1: 3}
    assert stats.total_mcts_simulations == 21
    assert stats.total_network_evaluations == 6


def test_real_engine_graphbuilder_srn_mcts_selfplay_smoke():
    torch.manual_seed(31415)
    model = SongoRelationalNetwork(SRNConfig(hidden_dim=16, num_relational_blocks=2))
    config = SelfPlayConfig(
        games=1,
        max_game_plies=6,
        repetition_limit=3,
        seed=31415,
        mcts=MCTSConfig(num_simulations=2, add_root_noise=False, seed=31415),
    )
    result = SelfPlayRunner(model, config).play_game()

    assert result.num_plies >= 1
    assert result.total_mcts_simulations == result.num_plies * 2
    for example in result.examples:
        game = SongoLegacyGame.from_state(example.state.to_engine_state())
        assert tuple(game.legal_mask()) == example.legal_mask
        assert sum(example.visit_counts) == 2
        assert sum(example.policy_target) == pytest.approx(1.0)
        assert example.visit_counts[example.metadata["action_played"]] >= 0
        assert example.legal_mask[example.metadata["action_played"]]
        assert all(
            count == 0
            for count, legal in zip(example.visit_counts, example.legal_mask)
            if not legal
        )
        if result.status.is_terminal:
            assert example.value_target in (-1.0, 0.0, 1.0)
        else:
            assert example.value_target is None


def test_generation_provenance_and_search_parameters_are_stored_per_example():
    config = SelfPlayConfig(
        games=1,
        max_game_plies=1,
        seed=1200,
        generation=2,
        checkpoint_id="g1-best-sha256",
        provenance={
            "generation_lineage": "G1_to_G2",
            "generator_checkpoint_sha256": "abc123",
        },
        mcts=MCTSConfig(
            num_simulations=7,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            add_root_noise=True,
        ),
    )
    example = SelfPlayRunner(
        object(), config, search_factory=lambda _: UniformLegalSearch()
    ).play_game().examples[0]

    assert example.metadata["generation_lineage"] == "G1_to_G2"
    assert example.metadata["generator_checkpoint_sha256"] == "abc123"
    assert example.metadata["selfplay_seed"] == 1200
    assert example.metadata["c_puct"] == 1.5
    assert example.metadata["root_noise"] is True


def test_selfplay_default_enables_root_noise_but_debug_can_disable_it():
    assert SelfPlayConfig().mcts.add_root_noise
    assert not SelfPlayConfig(mcts=MCTSConfig(add_root_noise=False)).mcts.add_root_noise


@pytest.mark.parametrize(
    "kwargs",
    [
        {"games": 0},
        {"max_game_plies": 0},
        {"repetition_limit": 1},
        {"target_temperature": -1.0},
        {"temperature_drop_ply": -1},
    ],
)
def test_selfplay_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SelfPlayConfig(**kwargs)
