from __future__ import annotations

import math
import random

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from songo_ai.dataset import RawSongoState
from songo_ai.model import SRNConfig, SongoGraphBuilder, SongoRelationalNetwork
from songo_ai.search.mcts import (
    MCTSConfig,
    MCTSNode,
    SongoMCTS,
    convert_value_perspective,
    puct_scores,
    select_puct_action,
    terminal_value,
    visit_counts_to_policy,
)
from songo_ai.songo.rules import DRAW, PLAYER_ONE, PLAYER_TWO, SongoLegacyGame


P1_IMMEDIATE_WIN = (1, 1, 6, 0, 1, 3, 0, 2, 1, 0, 0, 1, 0, 2, 34, 18)
P2_IMMEDIATE_WIN = (3, 1, 2, 6, 3, 0, 5, 3, 2, 0, 2, 4, 1, 2, 6, 30)
P2_IMMEDIATE_LOSS = (1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 2, 33, 32)
P2_IMMEDIATE_DRAW = (1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 2, 30, 35)
PARTIAL_LEGALITY = (14, 0, 0, 0, 0, 51, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0)


class ControlledNetwork(nn.Module):
    """Reseau deterministe pour isoler les tests du comportement MCTS."""

    def __init__(self, logits=None, value: float = 0.0):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.register_buffer(
            "fixed_logits",
            torch.tensor(logits if logits is not None else [0.0] * 7, dtype=torch.float32),
        )
        self.fixed_value = float(value)
        self.calls = 0

    def forward(self, graph):
        self.calls += graph.batch_size
        batch_size = graph.batch_size
        logits = self.fixed_logits.unsqueeze(0).expand(batch_size, -1) + self.anchor * 0.0
        value = torch.full(
            (batch_size,),
            self.fixed_value,
            dtype=logits.dtype,
            device=logits.device,
        ) + self.anchor * 0.0
        return logits, value


def _state(board=(5,) * 14 + (0, 0), player=PLAYER_ONE):
    return RawSongoState(tuple(board), player)


def _node(legal_mask=(True,) * 7):
    return MCTSNode(state=_state(), legal_mask=legal_mask, expanded=True)


@pytest.mark.parametrize(
    "winner,perspective,expected",
    [
        (PLAYER_ONE, PLAYER_ONE, 1.0),
        (PLAYER_ONE, PLAYER_TWO, -1.0),
        (PLAYER_TWO, PLAYER_TWO, 1.0),
        (PLAYER_TWO, PLAYER_ONE, -1.0),
        (DRAW, PLAYER_ONE, 0.0),
        (DRAW, PLAYER_TWO, 0.0),
    ],
)
def test_terminal_value_uses_explicit_player_perspective(winner, perspective, expected):
    assert terminal_value(winner, perspective) == expected


def test_value_conversion_compares_player_identity_instead_of_depth():
    assert convert_value_perspective(0.75, PLAYER_ONE, PLAYER_ONE) == 0.75
    assert convert_value_perspective(0.75, PLAYER_ONE, PLAYER_TWO) == -0.75
    assert convert_value_perspective(-0.25, PLAYER_TWO, PLAYER_ONE) == 0.25


def test_puct_prior_visit_count_q_and_legal_mask_have_expected_effects():
    node = _node((True, True, True, False, False, False, False))
    node.priors = [0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]
    assert select_puct_action(node, 1.0, random.Random(1)) == 0  # P domine au depart.

    node.visit_counts = [8, 1, 0, 0, 0, 0, 0]
    node.value_sums = [0.0] * 7
    node.priors = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert select_puct_action(node, 1.0, random.Random(1)) == 1  # N favorise l'action moins visitee.

    node.visit_counts = [1, 1, 0, 0, 0, 0, 0]
    node.value_sums = [0.9, 0.1, 0.0, 100.0, 0.0, 0.0, 0.0]
    node.priors = [0.0] * 7
    assert select_puct_action(node, 1.0, random.Random(1)) == 0  # Q domine l'exploitation.
    assert puct_scores(node, 1.0)[3] == float("-inf")  # Une action illegale reste exclue.


def test_puct_exact_ties_are_reproducible_with_a_seed():
    node = _node((True, True, False, False, False, False, False))
    node.priors = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    rng_1 = random.Random(91)
    rng_2 = random.Random(91)
    sequence_1 = [select_puct_action(node, 1.0, rng_1) for _ in range(5)]
    sequence_2 = [select_puct_action(node, 1.0, rng_2) for _ in range(5)]
    assert sequence_1 == sequence_2


def test_visit_counts_to_policy_handles_temperature_zero_positive_and_empty_counts():
    mask = (True, True, False, True, False, False, False)
    counts = (1, 3, 0, 0, 0, 0, 0)

    assert visit_counts_to_policy(counts, mask, 1.0) == pytest.approx((0.25, 0.75, 0, 0, 0, 0, 0))
    assert visit_counts_to_policy(counts, mask, 0.0) == (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert visit_counts_to_policy((0,) * 7, mask, 1.0) == pytest.approx(
        (1 / 3, 1 / 3, 0, 1 / 3, 0, 0, 0)
    )
    assert visit_counts_to_policy((0,) * 7, mask, 0.0) == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_visit_counts_to_policy_rejects_visits_on_illegal_actions():
    with pytest.raises(ValueError, match="illegal actions"):
        visit_counts_to_policy((0, 1, 0, 0, 0, 0, 0), (True, False, True, True, True, True, True))


def test_terminal_root_is_not_evaluated_by_network():
    model = ControlledNetwork()
    result = SongoMCTS(model, config=MCTSConfig(num_simulations=5, seed=3)).search(
        _state((0,) * 14 + (35, 35), PLAYER_ONE)
    )

    assert model.calls == 0
    assert result.network_evaluations == 0
    assert result.num_simulations == 0
    assert result.visit_counts == (0,) * 7
    assert result.policy == (0.0,) * 7
    assert result.root_value == 0.0


@pytest.mark.parametrize(
    "board,player,preferred_action,expected_value",
    [
        (P1_IMMEDIATE_WIN, PLAYER_ONE, 2, 1.0),
        (P2_IMMEDIATE_WIN, PLAYER_TWO, 4, 1.0),
        (P2_IMMEDIATE_LOSS, PLAYER_TWO, 6, -1.0),
        (P2_IMMEDIATE_DRAW, PLAYER_TWO, 6, 0.0),
    ],
)
def test_real_engine_terminal_transitions_backup_win_loss_and_draw(
    board, player, preferred_action, expected_value
):
    logits = [-20.0] * 7
    logits[preferred_action] = 20.0
    model = ControlledNetwork(logits=logits, value=0.0)
    result = SongoMCTS(model, config=MCTSConfig(num_simulations=1, seed=5)).search(
        _state(board, player)
    )

    assert result.visit_counts[preferred_action] == 1
    assert result.root_q_values[preferred_action] == expected_value
    assert result.root_value == expected_value
    # Seule la racine est evaluee : l'enfant terminal vient du moteur.
    assert result.network_evaluations == 1


@pytest.mark.parametrize(
    "board,player,winning_actions",
    [
        (P1_IMMEDIATE_WIN, PLAYER_ONE, {2, 5}),
        (P2_IMMEDIATE_WIN, PLAYER_TWO, {4, 6}),
    ],
)
def test_mcts_prefers_real_immediate_win_to_controlled_losing_nonterminal_lines(
    board, player, winning_actions
):
    # V=+1 dans les enfants non terminaux est bon pour l'adversaire au trait,
    # donc vaut -1 pour la racine. Les gains terminaux valent directement +1.
    model = ControlledNetwork(value=1.0)
    result = SongoMCTS(
        model,
        config=MCTSConfig(num_simulations=40, c_puct=1.5, seed=17),
    ).search(_state(board, player))

    assert result.selected_action in winning_actions
    assert max(result.visit_counts[action] for action in winning_actions) > max(
        result.visit_counts[action]
        for action, legal in enumerate(result.legal_mask)
        if legal and action not in winning_actions
    )


def test_nonterminal_engine_transitions_change_real_player_identity_for_both_sides():
    for player, expected_child_player in ((PLAYER_ONE, PLAYER_TWO), (PLAYER_TWO, PLAYER_ONE)):
        parent = SongoMCTS._root_node(_state(player=player))
        for action, legal in enumerate(parent.legal_mask):
            if legal:
                child = SongoMCTS._transition(parent, action)
                assert not child.terminal
                assert child.player_to_move == expected_child_player


def test_illegal_actions_receive_no_prior_no_visit_and_no_policy_mass():
    model = ControlledNetwork(logits=[0, 20, 20, 20, 20, 0, 20])
    result = SongoMCTS(model, config=MCTSConfig(num_simulations=8, seed=2)).search(
        _state(PARTIAL_LEGALITY, PLAYER_ONE)
    )

    assert result.legal_mask == (True, False, False, False, False, True, False)
    for action, legal in enumerate(result.legal_mask):
        if not legal:
            assert result.root_priors[action] == 0.0
            assert result.visit_counts[action] == 0
            assert result.policy[action] == 0.0
    assert sum(result.policy) == pytest.approx(1.0)


def test_root_dirichlet_noise_is_legal_normalized_optional_and_seeded():
    state = _state(PARTIAL_LEGALITY, PLAYER_ONE)
    base = SongoMCTS(
        ControlledNetwork(),
        config=MCTSConfig(num_simulations=1, add_root_noise=False, seed=44),
    ).search(state)
    noisy_config = MCTSConfig(
        num_simulations=1,
        add_root_noise=True,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.5,
        seed=44,
    )
    noisy_1 = SongoMCTS(ControlledNetwork(), config=noisy_config).search(state)
    noisy_2 = SongoMCTS(ControlledNetwork(), config=noisy_config).search(state)

    assert noisy_1.root_priors == pytest.approx(noisy_2.root_priors)
    assert noisy_1.root_priors != pytest.approx(base.root_priors)
    assert sum(noisy_1.root_priors) == pytest.approx(1.0)
    assert all(
        noisy_1.root_priors[action] == 0.0
        for action, legal in enumerate(noisy_1.legal_mask)
        if not legal
    )


def test_search_is_deterministic_without_noise_for_fixed_seed():
    config = MCTSConfig(num_simulations=18, add_root_noise=False, seed=991)
    state = _state()
    result_1 = SongoMCTS(ControlledNetwork(value=0.25), config=config).search(state)
    result_2 = SongoMCTS(ControlledNetwork(value=0.25), config=config).search(state)

    assert result_1.visit_counts == result_2.visit_counts
    assert result_1.policy == pytest.approx(result_2.policy)
    assert result_1.root_q_values == pytest.approx(result_2.root_q_values)
    assert result_1.root_value == pytest.approx(result_2.root_value)
    assert result_1.selected_action == result_2.selected_action


def test_mcts_uses_eval_no_grad_and_does_not_modify_model_parameters():
    model = ControlledNetwork(value=0.2)
    model.train()
    before = model.anchor.detach().clone()
    result = SongoMCTS(model, config=MCTSConfig(num_simulations=4, seed=9)).search(_state())

    assert result.num_simulations == 4
    assert model.training  # L'etat d'appel est restaure apres la recherche.
    assert torch.equal(before, model.anchor.detach())
    assert model.anchor.grad is None
    assert model.calls == result.network_evaluations


def test_real_srn_engine_graphbuilder_mcts_smoke():
    torch.manual_seed(20260924)
    model = SongoRelationalNetwork(SRNConfig(hidden_dim=16, num_relational_blocks=2))
    simulations = 12
    result = SongoMCTS(
        model,
        SongoGraphBuilder(),
        MCTSConfig(num_simulations=simulations, seed=20260924),
    ).search(_state(PARTIAL_LEGALITY, PLAYER_ONE))

    assert result.num_simulations == simulations
    assert sum(result.visit_counts) == simulations
    assert all(count == 0 for count, legal in zip(result.visit_counts, result.legal_mask) if not legal)
    assert sum(result.policy) == pytest.approx(1.0)
    assert all(math.isfinite(value) for value in result.policy)
    assert math.isfinite(result.root_value)
    assert result.num_nodes <= simulations + 1
    assert 1 <= result.network_evaluations == result.nodes_expanded <= result.num_nodes
    assert math.isfinite(result.simulations_per_second)
    assert result.simulations_per_second > 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_simulations": 0},
        {"c_puct": -1.0},
        {"dirichlet_alpha": 0.0},
        {"dirichlet_epsilon": 1.1},
    ],
)
def test_mcts_config_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        MCTSConfig(**kwargs)
