from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from songo_ai.dataset import RawSongoState
from songo_ai.model.srn_graph import SongoGraphBuilder
from songo_ai.model.srn_network import (
    SRNConfig,
    SongoRelationalNetwork,
    mask_policy_logits,
    masked_policy_cross_entropy,
    policy_probabilities,
)
from songo_ai.songo.rules import PLAYER_ONE, PLAYER_TWO, SongoLegacyGame


def _initial(player=PLAYER_ONE):
    return RawSongoState(tuple([5] * 14 + [0, 0]), player)


def test_srn_forward_supports_batch_size_one_and_bounds_value():
    model = SongoRelationalNetwork(SRNConfig(hidden_dim=16, num_relational_blocks=2))
    graph = SongoGraphBuilder().build(_initial())

    logits, value = model(graph)

    assert logits.shape == (1, 7)
    assert value.shape == (1,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()
    assert -1.0 <= value.item() <= 1.0


def test_srn_forward_supports_mixed_player_batch():
    model = SongoRelationalNetwork(SRNConfig(hidden_dim=16))
    graph = SongoGraphBuilder().build_batch([_initial(PLAYER_ONE), _initial(PLAYER_TWO)])

    logits, value = model(graph)

    assert logits.shape == (2, 7)
    assert value.shape == (2,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()


def test_policy_is_shared_and_conditioned_by_selected_action_nodes():
    torch.manual_seed(7)
    model = SongoRelationalNetwork(SRNConfig(hidden_dim=16, num_relational_blocks=1)).eval()
    graph = SongoGraphBuilder().build(_initial()).as_batch()
    reversed_graph = replace(graph, action_nodes=graph.action_nodes.flip(dims=(1,)))

    logits, value = model(graph)
    reversed_logits, reversed_value = model(reversed_graph)

    assert model.policy_mlp[-1].out_features == 1
    assert torch.allclose(reversed_logits, logits.flip(dims=(1,)))
    assert torch.allclose(reversed_value, value)


def test_safe_legal_mask_produces_exact_zeros_and_normalized_rows():
    logits = torch.tensor([[1.0, -2.0, 4.0, 0.5, 8.0, -1.0, 0.0]])
    legal_mask = torch.tensor([[True, True, False, True, False, True, False]])

    masked = mask_policy_logits(logits, legal_mask)
    probabilities = policy_probabilities(logits, legal_mask)

    assert torch.isfinite(masked).all()
    assert (masked[~legal_mask] < -1.0e8).all()
    assert probabilities[~legal_mask].tolist() == [0.0, 0.0, 0.0]
    assert probabilities.sum().item() == pytest.approx(1.0)


def test_safe_legal_mask_rejects_position_without_legal_action():
    with pytest.raises(ValueError, match="at least one legal"):
        policy_probabilities(torch.zeros(1, 7), torch.zeros(1, 7, dtype=torch.bool))


def test_masked_policy_loss_is_finite_and_rejects_illegal_target_mass():
    logits = torch.tensor([[0.2, 0.1, -0.5, 0.4, 0.9, -0.2, 0.3]], requires_grad=True)
    legal_mask = torch.tensor([[True, True, False, True, False, True, True]])
    target = torch.tensor([[0.1, 0.2, 0.0, 0.4, 0.0, 0.2, 0.1]])

    loss = masked_policy_cross_entropy(logits, legal_mask, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    invalid_target = target.clone()
    invalid_target[0, 2] = 0.1
    invalid_target[0, 3] = 0.3
    with pytest.raises(ValueError, match="illegal"):
        masked_policy_cross_entropy(logits.detach(), legal_mask, invalid_target)


def test_gradients_reach_node_global_relational_policy_and_value_parameters():
    torch.manual_seed(11)
    model = SongoRelationalNetwork(SRNConfig(hidden_dim=16, num_relational_blocks=2))
    graph = SongoGraphBuilder().build_batch([_initial(PLAYER_ONE), _initial(PLAYER_TWO)])
    logits, value = model(graph)
    mask = torch.ones_like(logits, dtype=torch.bool)
    target = torch.full_like(logits, 1 / 7)
    loss = masked_policy_cross_entropy(logits, mask, target) + F.mse_loss(
        value, torch.tensor([1.0, -1.0])
    )

    loss.backward()

    named_gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
    expected_prefixes = (
        "node_encoder",
        "global_encoder",
        "relational_blocks",
        "state_fusion",
        "policy_mlp",
        "value_mlp",
    )
    for prefix in expected_prefixes:
        gradients = [gradient for name, gradient in named_gradients.items() if name.startswith(prefix)]
        assert gradients
        assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_minimal_training_step_runs_on_real_engine_states_and_updates_parameters():
    torch.manual_seed(13)
    games = [
        SongoLegacyGame(),
        SongoLegacyGame.from_board(
            [14, 0, 0, 0, 0, 51, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0],
            turn=PLAYER_ONE,
        ),
    ]
    states = [RawSongoState.from_game(game) for game in games]
    legal_mask = torch.tensor([game.legal_mask() for game in games], dtype=torch.bool)
    policy_target = torch.stack(
        [
            legal_mask[0].float() / legal_mask[0].sum(),
            legal_mask[1].float() / legal_mask[1].sum(),
        ]
    )
    value_target = torch.tensor([0.0, 1.0])
    graph = SongoGraphBuilder().build_batch(states)
    model = SongoRelationalNetwork(SRNConfig(hidden_dim=16, num_relational_blocks=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.node_encoder[0].weight.detach().clone()

    logits, value = model(graph)
    loss = masked_policy_cross_entropy(logits, legal_mask, policy_target) + F.mse_loss(
        value, value_target
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert not torch.equal(before, model.node_encoder[0].weight.detach())


def test_config_exposes_baseline_choices_and_rejects_unimplemented_variants():
    config = SRNConfig()
    assert config.num_relational_blocks == 3
    assert config.pooling == "mean"
    assert config.relation_types == ("next", "prev")
    with pytest.raises(ValueError, match="mean pooling"):
        SRNConfig(pooling="attention")
    with pytest.raises(ValueError, match="not implemented"):
        SRNConfig(relation_types=("next", "mirror"))

