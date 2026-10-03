import pytest
import torch

from songo_ai.dataset import RawSongoState
from songo_ai.model.srn_graph import (
    BASE_RELATIONS,
    GLOBAL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    NUM_NODES,
    SongoGraphBuilder,
)
from songo_ai.songo.rules import PLAYER_ONE, PLAYER_TWO


def test_graph_builder_has_14_nodes_minimal_features_and_global_context():
    state = RawSongoState(tuple([5] * 14 + [0, 0]), PLAYER_ONE)
    graph = SongoGraphBuilder().build(state)

    assert graph.node_features.shape == (NUM_NODES, NODE_FEATURE_DIM)
    assert graph.global_features.shape == (GLOBAL_FEATURE_DIM,)
    assert tuple(graph.edges_by_relation) == BASE_RELATIONS
    assert graph.action_nodes.tolist() == list(range(7))
    assert graph.player_to_move.item() == PLAYER_ONE
    assert graph.node_features[0, 0].item() == pytest.approx(5 / 70)
    assert graph.node_features[0, 1:4].tolist() == [1.0, 0.0, 1.0]
    assert graph.node_features[7, 1:4].tolist() == [0.0, 1.0, 0.0]
    assert graph.global_features.tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0, 1.0])


def test_next_and_prev_relations_are_directed_cycles():
    graph = SongoGraphBuilder().build(RawSongoState(tuple([5] * 14 + [0, 0]), PLAYER_ONE))
    next_edges = graph.edges_by_relation["next"]
    prev_edges = graph.edges_by_relation["prev"]

    assert next_edges.shape == (2, 14)
    assert prev_edges.shape == (2, 14)
    assert next_edges[:, 0].tolist() == [0, 1]
    assert next_edges[:, -1].tolist() == [13, 0]
    assert prev_edges[:, 0].tolist() == [0, 13]
    assert prev_edges[:, 1].tolist() == [1, 0]


def test_player_two_keeps_raw_orientation_and_maps_actions_to_nodes_7_to_13():
    board = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5)
    graph = SongoGraphBuilder().build(RawSongoState(board, PLAYER_TWO))

    assert graph.action_nodes.tolist() == list(range(7, 14))
    assert graph.node_features[:, 0].tolist() == pytest.approx([value / 70 for value in board[:14]])
    assert graph.node_features[0, 3].item() == 0.0
    assert graph.node_features[7, 3].item() == 1.0
    assert graph.global_features.tolist() == pytest.approx([4 / 70, 5 / 70, 0.0, 1.0, 61 / 70])


def test_position_frontier_and_distance_features_are_explicit():
    graph = SongoGraphBuilder().build(RawSongoState(tuple([5] * 14 + [0, 0]), PLAYER_ONE))

    assert graph.node_features[0, 4:].tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert graph.node_features[6, 4:].tolist() == pytest.approx([6 / 13, 1.0, 1.0, 0.0])
    assert graph.node_features[13, 4:].tolist() == pytest.approx([1.0, 1.0, 1.0, 0.0])


def test_graph_builder_batches_player_one_and_player_two_states():
    builder = SongoGraphBuilder()
    states = [
        RawSongoState(tuple([5] * 14 + [0, 0]), PLAYER_ONE),
        RawSongoState(tuple([5] * 14 + [0, 0]), PLAYER_TWO),
    ]
    batch = builder.build_batch(states)

    assert batch.batch_size == 2
    assert batch.node_features.shape == (2, 14, NODE_FEATURE_DIM)
    assert batch.global_features.shape == (2, GLOBAL_FEATURE_DIM)
    assert batch.player_to_move.tolist() == [PLAYER_ONE, PLAYER_TWO]
    assert batch.action_nodes.tolist() == [list(range(7)), list(range(7, 14))]
    assert batch.to("cpu").node_features.device == torch.device("cpu")


def test_graph_builder_rejects_empty_batch_and_unimplemented_relations():
    with pytest.raises(ValueError, match="empty"):
        SongoGraphBuilder().build_batch([])
    with pytest.raises(ValueError, match="not implemented"):
        SongoGraphBuilder(("next", "dynamic_sowing"))


def test_vectorized_graph_builder_is_bitwise_equivalent_to_reference_batch():
    states = [
        RawSongoState(tuple([5] * 14 + [0, 0]), PLAYER_ONE),
        RawSongoState((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5), PLAYER_TWO),
    ]
    builder = SongoGraphBuilder()
    reference = builder.build_batch(states)
    optimized = builder.build_batch_vectorized(states)

    assert torch.equal(reference.node_features, optimized.node_features)
    assert torch.equal(reference.global_features, optimized.global_features)
    assert torch.equal(reference.player_to_move, optimized.player_to_move)
    assert torch.equal(reference.action_nodes, optimized.action_nodes)
    for relation in BASE_RELATIONS:
        assert torch.equal(reference.edges_by_relation[relation], optimized.edges_by_relation[relation])


def test_vectorized_graph_builder_rejects_empty_batch():
    with pytest.raises(ValueError, match="empty"):
        SongoGraphBuilder().build_batch_vectorized([])
