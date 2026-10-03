"""Construction tensorielle du graphe Songo pour le SRN v0.1.

Le builder recoit exclusivement un etat moteur brut. Il ne canonicalise pas
les joueurs et ne lit aucune annotation teacher. Les 14 cases jouables sont
des noeuds ; les magasins restent dans le contexte global.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch

from songo_ai.dataset.selfplay_schema import RawSongoState
from songo_ai.songo.rules import NUM_ACTIONS, PLAYER_ONE, PLAYER_TWO, TOTAL_SEEDS


NUM_NODES = 14
NODE_FEATURE_DIM = 8
GLOBAL_FEATURE_DIM = 5
BASE_RELATIONS = ("next", "prev")


@dataclass(frozen=True)
class SongoGraph:
    """Un graphe Songo non batche."""

    node_features: torch.Tensor  # [14, node_feature_dim]
    global_features: torch.Tensor  # [global_feature_dim]
    edges_by_relation: Mapping[str, torch.Tensor]  # relation -> [2, E]
    player_to_move: torch.Tensor  # scalaire long
    action_nodes: torch.Tensor  # [7] long

    def as_batch(self) -> "SongoGraphBatch":
        return SongoGraphBatch(
            node_features=self.node_features.unsqueeze(0),
            global_features=self.global_features.unsqueeze(0),
            edges_by_relation=self.edges_by_relation,
            player_to_move=self.player_to_move.reshape(1),
            action_nodes=self.action_nodes.unsqueeze(0),
        )


@dataclass(frozen=True)
class SongoGraphBatch:
    """Batch dense de graphes de taille fixe.

    Les relations NEXT/PREV sont identiques pour tous les exemples et ne sont
    donc stockees qu'une fois, tandis que les features conservent une dimension
    de batch explicite.
    """

    node_features: torch.Tensor  # [B, 14, node_feature_dim]
    global_features: torch.Tensor  # [B, global_feature_dim]
    edges_by_relation: Mapping[str, torch.Tensor]  # relation -> [2, E]
    player_to_move: torch.Tensor  # [B] long
    action_nodes: torch.Tensor  # [B, 7] long

    @property
    def batch_size(self) -> int:
        return int(self.node_features.shape[0])

    def to(self, device: torch.device | str) -> "SongoGraphBatch":
        return SongoGraphBatch(
            node_features=self.node_features.to(device),
            global_features=self.global_features.to(device),
            edges_by_relation={name: edges.to(device) for name, edges in self.edges_by_relation.items()},
            player_to_move=self.player_to_move.to(device),
            action_nodes=self.action_nodes.to(device),
        )


class SongoGraphBuilder:
    """Transforme l'etat brut en features minimales du SRN v0.1."""

    def __init__(self, relation_types: Sequence[str] = BASE_RELATIONS) -> None:
        relation_types = tuple(relation_types)
        unknown = set(relation_types) - set(BASE_RELATIONS)
        if unknown:
            raise ValueError(
                f"relations not implemented in SRN v0.1: {sorted(unknown)}; "
                f"available={list(BASE_RELATIONS)}"
            )
        if not relation_types:
            raise ValueError("at least one relation type is required")
        self.relation_types: Tuple[str, ...] = relation_types
        self._edges = self._build_edges(relation_types)

    @staticmethod
    def _build_edges(relation_types: Sequence[str]) -> Dict[str, torch.Tensor]:
        nodes = torch.arange(NUM_NODES, dtype=torch.long)
        builders = {
            "next": lambda: torch.stack((nodes, (nodes + 1) % NUM_NODES)),
            "prev": lambda: torch.stack((nodes, (nodes - 1) % NUM_NODES)),
        }
        return {name: builders[name]() for name in relation_types}

    @staticmethod
    def action_nodes_for_player(player_to_move: int) -> torch.Tensor:
        if player_to_move == PLAYER_ONE:
            start = 0
        elif player_to_move == PLAYER_TWO:
            start = 7
        else:
            raise ValueError("player_to_move must be PLAYER_ONE or PLAYER_TWO")
        return torch.arange(start, start + NUM_ACTIONS, dtype=torch.long)

    @staticmethod
    def _node_features(state: RawSongoState) -> torch.Tensor:
        rows = []
        for node_index in range(NUM_NODES):
            owner = PLAYER_ONE if node_index < 7 else PLAYER_TWO
            local_position = node_index % 7
            rows.append(
                [
                    state.board[node_index] / TOTAL_SEEDS,
                    1.0 if owner == PLAYER_ONE else 0.0,
                    1.0 if owner == PLAYER_TWO else 0.0,
                    1.0 if owner == state.player_to_move else 0.0,
                    node_index / (NUM_NODES - 1),
                    local_position / 6.0,
                    1.0 if local_position == 6 else 0.0,
                    (6 - local_position) / 6.0,
                ]
            )
        return torch.tensor(rows, dtype=torch.float32)

    @staticmethod
    def _global_features(state: RawSongoState) -> torch.Tensor:
        seeds_in_play = sum(state.board[:NUM_NODES])
        return torch.tensor(
            [
                state.board[14] / TOTAL_SEEDS,
                state.board[15] / TOTAL_SEEDS,
                1.0 if state.player_to_move == PLAYER_ONE else 0.0,
                1.0 if state.player_to_move == PLAYER_TWO else 0.0,
                seeds_in_play / TOTAL_SEEDS,
            ],
            dtype=torch.float32,
        )

    def build(self, state: RawSongoState) -> SongoGraph:
        return SongoGraph(
            node_features=self._node_features(state),
            global_features=self._global_features(state),
            edges_by_relation=self._edges,
            player_to_move=torch.tensor(state.player_to_move, dtype=torch.long),
            action_nodes=self.action_nodes_for_player(state.player_to_move),
        )

    def build_batch(self, states: Iterable[RawSongoState]) -> SongoGraphBatch:
        graphs = [self.build(state) for state in states]
        if not graphs:
            raise ValueError("cannot build an empty graph batch")
        return SongoGraphBatch(
            node_features=torch.stack([graph.node_features for graph in graphs]),
            global_features=torch.stack([graph.global_features for graph in graphs]),
            edges_by_relation=self._edges,
            player_to_move=torch.stack([graph.player_to_move for graph in graphs]),
            action_nodes=torch.stack([graph.action_nodes for graph in graphs]),
        )

    def build_batch_vectorized(self, states: Iterable[RawSongoState]) -> SongoGraphBatch:
        """Construit le même graphe dense sans créer un objet par état.

        La topologie et les caractéristiques constantes restent strictement
        identiques à :meth:`build_batch`; seules les colonnes dépendant du
        plateau et du joueur sont remplies par opérations batchées.
        """

        states = tuple(states)
        if not states:
            raise ValueError("cannot build an empty graph batch")
        boards = torch.tensor([state.board for state in states], dtype=torch.float32)
        players = torch.tensor([state.player_to_move for state in states], dtype=torch.long)
        batch_size = len(states)

        node_features = torch.empty(
            (batch_size, NUM_NODES, NODE_FEATURE_DIM), dtype=torch.float32
        )
        node_features[:, :, 0] = boards[:, :NUM_NODES] / TOTAL_SEEDS
        owners_p1 = torch.arange(NUM_NODES) < 7
        node_features[:, :, 1] = owners_p1.to(torch.float32)
        node_features[:, :, 2] = (~owners_p1).to(torch.float32)
        node_features[:, :, 3] = (
            owners_p1.unsqueeze(0) == (players == PLAYER_ONE).unsqueeze(1)
        ).to(torch.float32)
        node_indices = torch.arange(NUM_NODES, dtype=torch.float32)
        local_positions = torch.arange(NUM_NODES, dtype=torch.float32).remainder(7)
        node_features[:, :, 4] = node_indices / (NUM_NODES - 1)
        node_features[:, :, 5] = local_positions / 6.0
        node_features[:, :, 6] = (local_positions == 6).to(torch.float32)
        node_features[:, :, 7] = (6.0 - local_positions) / 6.0

        global_features = torch.empty((batch_size, GLOBAL_FEATURE_DIM), dtype=torch.float32)
        global_features[:, 0:2] = boards[:, 14:16] / TOTAL_SEEDS
        global_features[:, 2] = (players == PLAYER_ONE).to(torch.float32)
        global_features[:, 3] = (players == PLAYER_TWO).to(torch.float32)
        global_features[:, 4] = boards[:, :NUM_NODES].sum(dim=1) / TOTAL_SEEDS
        action_offsets = torch.where(players == PLAYER_ONE, 0, 7).unsqueeze(1)
        action_nodes = action_offsets + torch.arange(NUM_ACTIONS, dtype=torch.long)
        return SongoGraphBatch(
            node_features=node_features,
            global_features=global_features,
            edges_by_relation=self._edges,
            player_to_move=players,
            action_nodes=action_nodes,
        )
