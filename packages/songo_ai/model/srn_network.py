"""Reseau relationnel Songo (SRN) v0.1.

Cette baseline ne remplace pas le MLP historique. Elle consomme les graphes
construits par :mod:`songo_ai.model.srn_graph` et produit deux sorties :

* sept logits de politique, un par action locale du joueur au trait ;
* une valeur scalaire bornee dans ``[-1, 1]`` depuis cette meme perspective.

Le masquage des actions illegales reste une operation explicite, exterieure au
``forward`` : le moteur est l'unique autorite sur la legalite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from songo_ai.songo.rules import NUM_ACTIONS

from .srn_graph import (
    BASE_RELATIONS,
    GLOBAL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    NUM_NODES,
    SongoGraph,
    SongoGraphBatch,
)


@dataclass(frozen=True)
class SRNConfig:
    """Configuration minimale et explicite de la baseline SRN v0.1."""

    node_feature_dim: int = NODE_FEATURE_DIM
    global_feature_dim: int = GLOBAL_FEATURE_DIM
    hidden_dim: int = 64
    num_relational_blocks: int = 3
    pooling: str = "mean"
    relation_types: tuple[str, ...] = BASE_RELATIONS

    def __post_init__(self) -> None:
        if self.node_feature_dim <= 0:
            raise ValueError("node_feature_dim must be positive")
        if self.global_feature_dim <= 0:
            raise ValueError("global_feature_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_relational_blocks <= 0:
            raise ValueError("num_relational_blocks must be positive")
        if self.pooling != "mean":
            raise ValueError("SRN v0.1 implements only mean pooling")
        relation_types = tuple(self.relation_types)
        if not relation_types:
            raise ValueError("at least one relation type is required")
        if len(set(relation_types)) != len(relation_types):
            raise ValueError("relation_types must not contain duplicates")
        unknown = set(relation_types) - set(BASE_RELATIONS)
        if unknown:
            raise ValueError(f"relations not implemented in SRN v0.1: {sorted(unknown)}")
        object.__setattr__(self, "relation_types", relation_types)


class RelationalBlock(nn.Module):
    """Un bloc message-passing avec parametres propres a chaque relation."""

    def __init__(self, hidden_dim: int, relation_types: Sequence[str]) -> None:
        super().__init__()
        self.relation_types = tuple(relation_types)
        self.message_mlps = nn.ModuleDict(
            {
                relation: nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for relation in self.relation_types
            }
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        global_context: torch.Tensor,
        edges_by_relation: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if node_embeddings.ndim != 3:
            raise ValueError("node_embeddings must have shape [B, N, H]")
        if global_context.ndim != 2:
            raise ValueError("global_context must have shape [B, H]")

        aggregated = torch.zeros_like(node_embeddings)
        for relation in self.relation_types:
            if relation not in edges_by_relation:
                raise ValueError(f"missing graph relation: {relation}")
            edges = edges_by_relation[relation]
            if edges.ndim != 2 or edges.shape[0] != 2:
                raise ValueError(f"edges for {relation!r} must have shape [2, E]")
            source, destination = edges[0], edges[1]
            source_h = node_embeddings.index_select(1, source)
            destination_h = node_embeddings.index_select(1, destination)
            messages = self.message_mlps[relation](torch.cat((source_h, destination_h), dim=-1))
            relation_sum = torch.zeros_like(node_embeddings)
            relation_sum.index_add_(1, destination, messages)
            aggregated = aggregated + relation_sum

        expanded_global = global_context.unsqueeze(1).expand(-1, node_embeddings.shape[1], -1)
        update = self.update_mlp(torch.cat((node_embeddings, aggregated, expanded_global), dim=-1))
        return self.layer_norm(node_embeddings + update)


class SongoRelationalNetwork(nn.Module):
    """Policy-Value Network relationnel pour les 14 cases physiques du Songo."""

    def __init__(self, config: SRNConfig | None = None) -> None:
        super().__init__()
        self.config = config or SRNConfig()
        hidden_dim = self.config.hidden_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(self.config.node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(self.config.global_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.relational_blocks = nn.ModuleList(
            [
                RelationalBlock(hidden_dim, self.config.relation_types)
                for _ in range(self.config.num_relational_blocks)
            ]
        )
        self.state_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Un seul MLP partage ses parametres entre les sept actions.
        self.policy_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, graph: SongoGraph | SongoGraphBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(graph, SongoGraph):
            graph = graph.as_batch()
        if graph.node_features.ndim != 3:
            raise ValueError("node_features must have shape [B, 14, F]")
        if graph.node_features.shape[1] != NUM_NODES:
            raise ValueError(f"node_features must contain exactly {NUM_NODES} nodes")
        if graph.global_features.ndim != 2:
            raise ValueError("global_features must have shape [B, G]")
        if graph.node_features.shape[-1] != self.config.node_feature_dim:
            raise ValueError("node feature dimension does not match SRNConfig")
        if graph.global_features.shape[-1] != self.config.global_feature_dim:
            raise ValueError("global feature dimension does not match SRNConfig")
        if graph.action_nodes.shape != (graph.batch_size, NUM_ACTIONS):
            raise ValueError(f"action_nodes must have shape [B, {NUM_ACTIONS}]")

        node_h = self.node_encoder(graph.node_features)
        global_h = self.global_encoder(graph.global_features)
        for block in self.relational_blocks:
            node_h = block(node_h, global_h, graph.edges_by_relation)

        if self.config.pooling == "mean":
            pooled = node_h.mean(dim=1)
        else:  # Garde defensive ; SRNConfig rejette deja les autres valeurs.
            raise RuntimeError(f"unsupported pooling: {self.config.pooling}")
        state_h = self.state_fusion(torch.cat((pooled, global_h), dim=-1))

        gather_index = graph.action_nodes.unsqueeze(-1).expand(-1, -1, self.config.hidden_dim)
        action_h = torch.gather(node_h, dim=1, index=gather_index)
        state_for_actions = state_h.unsqueeze(1).expand(-1, NUM_ACTIONS, -1)
        policy_logits = self.policy_mlp(torch.cat((action_h, state_for_actions), dim=-1)).squeeze(-1)
        value = self.value_mlp(state_h).squeeze(-1)
        return policy_logits, value


def mask_policy_logits(
    policy_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    invalid_logit: float = -1.0e9,
) -> torch.Tensor:
    """Masque les logits avec une valeur finie pour eviter ``0 * -inf``."""

    if policy_logits.ndim != 2 or policy_logits.shape[-1] != NUM_ACTIONS:
        raise ValueError(f"policy_logits must have shape [B, {NUM_ACTIONS}]")
    if legal_mask.shape != policy_logits.shape:
        raise ValueError("legal_mask must have the same shape as policy_logits")
    if not torch.isfinite(policy_logits).all():
        raise ValueError("policy_logits must be finite")
    if not torch.isfinite(torch.tensor(invalid_logit)) or invalid_logit >= 0.0:
        raise ValueError("invalid_logit must be a finite negative value")
    mask = legal_mask.to(device=policy_logits.device, dtype=torch.bool)
    if not mask.any(dim=-1).all():
        raise ValueError("each position must contain at least one legal action")
    dtype_limit = torch.finfo(policy_logits.dtype).min
    safe_invalid_logit = max(invalid_logit, dtype_limit)
    return policy_logits.masked_fill(~mask, safe_invalid_logit)


def policy_probabilities(policy_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Retourne une distribution normalisee avec des zeros illegaux exacts."""

    mask = legal_mask.to(device=policy_logits.device, dtype=torch.bool)
    masked_logits = mask_policy_logits(policy_logits, mask)
    probabilities = torch.softmax(masked_logits, dim=-1)
    probabilities = probabilities * mask.to(dtype=probabilities.dtype)
    return probabilities / probabilities.sum(dim=-1, keepdim=True)


def masked_policy_cross_entropy(
    policy_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    policy_target: torch.Tensor,
) -> torch.Tensor:
    """Entropie croisee distributionnelle sure pour la cible ``pi`` de MCTS."""

    if policy_target.shape != policy_logits.shape:
        raise ValueError("policy_target must have the same shape as policy_logits")
    if not torch.isfinite(policy_target).all() or (policy_target < 0).any():
        raise ValueError("policy_target must contain finite non-negative values")
    mask = legal_mask.to(device=policy_logits.device, dtype=torch.bool)
    if (policy_target.masked_select(~mask) != 0).any():
        raise ValueError("policy_target must be zero on illegal actions")
    if not torch.allclose(
        policy_target.sum(dim=-1),
        torch.ones(policy_target.shape[0], device=policy_target.device, dtype=policy_target.dtype),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("each policy_target row must sum to 1")
    log_policy = torch.log_softmax(mask_policy_logits(policy_logits, mask), dim=-1)
    return -(policy_target * log_policy).sum(dim=-1).mean()
