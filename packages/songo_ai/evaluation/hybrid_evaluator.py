"""Evaluateurs Policy/Value hybrides reserves aux ablations experimentales."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EvaluatorCounters:
    """Cout logique depuis le dernier reset.

    ``model_evaluations`` compte les etats evalues par l'interface hybride.
    ``network_forward_calls`` compte les forwards reels des modeles sources.
    """

    model_evaluations: int
    network_forward_calls: int


class HybridPolicyValueEvaluator(nn.Module):
    """Retourne la Policy d'un modele et la Value d'un autre.

    Si les deux sources sont le meme objet, un seul forward est execute. Avec
    ``neutral_value=True``, la Value non terminale vaut exactement zero et seul
    le modele de Policy est evalue. Les terminaux restent geres par le moteur
    dans MCTS et ne passent jamais dans cet evaluateur.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module | None = None,
        *,
        neutral_value: bool = False,
        name: str = "hybrid",
    ) -> None:
        super().__init__()
        if neutral_value and value_model is not None:
            raise ValueError("neutral value cannot also have a value model")
        if not neutral_value and value_model is None:
            raise ValueError("a value model is required unless neutral_value=True")
        if not name:
            raise ValueError("name must not be empty")
        self.policy_model = policy_model
        self.value_model = value_model
        self.neutral_value = bool(neutral_value)
        self.name = name
        self._model_evaluations = 0
        self._network_forward_calls = 0

    @property
    def counters(self) -> EvaluatorCounters:
        return EvaluatorCounters(
            model_evaluations=self._model_evaluations,
            network_forward_calls=self._network_forward_calls,
        )

    def reset_counters(self) -> None:
        self._model_evaluations = 0
        self._network_forward_calls = 0

    def forward(self, graph):
        self._model_evaluations += int(graph.batch_size)
        if self.neutral_value:
            policy_logits, _ = self.policy_model(graph)
            self._network_forward_calls += 1
            value = torch.zeros(
                policy_logits.shape[0],
                device=policy_logits.device,
                dtype=policy_logits.dtype,
            )
            return policy_logits, value

        if self.policy_model is self.value_model:
            policy_logits, value = self.policy_model(graph)
            self._network_forward_calls += 1
            return policy_logits, value

        policy_logits, _ = self.policy_model(graph)
        _, value = self.value_model(graph)
        self._network_forward_calls += 2
        return policy_logits, value


class ScaledValueEvaluator(nn.Module):
    """Conserve la Policy et multiplie seulement la Value par ``alpha``."""

    def __init__(self, model: nn.Module, alpha: float, *, name: str = "scaled-value") -> None:
        super().__init__()
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not name:
            raise ValueError("name must not be empty")
        self.model = model
        self.alpha = float(alpha)
        self.name = name
        self._model_evaluations = 0
        self._network_forward_calls = 0

    @property
    def counters(self) -> EvaluatorCounters:
        return EvaluatorCounters(
            model_evaluations=self._model_evaluations,
            network_forward_calls=self._network_forward_calls,
        )

    def reset_counters(self) -> None:
        self._model_evaluations = 0
        self._network_forward_calls = 0

    def forward(self, graph):
        self._model_evaluations += int(graph.batch_size)
        policy_logits, value = self.model(graph)
        self._network_forward_calls += 1
        return policy_logits, value * self.alpha


class DepthPolicyValueEvaluator(nn.Module):
    """Ablation diagnostique : Policy racine/descendants et Value indépendantes."""
    def __init__(self, root_policy_model: nn.Module, descendant_policy_model: nn.Module, value_model: nn.Module):
        super().__init__();self.root_policy_model=root_policy_model;self.descendant_policy_model=descendant_policy_model;self.value_model=value_model
    def forward_at_depth(self, graph, depth: int):
        policy_model=self.root_policy_model if depth==0 else self.descendant_policy_model
        policy,_=policy_model(graph);_,value=self.value_model(graph);return policy,value
    def forward(self, graph):return self.forward_at_depth(graph,0)
