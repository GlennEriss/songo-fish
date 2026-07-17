"""Fonction de perte multi-tetes (section 7.3) :

loss = lambda_p * CE(policy, policy_target)
     + lambda_w * CE(wdl, wdl_target)
     + lambda_q * masked_loss(q_actions, action_values)
     + lambda_r * regularisation

policy_target/wdl_target sont des DISTRIBUTIONS (pas des classes dures) :
on utilise donc l'entropie croisee "soft target" (-sum(target*log_softmax))
et non nn.CrossEntropyLoss, qui attend un indice de classe. La
regularisation (lambda_r) est deleguee au weight_decay de l'optimiseur
(AdamW) plutot que dupliquee ici, pour eviter de calculer deux fois la
meme penalite L2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def soft_cross_entropy(logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1).mean()


def masked_q_loss(q_pred: torch.Tensor, action_values: torch.Tensor, q_mask: torch.Tensor) -> torch.Tensor:
    if q_mask.sum() == 0:
        return q_pred.new_zeros(())
    diff = (q_pred - action_values) ** 2
    return diff[q_mask].mean()


@dataclass
class LossWeights:
    policy: float = 1.0
    wdl: float = 1.0
    q_actions: float = 0.5


@dataclass
class LossBreakdown:
    total: torch.Tensor
    policy: torch.Tensor
    wdl: torch.Tensor
    q_actions: torch.Tensor


def compute_loss(policy_logits, wdl_logits, q_pred, batch, weights: LossWeights = LossWeights()) -> LossBreakdown:
    policy_loss = soft_cross_entropy(policy_logits, batch["policy_target"])
    wdl_loss = soft_cross_entropy(wdl_logits, batch["wdl_target"])
    q_loss = masked_q_loss(q_pred, batch["action_values"], batch["q_mask"])
    total = weights.policy * policy_loss + weights.wdl * wdl_loss + weights.q_actions * q_loss
    return LossBreakdown(total=total, policy=policy_loss, wdl=wdl_loss, q_actions=q_loss)
