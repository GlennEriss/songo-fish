"""Losses et pondérations bornées pour l'expérience dual-source Lot 20."""
from __future__ import annotations
import math
from typing import Mapping, Sequence
import torch
from .srn_network import mask_policy_logits

def _bounded_mean_one(values: Sequence[float], lower: float, upper: float) -> list[float]:
    if not values:return []
    result=[min(upper,max(lower,float(v))) for v in values]
    for _ in range(64):
        delta=1.0-sum(result)/len(result)
        if abs(delta)<1e-12:break
        result=[min(upper,max(lower,v+delta)) for v in result]
    return result

def bounded_regret_weights(regrets: Sequence[float|None], *, alpha: float, q95: float) -> list[float]:
    if alpha<0 or q95<=0 or not math.isfinite(q95): raise ValueError("invalid alpha or q95")
    raw=[1.0 if r is None else 1.0+alpha*min(1.0,max(0.0,float(r)/q95)) for r in regrets]
    return _bounded_mean_one(raw,1.0/(1.0+alpha) if alpha else 1.0,1.0+alpha)

def confidence_score(metadata: Mapping, legal_count: int) -> float:
    data=metadata.get("search_confidence",{}); margin=float(data.get("visit_margin",0.0)); entropy=float(data.get("target_entropy",0.0)); normalized_entropy=entropy/math.log(legal_count) if legal_count>1 else 0.0
    return min(1.0,max(0.0,0.5*margin+0.5*(1.0-normalized_entropy)))

def bounded_confidence_weights(metadata: Sequence[Mapping], legal_counts: Sequence[int], *, beta: float=1.0) -> list[float]:
    if beta<0 or beta>1: raise ValueError("beta must be in [0,1]")
    scores=[confidence_score(m,n) for m,n in zip(metadata,legal_counts)]; mean=sum(scores)/len(scores) if scores else 0.0
    return _bounded_mean_one([1.0+beta*(s-mean) for s in scores],0.5,1.5)

def weighted_policy_cross_entropy(logits: torch.Tensor, mask: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if weights.shape!=(logits.shape[0],): raise ValueError("one weight per example is required")
    per_example=-(target*torch.log_softmax(mask_policy_logits(logits,mask),dim=-1)).sum(dim=-1)
    return (per_example*weights).mean()

def parent_policy_kl(
    candidate_logits: torch.Tensor,
    parent_logits: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """KL(P_parent || P_candidate), normalisée sur les seules actions légales.

    Le parent est systématiquement détaché : ce terme ne peut donc jamais
    propager de gradient dans G2, même si l'appelant oublie ``torch.no_grad``.
    """
    parent_logp=torch.log_softmax(mask_policy_logits(parent_logits.detach(),legal_mask),dim=-1)
    candidate_logp=torch.log_softmax(mask_policy_logits(candidate_logits,legal_mask),dim=-1)
    parent_p=parent_logp.exp()
    terms=torch.where(legal_mask,parent_p*(parent_logp-candidate_logp),torch.zeros_like(parent_p))
    result=terms.sum(dim=-1).mean()
    if not torch.isfinite(result): raise FloatingPointError("non-finite parent-policy KL")
    return result

def dual_source_loss(rl_logits,rl_value,rl_mask,rl_target,rl_z,regret_weights,re_logits,re_mask,re_target,confidence_weights,*,lambda_rl=1.0,lambda_re=1.0,lambda_value=1.0):
    rl_policy=weighted_policy_cross_entropy(rl_logits,rl_mask,rl_target,regret_weights); re_policy=weighted_policy_cross_entropy(re_logits,re_mask,re_target,confidence_weights); value=(rl_value.reshape(-1)-rl_z).square().mean()
    return {"total":lambda_rl*rl_policy+lambda_re*re_policy+lambda_value*value,"policy_rl":rl_policy,"policy_re":re_policy,"value_rl":value}

def regularized_dual_source_loss(
    rl_logits,rl_value,rl_mask,rl_target,rl_z,regret_weights,
    re_logits,re_mask,re_target,confidence_weights,
    parent_rl_logits,parent_re_logits,*,lambda_rl=1.0,lambda_re=1.0,
    lambda_value=1.0,beta=0.0,
):
    """Loss Lot 21 : dual-source, Value D_RL-only et ancrage Policy G2."""
    base=dual_source_loss(
        rl_logits,rl_value,rl_mask,rl_target,rl_z,regret_weights,
        re_logits,re_mask,re_target,confidence_weights,
        lambda_rl=lambda_rl,lambda_re=lambda_re,lambda_value=lambda_value,
    )
    parent_rl=parent_policy_kl(rl_logits,parent_rl_logits,rl_mask)
    parent_re=parent_policy_kl(re_logits,parent_re_logits,re_mask)
    parent=.5*(parent_rl+parent_re)
    return {**base,"total":base["total"]+beta*parent,"parent_rl":parent_rl,"parent_re":parent_re,"parent":parent}
