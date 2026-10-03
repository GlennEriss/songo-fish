"""Mesures diagnostiques de ranking et de criticité pour le Lot 22."""
from __future__ import annotations

import math
from itertools import combinations
from typing import Sequence

from .policy_regret import legal_ranking


def legal_rank_map(policy:Sequence[float],legal_mask:Sequence[bool])->dict[int,int]:
    return {action:rank for rank,action in enumerate(legal_ranking(policy,legal_mask),1)}


def top_k_set(policy:Sequence[float],legal_mask:Sequence[bool],k:int)->frozenset[int]:
    if k<=0:raise ValueError("k must be positive")
    return frozenset(legal_ranking(policy,legal_mask)[:k])


def pairwise_inversions(parent:Sequence[float],candidate:Sequence[float],legal_mask:Sequence[bool])->tuple[tuple[int,int],...]:
    legal=[a for a,ok in enumerate(legal_mask) if ok];result=[]
    for a,b in combinations(legal,2):
        parent_sign=(parent[a]>parent[b])-(parent[a]<parent[b]);candidate_sign=(candidate[a]>candidate[b])-(candidate[a]<candidate[b])
        if parent_sign and candidate_sign and parent_sign!=candidate_sign:result.append((a,b))
    return tuple(result)


def strategic_gap(q_values:Sequence[float|None],a:int,b:int)->float:
    if q_values[a] is None or q_values[b] is None:raise ValueError("strategic gap requires legal Q values")
    return abs(float(q_values[a])-float(q_values[b]))


def strategically_weighted_inversions(parent,candidate,legal_mask,q_values)->float:
    return sum(strategic_gap(q_values,a,b) for a,b in pairwise_inversions(parent,candidate,legal_mask))


def decision_regret(q_values:Sequence[float|None],legal_mask:Sequence[bool],action:int)->float:
    legal=[a for a,ok in enumerate(legal_mask) if ok]
    if action not in legal:raise ValueError("decision action must be legal")
    best=max(float(q_values[a]) for a in legal if q_values[a] is not None);value=float(q_values[action]);regret=max(0.,best-value)
    if not math.isfinite(regret):raise FloatingPointError("non-finite regret")
    return regret


def classify_flip(delta_regret:float,epsilon:float=.02)->str:
    if epsilon<0:raise ValueError("epsilon must be non-negative")
    if delta_regret < -epsilon:return "BENEFICIAL_FLIP"
    if delta_regret > epsilon:return "HARMFUL_FLIP"
    return "NEUTRAL_FLIP"


def kendall_tau_from_rankings(first:Sequence[int],second:Sequence[int])->float:
    if set(first)!=set(second):raise ValueError("rankings must contain the same actions")
    if len(first)<2:return 1.
    pos={a:i for i,a in enumerate(second)};concordant=discordant=0
    for a,b in combinations(first,2):
        if pos[a]<pos[b]:concordant+=1
        else:discordant+=1
    return (concordant-discordant)/(concordant+discordant)


def search_amplification(policy_js:float,visit_js:float,selected_changed:bool)->dict:
    return {"policy_js":float(policy_js),"visit_js":float(visit_js),"ratio":float(visit_js/max(policy_js,1e-12)),"selected_action_changed":bool(selected_changed)}
