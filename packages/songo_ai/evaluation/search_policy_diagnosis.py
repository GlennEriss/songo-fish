"""Métriques pures du diagnostic Search–Policy du Lot 27."""
from __future__ import annotations
import math
from typing import Sequence

def js_divergence(left:Sequence[float],right:Sequence[float])->float:
    mid=[(float(a)+float(b))/2 for a,b in zip(left,right)]
    def kl(p,q):return sum(a*math.log(a/max(b,1e-12)) for a,b in zip(p,q) if a>0)
    return .5*kl(left,mid)+.5*kl(right,mid)

def l1_distance(left:Sequence[float],right:Sequence[float])->float:return sum(abs(float(a)-float(b)) for a,b in zip(left,right))

def amplification_ratio(policy_js:float,visit_js:float,epsilon:float=1e-9)->float:return visit_js/max(policy_js,epsilon)

def first_divergence(left:Sequence[int|None],right:Sequence[int|None]):
    return next((i+1 for i,(a,b) in enumerate(zip(left,right)) if a!=b),None)

def search_category(policy_same:bool,search_same:bool)->str:
    return ("A" if policy_same and search_same else "B" if not policy_same and search_same else "C" if policy_same else "D")

def regret(q_values:Sequence[float|None],action:int,legal_mask:Sequence[bool])->float:
    legal=[i for i,x in enumerate(legal_mask) if x and q_values[i] is not None];return max(float(q_values[i]) for i in legal)-float(q_values[action])

def rwpm(policy:Sequence[float],q_values:Sequence[float|None],legal_mask:Sequence[bool])->float:
    return sum(float(policy[a])*regret(q_values,a,legal_mask) for a,x in enumerate(legal_mask) if x)

def search_recovery(actions_by_budget:Sequence[int])->bool:return len(set(actions_by_budget))>1 and actions_by_budget[-1]!=actions_by_budget[0]

def early_prior_lockin(prior_action:int,actions_by_budget:Sequence[int],q_values:Sequence[float|None],legal_mask:Sequence[bool],epsilon:float=.02)->bool:
    return all(a==prior_action for a in actions_by_budget) and regret(q_values,prior_action,legal_mask)>epsilon
