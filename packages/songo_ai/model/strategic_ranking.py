"""Loss et métriques pairwise stratégiques du Lot 23."""
from __future__ import annotations

import math
from itertools import combinations
from typing import Sequence

import torch
import torch.nn.functional as F


def strategic_weight(gap:float,epsilon:float,scale:float)->float:
    if epsilon<0 or scale<=epsilon:raise ValueError("invalid epsilon/scale")
    return min(1.,max(0.,(abs(float(gap))-epsilon)/(scale-epsilon)))


def build_legal_pairs(q_values:Sequence[float|None],legal_mask:Sequence[bool],*,epsilon:float,scale:float,stable_pairs:set[tuple[int,int]]|None=None):
    legal=[a for a,ok in enumerate(legal_mask) if ok];pairs=[]
    for a,b in combinations(legal,2):
        if q_values[a] is None or q_values[b] is None:raise ValueError("legal action requires Qdiag")
        gap=float(q_values[a])-float(q_values[b]);key=(min(a,b),max(a,b))
        if abs(gap)<=epsilon or (stable_pairs is not None and key not in stable_pairs):continue
        preferred,other=(a,b) if gap>0 else (b,a);weight=strategic_weight(gap,epsilon,scale)
        if weight>0:pairs.append((preferred,other,weight,abs(gap)))
    return tuple(pairs)


def strategic_ranking_loss(logits:torch.Tensor,pairs_by_example:Sequence[Sequence[tuple[int,int,float,float]]])->torch.Tensor:
    terms=[];weights=[]
    for row,pairs in enumerate(pairs_by_example):
        for preferred,other,weight,_ in pairs:
            terms.append(F.softplus(-(logits[row,preferred]-logits[row,other]))*weight);weights.append(weight)
    if not terms:return logits.sum()*0.
    return torch.stack(terms).sum()/max(sum(weights),1e-12)


def pairwise_metrics(logits:torch.Tensor,pairs_by_example:Sequence[Sequence[tuple[int,int,float,float]]],g2_logits:torch.Tensor|None=None)->dict:
    total=correct=weight_total=weight_correct=corrections=correction_total=preserved=preservation_total=broken_weight=corrected_weight=0.
    for row,pairs in enumerate(pairs_by_example):
        for preferred,other,weight,_ in pairs:
            total+=1;ok=float(logits[row,preferred])>float(logits[row,other]);correct+=ok;weight_total+=weight;weight_correct+=weight*ok
            if g2_logits is not None:
                g2ok=float(g2_logits[row,preferred])>float(g2_logits[row,other])
                if g2ok:preservation_total+=1;preserved+=ok;broken_weight+=weight*(not ok)
                else:correction_total+=1;corrections+=ok;corrected_weight+=weight*ok
    return {"pairs":int(total),"pairwise_accuracy":correct/total if total else None,"weighted_pairwise_accuracy":weight_correct/weight_total if weight_total else None,"correction_rate":corrections/correction_total if correction_total else None,"preservation_rate":preserved/preservation_total if preservation_total else None,"net_strategic_gain":corrected_weight-broken_weight,"corrected_weight":corrected_weight,"broken_weight":broken_weight}
