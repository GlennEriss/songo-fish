"""Objectif asymétrique correction/préservation du Lot 24."""
from __future__ import annotations

from typing import Sequence
import torch
import torch.nn.functional as F


def classify_pairs(g2_logits:Sequence[float],pairs:Sequence[tuple[int,int,float,float]]):
    correction=[];preservation=[]
    for preferred,other,weight,gap in pairs:
        parent_margin=float(g2_logits[preferred])-float(g2_logits[other]);entry=(preferred,other,float(weight),float(gap),parent_margin)
        (preservation if parent_margin>0 else correction).append(entry)
    return tuple(correction),tuple(preservation)


def correction_loss(logits:torch.Tensor,pairs_by_example)->torch.Tensor:
    terms=[];weights=[]
    for row,pairs in enumerate(pairs_by_example):
        for preferred,other,weight,_,_ in pairs:terms.append(weight*F.softplus(-(logits[row,preferred]-logits[row,other])));weights.append(weight)
    return torch.stack(terms).sum()/max(sum(weights),1e-12) if terms else logits.sum()*0


def preservation_loss(logits:torch.Tensor,pairs_by_example,*,rho:float)->torch.Tensor:
    if not 0<rho<=1:raise ValueError("rho must be in (0,1]")
    terms=[];weights=[]
    for row,pairs in enumerate(pairs_by_example):
        for preferred,other,weight,_,parent_margin in pairs:
            safe=rho*parent_margin;margin=logits[row,preferred]-logits[row,other];terms.append(weight*F.relu(safe-margin));weights.append(weight)
    return torch.stack(terms).sum()/max(sum(weights),1e-12) if terms else logits.sum()*0


def correct_preserve_metrics(logits:torch.Tensor,correction_pairs,preservation_pairs)->dict:
    corrections=correction_total=preserved=preservation_total=corrected_weight=damaged_weight=0.
    for row,pairs in enumerate(correction_pairs):
        for preferred,other,weight,_,_ in pairs:correction_total+=1;ok=float(logits[row,preferred])>float(logits[row,other]);corrections+=ok;corrected_weight+=weight*ok
    for row,pairs in enumerate(preservation_pairs):
        for preferred,other,weight,_,_ in pairs:preservation_total+=1;ok=float(logits[row,preferred])>float(logits[row,other]);preserved+=ok;damaged_weight+=weight*(not ok)
    correction_rate=corrections/correction_total if correction_total else None;preservation_rate=preserved/preservation_total if preservation_total else None;damage=1-preservation_rate if preservation_rate is not None else None
    return {"correction_pairs":int(correction_total),"preservation_pairs":int(preservation_total),"correction_rate":correction_rate,"preservation_rate":preservation_rate,"damage_rate":damage,"correction_efficiency":corrections/(preservation_total-preserved) if preservation_total>preserved else None,"strategic_utility":corrected_weight-damaged_weight,"corrected_weight":corrected_weight,"damaged_weight":damaged_weight}
