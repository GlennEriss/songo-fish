"""Sélection structurelle et mesures de stabilité du Lot 19."""
from __future__ import annotations
import hashlib, math, random
from collections import defaultdict
from typing import Mapping, Sequence

def structural_descriptor(board: Sequence[int], player: int, legal_mask: Sequence[bool], provenance: str="") -> dict:
    pits=board[:14]; p1=pits[:7]; p2=pits[7:14]; s1,s2=int(board[14]),int(board[15])
    return {"player_to_move":int(player),"legal_count":sum(bool(v) for v in legal_mask),"seeds_in_play":sum(pits),"store_p1":s1,"store_p2":s2,"store_difference":s1-s2,"nonempty_pits_p1":sum(v>0 for v in p1),"nonempty_pits_p2":sum(v>0 for v in p2),"territory_seeds_p1":sum(p1),"territory_seeds_p2":sum(p2),"seed_distribution":[sum(v==0 for v in pits),sum(1<=v<=3 for v in pits),sum(4<=v<=7 for v in pits),sum(v>=8 for v in pits)],"provenance":provenance}

def _bin(value: float, boundaries: Sequence[float]) -> int:
    return next((i for i,b in enumerate(boundaries) if value<=b),len(boundaries))

def diversity_stratum(d: Mapping) -> tuple:
    return (d["player_to_move"],d["legal_count"],_bin(d["seeds_in_play"],[10,20,35,50]),_bin(d["store_p1"],[7,17,27]),_bin(d["store_p2"],[7,17,27]),_bin(d["nonempty_pits_p1"],[1,3,5]),_bin(d["nonempty_pits_p2"],[1,3,5]),d.get("provenance",""))

def calibration_stratum(d: Mapping) -> tuple:
    return diversity_stratum(d)[:4]+(_bin(d["g2_entropy"],[.5,1.,1.5]),_bin(d["g2_margin"],[.01,.05,.15,.35]))

def balanced_sample(rows: Sequence[Mapping], count: int, *, seed: int, calibration: bool=False) -> list[int]:
    if not 0<count<=len(rows): raise ValueError("count must be in 1..len(rows)")
    groups=defaultdict(list); key_fn=calibration_stratum if calibration else diversity_stratum
    for i,row in enumerate(rows): groups[key_fn(row)].append(i)
    rng=random.Random(seed)
    for values in groups.values(): rng.shuffle(values)
    keys=sorted(groups,key=repr); selected=[]; cursor=0
    while len(selected)<count:
        progressed=False
        for key in keys:
            if cursor<len(groups[key]): selected.append(groups[key][cursor]); progressed=True
            if len(selected)==count: break
        if not progressed: break
        cursor+=1
    return selected

def position_hash(board: Sequence[int], player: int) -> str:
    return hashlib.sha256((",".join(map(str,board))+f"|{player}").encode()).hexdigest()

def jensen_shannon(left: Sequence[float], right: Sequence[float]) -> float:
    middle=[(float(a)+float(b))/2 for a,b in zip(left,right)]
    def kl(p,q): return sum(a*math.log(a/b) for a,b in zip(p,q) if a>0 and b>0)
    return (kl(left,middle)+kl(right,middle))/2

def policy_stability(left: Sequence[float], right: Sequence[float], legal_mask: Sequence[bool]) -> dict:
    legal=[i for i,v in enumerate(legal_mask) if v]; rank_l=sorted(legal,key=lambda i:(-left[i],i)); rank_r=sorted(legal,key=lambda i:(-right[i],i))
    return {"js":jensen_shannon(left,right),"argmax_agreement":rank_l[0]==rank_r[0],"top2_overlap":len(set(rank_l[:2])&set(rank_r[:2]))/min(2,len(legal)),"top1_probability_delta":abs(left[rank_l[0]]-right[rank_l[0]])}
