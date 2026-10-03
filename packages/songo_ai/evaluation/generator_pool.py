"""Mesures pures de couverture du pilote generator-pool."""
from __future__ import annotations
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

def physical_state_key(example)->tuple:
    return tuple(example.state.board),int(example.state.player_to_move)

def state_coverage(examples:Sequence)->dict[str,float|int]:
    raw=len(examples);unique=len({physical_state_key(x) for x in examples})
    return {"raw_positions":raw,"unique_physical_states":unique,"duplicate_positions":raw-unique,"duplicate_rate":1-unique/raw if raw else 0.,"unique_per_10000_raw":10000*unique/raw if raw else 0.}

def source_state_sets(examples:Iterable,field:str="source_type")->dict[str,set[tuple]]:
    out={}
    for x in examples:out.setdefault(str(x.metadata[field]),set()).add(physical_state_key(x))
    return out

def marginal_novelty(ordered_sources:Sequence[str],sets:Mapping[str,set])->dict[str,dict[str,float|int]]:
    seen=set();out={}
    for source in ordered_sources:
        current=set(sets.get(source,set()));new=current-seen;out[source]={"unique_states":len(current),"marginal_unique_states":len(new),"marginal_fraction":len(new)/len(current) if current else 0.};seen|=current
    return out

def crossplay_only_states(sets:Mapping[str,set],cross_key:str="CROSS_PLAY")->set:
    homogeneous=set().union(*(v for k,v in sets.items() if k!=cross_key));return set(sets.get(cross_key,set()))-homogeneous

def action_metrics(examples:Sequence)->dict:
    counts=Counter(int(x.metadata["action_played"]) for x in examples);total=sum(counts.values());probs=[counts.get(i,0)/total for i in range(7)] if total else [0.]*7
    return {"counts":{str(i):counts.get(i,0) for i in range(7)},"frequencies":probs,"entropy":-sum(p*math.log(p) for p in probs if p),"mean_legal_actions":sum(sum(x.legal_mask) for x in examples)/len(examples) if examples else 0.}

def overlap(left:set,right:set)->dict[str,float|int]:
    inter=left&right;union=left|right
    return {"intersection":len(inter),"left_only":len(left-right),"right_only":len(right-left),"union":len(union),"jaccard":len(inter)/len(union) if union else 1.}
