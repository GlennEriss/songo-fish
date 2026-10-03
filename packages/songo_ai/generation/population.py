"""Contrats minimaux pour une génération autonome multi-modèle.

Ce module ne change ni le moteur, ni MCTS, ni le SRN. Il formalise seulement
les rôles, la provenance et l'ordonnancement du futur pilote du Lot 30.
"""
from __future__ import annotations
import hashlib
import random
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class ModelRole(str, Enum):
    CHAMPION="CHAMPION";GENERATOR="GENERATOR";ACTIVE_GENERATOR="ACTIVE_GENERATOR";PROBATIONARY_GENERATOR="PROBATIONARY_GENERATOR";CHALLENGER="CHALLENGER";HISTORICAL="HISTORICAL";HISTORICAL_OPPONENT="HISTORICAL_OPPONENT";RETIRED="RETIRED";REJECTED="REJECTED"


@dataclass(frozen=True)
class RegisteredModel:
    model_id:str;fingerprint:str;roles:frozenset[ModelRole];policy_source:str;value_source:str
    def __post_init__(self):
        if not self.model_id or not self.fingerprint:raise ValueError("model identity and fingerprint are required")
        if not self.roles:raise ValueError("at least one role is required")
        if ModelRole.RETIRED in self.roles and ({ModelRole.GENERATOR,ModelRole.ACTIVE_GENERATOR}&self.roles):raise ValueError("retired model cannot remain a generator")


def validate_generator_pool(models:Sequence[RegisteredModel],*,max_active_generators:int=3)->bool:
    """Valide le contrat écologique d'une population de génération figée."""
    if max_active_generators<=0:raise ValueError("max_active_generators must be positive")
    champions=[m for m in models if ModelRole.CHAMPION in m.roles]
    active=[m for m in models if ModelRole.ACTIVE_GENERATOR in m.roles]
    if len(champions)!=1:raise ValueError("generator pool requires exactly one champion")
    if champions[0] not in active:raise ValueError("current champion must be an active generator")
    if len(active)>max_active_generators:raise ValueError("active generator limit exceeded")
    if any(ModelRole.RETIRED in m.roles for m in active):raise ValueError("retired model cannot be active")
    return True


@dataclass(frozen=True)
class MatchupQuota:
    p1_model_id:str;p2_model_id:str;p1_role:ModelRole;p2_role:ModelRole;games:int;provenance:str
    def __post_init__(self):
        if self.games<=0:raise ValueError("games must be positive")
        if not self.provenance:raise ValueError("provenance is required")


@dataclass(frozen=True)
class ScheduledGame:
    game_id:str;generation_id:str;p1_model_id:str;p2_model_id:str;p1_role:ModelRole;p2_role:ModelRole;seed:int;mcts_budget:int;provenance:str


def deterministic_schedule(generation_id:str,quotas:Sequence[MatchupQuota],*,seed:int,mcts_budget:int)->tuple[ScheduledGame,...]:
    if not generation_id or mcts_budget<=0:raise ValueError("generation_id and positive MCTS budget are required")
    rows=[]
    for quota_index,q in enumerate(quotas):
        for ordinal in range(q.games):
            raw=f"{seed}:{generation_id}:{quota_index}:{ordinal}:{q.p1_model_id}:{q.p2_model_id}"
            game_seed=int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8],"big")
            rows.append(ScheduledGame(f"{generation_id}-{quota_index:02d}-{ordinal:06d}",generation_id,q.p1_model_id,q.p2_model_id,q.p1_role,q.p2_role,game_seed,mcts_budget,q.provenance))
    random.Random(seed).shuffle(rows)
    return tuple(rows)


REQUIRED_PROVENANCE_FIELDS=("generation_id","game_id","p1_model_id","p2_model_id","p1_role","p2_role","p1_fingerprint","p2_fingerprint","mcts_budget","seed","result","generator_role","provenance")


def validate_game_provenance(record:Mapping)->None:
    missing=[x for x in REQUIRED_PROVENANCE_FIELDS if x not in record]
    if missing:raise ValueError(f"missing provenance fields: {missing}")
    if record["result"] not in (-1,0,1):raise ValueError("result must be terminal -1/0/+1")


def source_balanced_indices(source_ids:Sequence[str],batch_size:int,*,seed:int)->tuple[int,...]:
    """Échantillonne cycliquement chaque source afin que le volume brut ne domine pas."""
    if batch_size<=0 or not source_ids:raise ValueError("non-empty sources and positive batch size required")
    groups={}
    for i,s in enumerate(source_ids):groups.setdefault(str(s),[]).append(i)
    rng=random.Random(seed);keys=sorted(groups)
    for values in groups.values():rng.shuffle(values)
    cursors={k:0 for k in keys};out=[]
    for step in range(batch_size):
        key=keys[step%len(keys)];values=groups[key];out.append(values[cursors[key]%len(values)]);cursors[key]+=1
    return tuple(out)
