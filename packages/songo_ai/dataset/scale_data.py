"""Contrats reproductibles de construction de D_SCALE_V1 (Lot 25)."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

TEACHER_FIELDS=frozenset({"best_action","action_values","pv","depth","margin","evaluation","teacher_value","minimax_score"})


def physical_state_key(board:Sequence[int],player_to_move:int)->tuple[tuple[int,...],int]:
    return tuple(map(int,board)),int(player_to_move)


def position_only(record:Mapping)->dict:
    """Extrait exclusivement l'état physique; aucune annotation ne traverse."""
    state=record.get("state",record)
    board=state.get("board")
    player=state.get("player_to_move",record.get("player_to_move"))
    if board is None or player is None:raise ValueError("physical state is incomplete")
    return {"state":{"board":list(map(int,board)),"player_to_move":int(player)}}


def deduplicate_physical(records:Iterable[Mapping])->list[dict]:
    unique={}
    for record in records:
        clean=position_only(record);state=clean["state"];key=physical_state_key(state["board"],state["player_to_move"])
        unique.setdefault(key,clean)
    return [unique[k] for k in sorted(unique)]


def deterministic_stratified_sample(rows:Sequence[Mapping],count:int,*,seed:int,stratum_fields:Sequence[str])->list[Mapping]:
    if not 0<=count<=len(rows):raise ValueError("invalid sample size")
    groups=defaultdict(list)
    for row in rows:groups[tuple(row.get(f) for f in stratum_fields)].append(row)
    rng=random.Random(seed)
    for values in groups.values():values.sort(key=lambda x:x.get("position_hash",repr(x)));rng.shuffle(values)
    keys=sorted(groups,key=repr);out=[];cursor=0
    while len(out)<count:
        progressed=False
        for key in keys:
            if cursor<len(groups[key]):out.append(groups[key][cursor]);progressed=True
            if len(out)==count:break
        if not progressed:break
        cursor+=1
    return out


def intersection_matrix(named_keys:Mapping[str,set])->dict:
    names=sorted(named_keys);return {a:{b:len(named_keys[a]&named_keys[b]) for b in names} for a in names}


def shard_for(key:str,shards:int)->int:
    if shards<=0:raise ValueError("shards must be positive")
    return int(hashlib.sha256(key.encode()).hexdigest()[:16],16)%shards


def manifest_hash(files:Sequence[Path],configuration:Mapping)->str:
    digest=hashlib.sha256()
    digest.update(json.dumps(configuration,sort_keys=True,separators=(",",":")).encode())
    for path in sorted(map(Path,files),key=str):
        digest.update(str(path).encode());digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def validate_reanalysis_record(record:Mapping,*,expected_visits:int)->None:
    forbidden=TEACHER_FIELDS&set(record)
    if forbidden:raise ValueError(f"Teacher fields forbidden: {sorted(forbidden)}")
    if "value_target" in record or "z" in record:raise ValueError("reanalysis is Policy-only")
    if record.get("source_dataset")!="REANALYSIS":raise ValueError("invalid provenance")
    if sum(record["visit_counts"])!=expected_visits:raise ValueError("incorrect MCTS visit total")
    if not record.get("generation_model"):raise ValueError("generation model missing")

