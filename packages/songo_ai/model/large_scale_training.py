"""Contrats source-aware et splits immuables du Lot 26."""
from __future__ import annotations
import hashlib,random
from dataclasses import dataclass
from typing import Sequence

def split_bucket(identity:str,*,seed:int,buckets:int=10)->int:
    return int(hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()[:16],16)%buckets

def split_game_ids(game_ids:Sequence[str],*,seed:int,validation_bucket:int=0):
    train={g for g in game_ids if split_bucket(g,seed=seed)!=validation_bucket};val=set(game_ids)-train
    if not train or not val:raise ValueError("split must contain train and validation games")
    return train,val

def split_physical_hashes(hashes:Sequence[str],*,seed:int,validation_bucket:int=0):
    train={h for h in hashes if split_bucket(h,seed=seed)!=validation_bucket};val=set(hashes)-train
    if train&val:raise AssertionError("physical-state leakage")
    return train,val

@dataclass(frozen=True)
class SourceAwareSchedule:
    selfplay_per_batch:int
    reanalysis_per_batch:int
    strategic_per_step:int=0
    def __post_init__(self):
        if self.selfplay_per_batch<=0 or self.reanalysis_per_batch<=0 or self.strategic_per_step<0:raise ValueError("invalid source schedule")
    @property
    def batch_size(self):return self.selfplay_per_batch+self.reanalysis_per_batch

def deterministic_epoch_indices(size:int,count:int,*,seed:int,epoch:int)->list[int]:
    if size<=0 or count<=0:raise ValueError("size/count must be positive")
    rng=random.Random(seed+epoch*1_000_003);order=list(range(size));rng.shuffle(order)
    return [order[i%size] for i in range(count)]

def verify_value_sources(source_names:Sequence[str],value_present:Sequence[bool])->None:
    for source,present in zip(source_names,value_present):
        if present and source!="SELFPLAY":raise ValueError("Value target is allowed only for SELFPLAY")
