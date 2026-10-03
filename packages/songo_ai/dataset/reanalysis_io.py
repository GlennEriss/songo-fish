"""JSONL versionné, distinct de D_RL, pour les cibles Policy réanalysées."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Iterator
from .reanalysis_schema import ReanalysisPolicyExample
from .selfplay_schema import RawSongoState

FORMAT="songo_policy_reanalysis_jsonl"; VERSION=1

def record_from_reanalysis(e: ReanalysisPolicyExample) -> dict:
    return {"record_type":"reanalysis_policy_example","state":{"board":list(e.state.board),"player_to_move":e.state.player_to_move},"legal_mask":list(e.legal_mask),"visit_counts":list(e.visit_counts),"policy_target":list(e.policy_target),"search_metadata":dict(e.search_metadata),"source_position_metadata":dict(e.source_position_metadata)}

def reanalysis_from_record(row: dict) -> ReanalysisPolicyExample:
    if row.get("record_type")!="reanalysis_policy_example": raise ValueError("invalid record type")
    if "value_target" in row or "z" in row: raise ValueError("value targets are forbidden")
    state=row["state"]
    return ReanalysisPolicyExample(RawSongoState(tuple(state["board"]),int(state["player_to_move"])),tuple(row["legal_mask"]),tuple(row["visit_counts"]),tuple(row["policy_target"]),row["search_metadata"],row["source_position_metadata"])

def write_reanalysis_jsonl(path: str|Path, examples: Iterable[ReanalysisPolicyExample], *, metadata: dict) -> int:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); count=0
    with path.open("w",encoding="utf-8") as stream:
        stream.write(json.dumps({"record_type":"manifest","format":FORMAT,"version":VERSION,"dataset_family":"D_REANALYSIS_POLICY","metadata":metadata},sort_keys=True)+"\n")
        for e in examples: stream.write(json.dumps(record_from_reanalysis(e),sort_keys=True)+"\n"); count+=1
    return count

def iter_reanalysis_jsonl(path: str|Path) -> Iterator[ReanalysisPolicyExample]:
    with Path(path).open(encoding="utf-8") as stream:
        header=json.loads(next(stream))
        if header.get("format")!=FORMAT or header.get("version")!=VERSION: raise ValueError("unsupported manifest")
        for line in stream:
            if line.strip(): yield reanalysis_from_record(json.loads(line))
