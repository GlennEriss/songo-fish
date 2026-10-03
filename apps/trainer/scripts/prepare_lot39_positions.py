#!/usr/bin/env python3
"""Sélectionne 256 états autonomes Lot34, sans importer leurs labels."""
import argparse,hashlib,json
from pathlib import Path
from songo_ai.dataset import RawSongoState
from songo_ai.songo.rules import SongoLegacyGame
from run_srn_lot12 import write_json

def main():
 p=argparse.ArgumentParser();p.add_argument("--source",type=Path,default=Path("data/experiments/lot34_g4_training/pool_data"));p.add_argument("--output",type=Path,default=Path("data/colab_bridge/lot39_benchmark_positions.json"));a=p.parse_args();selected={};files=sorted(a.source.glob("part-*.jsonl"))
 if len(files)<256:raise RuntimeError("insufficient autonomous shards")
 for index,path in enumerate(files):
  target=2+(index*37)%97
  with path.open() as f:
   for line_number,line in enumerate(f,1):
    if line_number<target:continue
    row=json.loads(line)
    if row.get("record_type")=="manifest":continue
    raw=row["state"];state=RawSongoState(tuple(raw["board"]),int(raw["player_to_move"]));game=SongoLegacyGame.from_state(state.to_engine_state());game.normalize_terminal()
    if game.finished or not game.legal_local_actions():continue
    key=hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    selected.setdefault(key,{"board":list(state.board),"player_to_move":state.player_to_move,"source_shard":path.name,"source_line":line_number,"source_game_id":row.get("metadata",{}).get("game_id"),"source_ply":row.get("metadata",{}).get("ply")});break
  if len(selected)>=256:break
 if len(selected)!=256:raise RuntimeError(f"only {len(selected)} unique valid positions found")
 payload={"count":256,"provenance":{"dataset":"LOT34_POOL_G2_SELFPLAY","family":"autonomous self-play","source":str(a.source),"selection":"one deterministic offset per sorted shard","teacher_labels_used":False,"policy_labels_used":False,"value_labels_used":False},"states":list(selected.values())};a.output.parent.mkdir(parents=True,exist_ok=True);write_json(a.output,payload);print(a.output)
if __name__=="__main__":main()
