"""Durable arena orchestration primitives. No engine or MCTS dependency."""
from __future__ import annotations
import hashlib, json, os, sqlite3, tarfile, tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA=1
def canonical(value:Any)->bytes:return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def fingerprint(value:Any)->str:return hashlib.sha256(canonical(value)).hexdigest()
def file_sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()

@dataclass(frozen=True)
class Candidate:
 name:str;checkpoint_path:str;checkpoint_sha256:str;source_commit:str;architecture_id:str;model_config_fingerprint:str;training_experiment_id:str;training_step:int;policy_value_provenance:str="COMPLETE_NETWORK"
 def validate(self,root:Path|None=None)->None:
  if len(self.checkpoint_sha256)!=64 or self.training_step<0:raise ValueError("invalid candidate provenance")
  if root is not None:
   p=root/self.checkpoint_path
   if not p.is_file():raise FileNotFoundError(p)
   if file_sha256(p)!=self.checkpoint_sha256:raise ValueError("checkpoint SHA256 mismatch")

@dataclass(frozen=True)
class ExperimentManifest:
 experiment_id:str;protocol_version:str;source_commit:str;engine_fingerprint:str;candidates:tuple[Candidate,...];comparisons:tuple[dict,...];opening_ids:tuple[str,...];base_seed:int;statistical_config:dict;device:str;precision:str;software_versions:dict
 def scientific(self)->dict:return asdict(self)
 @property
 def fingerprint(self)->str:return fingerprint(self.scientific())

@dataclass(frozen=True)
class ScheduledGame:
 game_id:str;experiment_fingerprint:str;comparison_id:str;opening_id:str;pair_id:str;candidate_p1:str;candidate_p2:str;budget_p1:int;budget_p2:int;seed:int

def schedule(manifest:ExperimentManifest)->tuple[ScheduledGame,...]:
 out=[]
 for comp in manifest.comparisons:
  a,b=comp["candidate_a"],comp["candidate_b"];ba,bb=int(comp["budget_a"]),int(comp["budget_b"])
  for i,opening in enumerate(manifest.opening_ids):
   pair=f'{comp["comparison_id"]}:{opening}';seed=int.from_bytes(hashlib.sha256(f'{manifest.base_seed}:{pair}'.encode()).digest()[:8],"big")
   for side in (0,1):
    p1,p2=(a,b) if side==0 else (b,a);bp1,bp2=(ba,bb) if side==0 else (bb,ba)
    body={"experiment_fingerprint":manifest.fingerprint,"comparison_id":comp["comparison_id"],"opening_id":opening,"pair_id":pair,"side":side,"seed":seed,"budget_p1":bp1,"budget_p2":bp2,"candidate_p1":p1,"candidate_p2":p2}
    out.append(ScheduledGame(fingerprint(body),**{k:v for k,v in body.items() if k!="side"}))
 ids=[g.game_id for g in out]
 if len(ids)!=len(set(ids)):raise ValueError("game id collision")
 return tuple(out)

@dataclass(frozen=True)
class GameResult:
 game_id:str;experiment_fingerprint:str;comparison_id:str;opening_id:str;pair_id:str;candidate_p1:str;candidate_p2:str;budget_p1:int;budget_p2:int;seed:int;terminal_result:str;winner:str|None;draw:bool;game_length:int;terminal_state_fingerprint:str;game_trace_hash:str;execution_metadata:dict;result_schema_version:int=SCHEMA
 def validate(self,game:ScheduledGame)->None:
  for k in ("game_id","experiment_fingerprint","comparison_id","opening_id","pair_id","candidate_p1","candidate_p2","budget_p1","budget_p2","seed"):
   if getattr(self,k)!=getattr(game,k):raise ValueError(f"result/schedule mismatch: {k}")
  if self.result_schema_version!=SCHEMA or self.game_length<0:raise ValueError("invalid result schema")
  if self.terminal_result not in ("P1_WIN","P2_WIN","DRAW"):raise ValueError("non-terminal result")
  expected=None if self.terminal_result=="DRAW" else (self.candidate_p1 if self.terminal_result=="P1_WIN" else self.candidate_p2)
  if self.draw!=(expected is None) or self.winner!=expected:raise ValueError("inconsistent winner")

class ArenaStore:
 def __init__(self,path:Path):
  self.path=path;self.db=sqlite3.connect(path);self.db.execute("PRAGMA journal_mode=WAL");self.db.execute("PRAGMA synchronous=FULL")
  self.db.executescript("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v BLOB NOT NULL);CREATE TABLE IF NOT EXISTS schedule(game_id TEXT PRIMARY KEY,payload BLOB NOT NULL);CREATE TABLE IF NOT EXISTS results(game_id TEXT PRIMARY KEY,payload BLOB NOT NULL);CREATE TABLE IF NOT EXISTS violations(id INTEGER PRIMARY KEY AUTOINCREMENT,game_id TEXT,detail TEXT NOT NULL);")
 def initialize(self,m:ExperimentManifest,games:Iterable[ScheduledGame])->None:
  with self.db:
   row=self.db.execute("SELECT v FROM meta WHERE k='manifest'").fetchone();data=canonical(m.scientific())
   if row and row[0]!=data:raise ValueError("incompatible immutable manifest")
   self.db.execute("INSERT OR IGNORE INTO meta VALUES('manifest',?)",(data,));self.db.execute("INSERT OR IGNORE INTO meta VALUES('fingerprint',?)",(m.fingerprint,))
   for g in games:
    payload=canonical(asdict(g));old=self.db.execute("SELECT payload FROM schedule WHERE game_id=?",(g.game_id,)).fetchone()
    if old and old[0]!=payload:raise ValueError("schedule corruption")
    self.db.execute("INSERT OR IGNORE INTO schedule VALUES(?,?)",(g.game_id,payload))
 def complete(self,r:GameResult)->str:
  row=self.db.execute("SELECT payload FROM schedule WHERE game_id=?",(r.game_id,)).fetchone()
  if not row:raise ValueError("unknown game id")
  g=ScheduledGame(**json.loads(row[0]));r.validate(g);payload=canonical(asdict(r))
  try:
   with self.db:self.db.execute("INSERT INTO results VALUES(?,?)",(r.game_id,payload))
   return "COMMITTED"
  except sqlite3.IntegrityError:
   old=self.db.execute("SELECT payload FROM results WHERE game_id=?",(r.game_id,)).fetchone()[0]
   if old==payload:return "IDEMPOTENT"
   with self.db:self.db.execute("INSERT INTO violations(game_id,detail) VALUES(?,?)",(r.game_id,"conflicting completion"))
   raise ValueError("conflicting duplicate result")
 def pending(self)->tuple[str,...]:return tuple(r[0] for r in self.db.execute("SELECT game_id FROM schedule WHERE game_id NOT IN (SELECT game_id FROM results) ORDER BY game_id"))
 def paired_records(self)->dict[str,list[dict]]:
  rows=[json.loads(r[0]) for r in self.db.execute("SELECT payload FROM results ORDER BY game_id")];out={}
  for r in rows:out.setdefault(r["pair_id"],[]).append(r)
  return {k:v for k,v in out.items() if len(v)==2}

def require_execution(*,execute:bool,authorized_ids:set[str],manifest:ExperimentManifest,checkpoints_valid:bool)->None:
 if not execute:raise PermissionError("DRY_RUN=TRUE; execution not authorized")
 if manifest.experiment_id not in authorized_ids or not checkpoints_valid:raise PermissionError("execution readiness gate failed")

def export_store(store:ArenaStore,manifest:ExperimentManifest,destination:Path)->dict:
 if destination.exists():raise FileExistsError(destination)
 destination.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(dir=destination.parent) as td:
  root=Path(td);(root/"manifest.json").write_bytes(canonical(manifest.scientific()))
  rows=[json.loads(r[0]) for r in store.db.execute("SELECT payload FROM results ORDER BY game_id")]
  (root/"results.json").write_bytes(canonical(rows));(root/"integrity.json").write_bytes(canonical({"schema":SCHEMA,"schedule_fingerprint":fingerprint([asdict(g) for g in schedule(manifest)]),"completed":len(rows),"pending":len(store.pending())}))
  tmp=destination.with_suffix(destination.suffix+".tmp")
  with tarfile.open(tmp,"w:gz") as t:
   for p in sorted(root.iterdir()):t.add(p,arcname=p.name)
  with tarfile.open(tmp,"r:gz") as t:t.getmembers()
  os.replace(tmp,destination)
 return {"path":str(destination),"sha256":file_sha256(destination),"completed":len(rows),"pending":len(store.pending())}
