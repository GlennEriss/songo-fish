from dataclasses import replace
import pytest
from songo_ai.evaluation.arena_runner import *

def manifest(tmp_path):
 c=lambda n:Candidate(n,"x.pt","a"*64,"c","srn","m","train",5000)
 return ExperimentManifest("AUTH","c0.1","c","e"*64,(c("A"),c("B")),({"comparison_id":"A-B","candidate_a":"A","candidate_b":"B","budget_a":512,"budget_b":4096},),("o1","o2"),7,{"margin":-0.05},"cuda","fp32",{"python":"3"})
def result(g):return GameResult(**g.__dict__,terminal_result="DRAW",winner=None,draw=True,game_length=4,terminal_state_fingerprint="f"*64,game_trace_hash="d"*64,execution_metadata={})

def test_schedule_deterministic_balanced_and_asymmetric(tmp_path):
 m=manifest(tmp_path);a=schedule(m);b=schedule(m);assert a==b and len(a)==4 and len({x.game_id for x in a})==4
 for x,y in zip(a[::2],a[1::2]):assert (x.candidate_p1,x.budget_p1)==(y.candidate_p2,y.budget_p2) and x.pair_id==y.pair_id
def test_manifest_mismatch_resume_and_exactly_once(tmp_path):
 m=manifest(tmp_path);games=schedule(m);s=ArenaStore(tmp_path/"a.db");s.initialize(m,games);assert len(s.pending())==4
 r=result(games[0]);assert s.complete(r)=="COMMITTED" and s.complete(r)=="IDEMPOTENT" and len(s.pending())==3
 with pytest.raises(ValueError):s.complete(replace(r,game_length=5))
 with pytest.raises(ValueError):s.initialize(replace(m,base_seed=8),schedule(replace(m,base_seed=8)))
def test_unknown_invalid_and_crash_transaction(tmp_path):
 m=manifest(tmp_path);g=schedule(m)[0];s=ArenaStore(tmp_path/"a.db");s.initialize(m,[g])
 with pytest.raises(ValueError):s.complete(replace(result(g),game_id="missing"))
 s.db.execute("BEGIN");s.db.execute("INSERT INTO results VALUES(?,?)",(g.game_id,b"partial"));s.db.rollback();assert s.pending()==(g.game_id,)
def test_complete_pairs_export_and_gate(tmp_path):
 m=manifest(tmp_path);games=schedule(m);s=ArenaStore(tmp_path/"a.db");s.initialize(m,games)
 s.complete(result(games[0]));assert s.paired_records()=={}
 s.complete(result(games[1]));assert len(s.paired_records())==1
 info=export_store(s,m,tmp_path/"export.tar.gz");assert len(info["sha256"])==64 and info["pending"]==2
 with pytest.raises(FileExistsError):export_store(s,m,tmp_path/"export.tar.gz")
 with pytest.raises(PermissionError):require_execution(execute=False,authorized_ids={"AUTH"},manifest=m,checkpoints_valid=True)
 require_execution(execute=True,authorized_ids={"AUTH"},manifest=m,checkpoints_valid=True)
def test_checkpoint_provenance(tmp_path):
 p=tmp_path/"x";p.write_bytes(b"x");c=Candidate("A","x",file_sha256(p),"c","a","m","t",1);c.validate(tmp_path)
 with pytest.raises(ValueError):replace(c,checkpoint_sha256="0"*64).validate(tmp_path)
