from dataclasses import replace
from types import SimpleNamespace
import pytest
from songo_ai.evaluation.arena_runner import *
from songo_ai.evaluation.arena_integration import *

def setup(tmp_path):
 p=tmp_path/"a.pt";p.write_bytes(b"model");c=lambda n:Candidate(n,"a.pt",file_sha256(p),"c","srn","m","train",5)
 o=OpeningRecord("o",{"board":[1],"turn":1},fingerprint({"board":[1],"turn":1}),"independent","arena",0)
 m=ExperimentManifest("X","c0.2","c","e"*64,(c("A"),c("B")),({"comparison_id":"ab","candidate_a":"A","candidate_b":"B","budget_a":512,"budget_b":4096},),("o",),7,{"margin":-0.05},"cuda","fp32",{})
 s=ArenaStore(tmp_path/"db");calls=[]
 def execute(**kw):calls.append(kw);return {"terminal_result":"DRAW","winner":None,"draw":True,"game_length":4,"terminal_state_fingerprint":"f"*64,"game_trace_hash":"d"*64,"execution_metadata":{}}
 a=ArenaAdapter(m,{"o":o},s,model_loader=lambda c:c.name,game_executor=execute,engine_fingerprint="e"*64,mcts_settings=MCTSSettings())
 return p,m,s,a,calls
def test_dry_run_translation_never_executes(tmp_path):
 p,m,s,a,calls=setup(tmp_path);plan=a.validate(tmp_path);assert plan["planned_games"]==2 and not calls
 g=schedule(m);assert g[0].budget_p1==512 and g[1].budget_p2==512
def test_fail_closed_then_synthetic_execution_and_resume(tmp_path):
 p,m,s,a,calls=setup(tmp_path);g=schedule(m)[0]
 with pytest.raises(PermissionError):a.execute_one(g,None,checkpoint_root=tmp_path)
 auth=ExecutionAuthorization("X",m.fingerprint,True,"review-1");assert a.execute_one(g,auth,checkpoint_root=tmp_path)=="COMMITTED";assert len(calls)==1 and len(s.pending())==1
 assert a.execute_one(g,auth,checkpoint_root=tmp_path)=="IDEMPOTENT"
def test_opening_mcts_checkpoint_validation(tmp_path):
 p,m,s,a,calls=setup(tmp_path)
 with pytest.raises(ValueError):replace(a.openings["o"],state_fingerprint="0"*64).validate()
 with pytest.raises(ValueError):MCTSSettings(root_noise=True).validate()
 p.write_bytes(b"changed")
 with pytest.raises(ValueError):a.validate(tmp_path)
def test_cuda_reports_unavailable_and_available():
 no=SimpleNamespace(__version__="x",version=SimpleNamespace(cuda=None),cuda=SimpleNamespace(is_available=lambda:False));assert cuda_readiness(no)["status"]=="GPU_NOT_AVAILABLE"
 yes=SimpleNamespace(__version__="x",version=SimpleNamespace(cuda="13"),cuda=SimpleNamespace(is_available=lambda:True,get_device_name=lambda i:"T4",mem_get_info=lambda i:(10,20)));assert cuda_readiness(yes)["gpu_name"]=="T4"
def test_consistent_snapshot_restore_corruption_and_stale(tmp_path):
 p,m,s,a,calls=setup(tmp_path);a.validate(tmp_path);mgr=SQLiteSnapshotManager(tmp_path/"drive",m.fingerprint);info=mgr.publish(s,1);rest=tmp_path/"rest.db";mgr.restore(1,rest);assert ArenaStore(rest).pending()==s.pending()
 Path(info["path"]).write_bytes(b"bad")
 with pytest.raises(ValueError):mgr.restore(1,tmp_path/"other.db")
 with pytest.raises(FileExistsError):mgr.publish(s,1)
def test_metrics_are_synthetic_only():
 x=PerformanceMetrics(search_duration_s=2,simulations=10);assert x.simulations_per_second==5
