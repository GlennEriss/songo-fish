import json, subprocess, sys
from dataclasses import asdict
from pathlib import Path
import pytest

from songo_ai.evaluation.arena_runner import ArenaStore, Candidate, GameResult, schedule
from songo_ai.evaluation.stage_c0_4 import proposed_manifest
from songo_ai.evaluation.stage_c0_4_real import *

def candidate(tmp_path,name):
    p=tmp_path/f"{name}.pt";p.write_bytes(name.encode())
    from songo_ai.evaluation.arena_runner import file_sha256
    return Candidate(name,str(p),file_sha256(p),"c","srn","cfg","training",1)

def fixture(tmp_path):
    opening={"opening_id":OPENING_ID}
    m=proposed_manifest("602abbdb09af63f222f273b36b4f6ce8ec2261dd",(candidate(tmp_path,MODEL_A),candidate(tmp_path,MODEL_B)),opening)
    ids={MODEL_A:m.candidates[0].checkpoint_sha256,MODEL_B:m.candidates[1].checkpoint_sha256}
    return m,ids

def auth(tmp_path,m,ids,mutate=None):
    value=authorization_binding(m,ids)
    if mutate:mutate(value)
    p=tmp_path/"authorization.json";p.write_text(json.dumps(value));return p

def result(g):
    return GameResult(**g.__dict__,terminal_result="DRAW",winner=None,draw=True,game_length=1,terminal_state_fingerprint="f"*64,game_trace_hash="e"*64,execution_metadata={"synthetic":True})

def test_schedule_is_exact_pair(tmp_path):
    m,_=fixture(tmp_path);c=schedule_contract(m);assert len(c["game_ids"])==2;assert c["side_assignment_schedule"]==[[MODEL_A,MODEL_B],[MODEL_B,MODEL_A]]

@pytest.mark.parametrize("field,value",[("experiment_id","wrong"),("registry_sha256","0"*64),("opening_id","wrong"),("mcts_simulations_per_decision",511),("total_scheduled_games",3),("authorization_scope","generic")])
def test_authorization_mismatch_fails(tmp_path,field,value):
    m,ids=fixture(tmp_path);p=auth(tmp_path,m,ids,lambda x:x.__setitem__(field,value))
    with pytest.raises(PermissionError):validate_authorization(p,m,ids)

def test_missing_authorization_fails(tmp_path):
    m,ids=fixture(tmp_path)
    with pytest.raises(PermissionError):validate_authorization(tmp_path/"missing",m,ids)

def test_exact_authorization_passes(tmp_path):
    m,ids=fixture(tmp_path);assert validate_authorization(auth(tmp_path,m,ids),m,ids)["approved"]

def test_production_rejects_synthetic_fixture(tmp_path):
    m,ids=fixture(tmp_path);p=auth(tmp_path,m,ids,lambda x:x.__setitem__("fixture_only",True))
    with pytest.raises(PermissionError,match="fixture"):validate_authorization(p,m,ids)

def test_fail_closed_ownership():
    with pytest.raises(PermissionError):validate_ownership(OwnershipConfirmation(EXPERIMENT_ID,"",False))
    validate_ownership(OwnershipConfirmation(EXPERIMENT_ID,"operator-session",True))

def test_state_machine_rejects_unsafe_start():
    with pytest.raises(ValueError):transition(RunState.CREATED,RunState.RUNNING)
    assert transition(RunState.AUTHORIZED,RunState.RUNNING)==RunState.RUNNING

def test_exactly_once_and_resume(tmp_path):
    m,ids=fixture(tmp_path);store=ArenaStore(tmp_path/"arena.sqlite");store.initialize(m,schedule(m));games=schedule(m)
    assert store.complete(result(games[0]))=="COMMITTED";assert store.complete(result(games[0]))=="IDEMPOTENT"
    assert verify_resume(store,m)==(games[1],)

def test_conflicting_duplicate_rejected(tmp_path):
    m,_=fixture(tmp_path);store=ArenaStore(tmp_path/"arena.sqlite");store.initialize(m,schedule(m));g=schedule(m)[0];store.complete(result(g))
    bad=GameResult(**g.__dict__,terminal_result="P1_WIN",winner=g.candidate_p1,draw=False,game_length=2,terminal_state_fingerprint="a"*64,game_trace_hash="b"*64,execution_metadata={})
    with pytest.raises(ValueError):store.complete(bad)

def test_interrupted_attempt_does_not_count(tmp_path):
    m,_=fixture(tmp_path);store=ArenaStore(tmp_path/"a.sqlite");store.initialize(m,schedule(m));j=ExecutionJournal(store);aid=j.begin(schedule(m)[0],"cuda");j.interrupt(aid,"oom")
    assert len(store.pending())==2 and j.history()[0]["status"]=="INTERRUPTED"
    assert j.begin(schedule(m)[0],"cpu")!=aid

def test_synthetic_execution_pair_and_snapshots(tmp_path):
    m,ids=fixture(tmp_path);store=ArenaStore(tmp_path/"a.sqlite");store.initialize(m,schedule(m))
    out=execute_pending(manifest=m,store=store,authorization_file=auth(tmp_path,m,ids),checkpoint_identities=ids,ownership=OwnershipConfirmation(EXPERIMENT_ID,"one",True),device="cpu",executor=lambda g,d:result(g),snapshot_directory=tmp_path/"snap")
    assert out["completed"]==2 and out["pending"]==0 and len(out["attempt_history"])==2

def test_production_control_flow_synthetic_end_to_end(tmp_path):
    m,ids=fixture(tmp_path);store=ArenaStore(tmp_path/"prod.sqlite");store.initialize(m,schedule(m));seen=[]
    def factory(device):
        def execute(g,d):seen.append((g.candidate_p1,g.candidate_p2,d));return result(g)
        return execute
    evidence={"status":"PASS","synthetic":True}
    out=production_control_flow(preflight=lambda:evidence,manifest=m,store=store,authorization_file=auth(tmp_path,m,ids),checkpoint_identities=ids,ownership=OwnershipConfirmation(EXPERIMENT_ID,"synthetic-test",True),device_selector=lambda:"cpu",executor_factory=factory,snapshot_directory=tmp_path/"snap-prod")
    assert out["preflight"]==evidence and out["completed"]==2 and out["pending"]==0
    assert seen==[(MODEL_A,MODEL_B,"cpu"),(MODEL_B,MODEL_A,"cpu")]
    assert verify_resume(store,m)==()

def test_snapshot_corruption_rejected(tmp_path):
    m,_=fixture(tmp_path);store=ArenaStore(tmp_path/"a.sqlite");store.initialize(m,schedule(m));mgr=SQLiteSnapshotManager(tmp_path/"snap",m.fingerprint);info=mgr.publish(store,0);Path(info["path"]).write_bytes(b"broken")
    with pytest.raises(ValueError):mgr.restore(0,tmp_path/"restored")

def test_lazy_adapter_compatibility_only():
    assert RealAdapters.compatibility()=={"engine":"SongoLegacyGame","state":"RawSongoState","mcts":"SongoMCTS"}

def test_frozen_state_adapter_boundary_without_game_progression(monkeypatch):
    import songo_ai.evaluation.srn_arena as arena
    from dataclasses import dataclass
    seen={}
    @dataclass
    class SyntheticResult:
        status:str
    def fake(*args,**kwargs):
        seen.update(kwargs)
        return SyntheticResult("SYNTHETIC")
    monkeypatch.setattr(arena,"play_arena_game",fake)
    assert RealAdapters.execute_game()["status"]=="SYNTHETIC" and seen["validate_prefix"] is False

def test_runner_default_is_safe(tmp_path):
    script=Path("apps/trainer/scripts/run_lot46b_stage_c0_4_real.py")
    r=subprocess.run([sys.executable,str(script),"--registry-path",str(tmp_path/"r"),"--checkpoint-root",str(tmp_path),"--output-dir",str(tmp_path/"o"),"--drive-output",str(tmp_path/"d")],text=True,capture_output=True)
    assert r.returncode==0 and '"REAL_GAMES_EXECUTED": 0' in r.stdout and '"MCTS_SEARCH": "NOT_RUN"' in r.stdout

def test_runner_execute_fails_closed_without_authorization(tmp_path):
    script=Path("apps/trainer/scripts/run_lot46b_stage_c0_4_real.py")
    r=subprocess.run([sys.executable,str(script),"--registry-path",str(tmp_path/"r"),"--checkpoint-root",str(tmp_path),"--output-dir",str(tmp_path/"o"),"--drive-output",str(tmp_path/"d"),"--execute"],text=True,capture_output=True)
    assert r.returncode!=0 and "authorization-file" in (r.stdout+r.stderr)

def test_old_dry_run_firewall_unchanged():
    source=Path("apps/trainer/scripts/prepare_lot46b_stage_c0_4.py").read_text();assert 'action="store_true",required=True' in source
    assert "songo_ai.search" not in source and "songo_ai.songo" not in source

def test_real_runner_safe_default_and_lazy_adapters():
    source=Path("apps/trainer/scripts/run_lot46b_stage_c0_4_real.py").read_text()
    assert '"status":"SAFE_PREFLIGHT_ONLY"' in source
    assert "if not a.execute:" in source and "--single-runner-confirmed" in source
    before_factory=source.split("def real_executor_factory",1)[0]
    assert "songo_ai.songo" not in before_factory and "songo_ai.search" not in before_factory
