import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from songo_ai.evaluation.arena_runner import ArenaStore, Candidate, GameResult, file_sha256, schedule
from songo_ai.evaluation.arena_integration import SQLiteSnapshotManager
from songo_ai.evaluation.stage_c0_4 import *


class FakeCuda:
    def __init__(self, available=True): self.available=available;self.empty_calls=0
    def is_available(self): return self.available
    def get_device_name(self, index): return "Synthetic T4"
    def empty_cache(self): self.empty_calls+=1


def fake_torch(available=True):
    return SimpleNamespace(__version__="test",version=SimpleNamespace(cuda="12.1"),cuda=FakeCuda(available),empty=lambda *a,**k:None)


def opening(tmp_path):
    payload={"records":[{"opening_id":OPENING_ID,"state":{"board":[0]*16,"player_to_move":1},"state_fingerprint":"a"*64}]}
    path=tmp_path/"registry.json";path.write_text(json.dumps(payload,separators=(",",":")))
    return path,payload["records"][0]


def candidates(tmp_path):
    p=tmp_path/"a.pt";p.write_bytes(b"model")
    def c(name):return Candidate(name,str(p),file_sha256(p),"c","SRN_COMPLETE_NETWORK","m","train",5000)
    return c("POOL_G4R"),c("DEEP_POLICY_UNCONSTRAINED")


def test_device_auto_cuda_unavailable_and_forced_cpu_never_touches_cuda():
    yes=fake_torch();assert select_device(yes,"auto").selected_device=="cuda"
    no=fake_torch(False);p=select_device(no,"auto");assert p.selected_device=="cpu" and p.fallback_reason
    class ExplodingCuda:
        def is_available(self):raise AssertionError("CUDA touched")
    cpu=SimpleNamespace(__version__="x",version=SimpleNamespace(cuda=None),cuda=ExplodingCuda())
    assert select_device(cpu,"cpu").selected_device=="cpu"


def test_forced_cuda_and_initialization_failure():
    with pytest.raises(DeviceSelectionError):select_device(fake_torch(False),"cuda")
    p=select_device(fake_torch(),"auto",cuda_probe=lambda:(_ for _ in ()).throw(RuntimeError("driver")))
    assert p.selected_device=="cpu" and "driver" in p.fallback_reason
    with pytest.raises(DeviceSelectionError):select_device(fake_torch(),"cuda",cuda_probe=lambda:(_ for _ in ()).throw(RuntimeError("driver")))


def test_registry_hash_opening_schedule_side_reversal_and_two_games(tmp_path):
    path,item=opening(tmp_path);digest=file_sha256(path);assert load_frozen_opening(path,digest)==item
    with pytest.raises(ValueError):load_frozen_opening(path,"0"*64)
    manifest=proposed_manifest("c",candidates(tmp_path),item);games=schedule(manifest)
    assert len(games)==2 and {g.opening_id for g in games}=={OPENING_ID}
    assert (games[0].candidate_p1,games[0].candidate_p2)==(games[1].candidate_p2,games[1].candidate_p1)
    assert all(g.budget_p1==g.budget_p2==512 for g in games)


def test_dry_run_and_real_gate_always_fail_closed():
    with pytest.raises(PermissionError):require_real_execution(dry_run=True,authorization=None)
    with pytest.raises(PermissionError):require_real_execution(dry_run=False,authorization="LOT46B_STAGE_C0_4_REAL_SMOKE_APPROVED")


def test_attempt_failure_cpu_restart_and_exactly_once(tmp_path):
    _,item=opening(tmp_path);manifest=proposed_manifest("c",candidates(tmp_path),item);games=schedule(manifest)
    store=ArenaStore(tmp_path/"arena.db");store.initialize(manifest,games);attempts=AttemptRecorder(store)
    attempts.begin(games[0],select_device(fake_torch(),"auto"),"try-gpu")
    cpu=fallback_before_commit(RuntimeError("CUDA out of memory"),result_committed=False,torch_module=fake_torch(),attempt_recorder=attempts,game=games[0],attempt_id="try-gpu")
    assert cpu.selected_device=="cpu" and len(store.pending())==2
    attempts.begin(games[0],cpu,"try-cpu")
    r=GameResult(**games[0].__dict__,terminal_result="DRAW",winner=None,draw=True,game_length=1,terminal_state_fingerprint="f"*64,game_trace_hash="d"*64,execution_metadata={"attempt_id":"try-cpu","device":"cpu"})
    assert store.complete(r)=="COMMITTED" and store.complete(r)=="IDEMPOTENT" and len(store.pending())==1
    with pytest.raises(DeviceSelectionError):fallback_before_commit(RuntimeError("CUDA failure"),result_committed=True,torch_module=fake_torch(),attempt_recorder=attempts,game=games[0],attempt_id="try-cpu")


def test_snapshot_restore_integrity_and_manifest_resume(tmp_path):
    _,item=opening(tmp_path);manifest=proposed_manifest("c",candidates(tmp_path),item);store=ArenaStore(tmp_path/"arena.db");store.initialize(manifest,schedule(manifest))
    mgr=SQLiteSnapshotManager(tmp_path/"drive",manifest.fingerprint);mgr.publish(store,0);restored=tmp_path/"restored.db";mgr.restore(0,restored)
    resumed=ArenaStore(restored);resumed.initialize(manifest,schedule(manifest));assert len(resumed.pending())==2
    with pytest.raises(ValueError):resumed.initialize(replace(manifest,base_seed=2),schedule(replace(manifest,base_seed=2)))


def test_full_network_cpu_finite_policy_value_inference_and_no_grad():
    class FullNetwork(torch.nn.Module):
        def __init__(self):super().__init__();self.encoder=torch.nn.Linear(4,8);self.policy=torch.nn.Linear(8,3);self.value=torch.nn.Linear(8,1)
        def forward(self,x):h=self.encoder(x);return self.policy(h),torch.tanh(self.value(h))
    model=FullNetwork().cpu().eval().requires_grad_(False)
    with torch.inference_mode():policy,value=model(torch.ones(2,4));masked=policy.masked_fill(~torch.tensor([[1,1,0],[1,0,1]],dtype=torch.bool),-torch.inf)
    assert torch.isfinite(policy).all() and torch.isfinite(value).all() and not torch.isnan(masked).any()
    assert all(not p.requires_grad for p in model.parameters())


def test_checkpoint_validation_and_performance_fields(tmp_path):
    p=tmp_path/"x";p.write_bytes(b"x");assert validate_checkpoint(p,file_sha256(p))["status"]=="PASS"
    assert validate_checkpoint(tmp_path/"missing","a"*64)["status"]=="NOT_VERIFIED"
    with pytest.raises(ValueError):validate_checkpoint(p,"0"*64)
    assert set(performance_schema())=={"MODEL_LOAD_TIME_SECONDS","DEVICE_INITIALIZATION_SECONDS","INFERENCE_TIME_SECONDS","MCTS_SEARCH_TIME_SECONDS","MCTS_SIMULATIONS_COMPLETED","MCTS_SIMULATIONS_PER_SECOND","AVERAGE_MOVE_TIME_SECONDS","GAME_DURATION_SECONDS","SQLITE_COMMIT_TIME_SECONDS","DRIVE_SNAPSHOT_TIME_SECONDS","PEAK_MEMORY_USAGE","DEVICE_TYPE"}
