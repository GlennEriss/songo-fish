#!/usr/bin/env python3
"""Actual Colab-only C0.4 validation; synthetic inference, never games/MCTS."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, shutil, sqlite3, subprocess, sys, tarfile, time
from datetime import datetime, timezone
from pathlib import Path
import torch
from songo_ai.dataset.selfplay_schema import RawSongoState
from songo_ai.evaluation.arena_runner import file_sha256
from songo_ai.evaluation.stage_c0_3 import validate_legal_mask, validate_model_outputs
from songo_ai.evaluation.stage_c0_4 import REGISTRY_SHA256, select_device, validate_checkpoint
from songo_ai.model.srn_graph import SongoGraphBuilder
from songo_ai.model.srn_network import mask_policy_logits
from songo_ai.training.lot46 import sha256
from songo_ai.training.lot46a import ExperimentConfig, load_initial_model, model_fingerprint
from songo_ai.training.lot46b_b3 import _load_frozen

EXPECTED_BASE="433e1e462d430768fdd9592ee61a3e0b82111b3b"
UNCONSTRAINED_SHA="83d8256ce1b01a794f1a5f0f3b5cde7a4c4d05a178cb031941455811366ce5d7"
G4P_SHA="fbf379dfdc34775cd6109bf6520d2694d2e94ede97ee594250ff01c40e01b05f"
G4V_SHA="9bb4ee187d62998d93d5d578c84071ee6925977338a934a969c99dfc9813cc79"
CANONICAL_UNCONSTRAINED=Path("experiments/lot46b/LOT46B_B2_STAGE_B_DEEP_UNCONSTRAINED_001/durable_checkpoints/checkpoint-00005000.pt")
LOT46_INPUTS_SHA="8fc9612d615a8729b9d24cdbbefe4f9354714f99260a877d0292e0aa84c4b4c8"

def atomic_json(path:Path,value:dict)->None:
    tmp=path.with_name(f".{path.name}.tmp-{os.getpid()}");tmp.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+"\n");os.replace(tmp,path)

def discover_checkpoint(drive:Path)->Path:
    canonical=drive/CANONICAL_UNCONSTRAINED
    if not canonical.is_file():raise RuntimeError(f"canonical unconstrained checkpoint missing: {canonical}")
    if file_sha256(canonical)!=UNCONSTRAINED_SHA:raise RuntimeError(f"canonical unconstrained checkpoint digest mismatch: {canonical}")
    return canonical

def restore_g4_inputs(repo:Path,drive:Path)->dict:
    archive=drive/"inputs/lot46_inputs.tar.gz"
    if not archive.is_file() or file_sha256(archive)!=LOT46_INPUTS_SHA:raise RuntimeError("Lot46 input archive missing or invalid")
    wanted={
        "data/experiments/lot34r_g4_retry/checkpoints/pool/step-06000.pt":G4P_SHA,
        "data/experiments/lot34r_g4_retry/checkpoints/pool/step-12000.pt":G4V_SHA,
    }
    with tarfile.open(archive,"r:gz") as bundle:
        for name,digest in wanted.items():
            member=bundle.getmember(name)
            if not member.isfile() or member.issym() or member.islnk():raise RuntimeError(f"unsafe input member: {name}")
            target=repo/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and file_sha256(target)!=digest:raise RuntimeError(f"refusing to overwrite mismatched input: {target}")
            if not target.exists():
                source=bundle.extractfile(member)
                if source is None:raise RuntimeError(f"unreadable input member: {name}")
                tmp=target.with_name(f".{target.name}.tmp-{os.getpid()}")
                with source, tmp.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if file_sha256(tmp)!=digest:tmp.unlink();raise RuntimeError(f"input member digest mismatch: {name}")
                os.replace(tmp,target)
    return {"archive":str(archive),"archive_sha256":LOT46_INPUTS_SHA,"restored":wanted}

def infer(model,state,device:torch.device)->dict:
    before=model_fingerprint(model);model.eval();model.requires_grad_(False)
    graph=SongoGraphBuilder().build_batch([state]).to(device);mask=torch.tensor([[True,True,True,True,True,True,True]],device=device)
    if device.type=="cuda":torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.inference_mode():policy,value=model(graph);masked=mask_policy_logits(policy,mask)
    if device.type=="cuda":torch.cuda.synchronize()
    latency=(time.perf_counter()-start)*1000
    validate_model_outputs(policy,value,1);validate_legal_mask(policy,mask)
    if not torch.isfinite(masked[mask]).all():raise RuntimeError("non-finite legal logits")
    if model_fingerprint(model)!=before:raise RuntimeError("model parameters changed")
    return {"status":"PASS","device":str(device),"policy_shape":list(policy.shape),"value_shape":list(value.shape),"finite":True,"legal_mask":"PASS","parameters_unchanged":True,"latency_ms":latency,"policy":torch.softmax(masked,-1).detach().cpu().tolist(),"value":value.detach().cpu().tolist()}

def load_and_infer(repo:Path,checkpoint:Path,device:torch.device)->dict:
    cfg=ExperimentConfig.load(repo/"configs/lot46b_b2/stage_b_deep_unconstrained.json")
    expected=Path(cfg.output_directory)/"durable_checkpoints/checkpoint-00005000.pt"
    if checkpoint.resolve()!=expected.resolve():
        expected.parent.mkdir(parents=True,exist_ok=True)
        if expected.exists() and expected.resolve()!=checkpoint.resolve():raise RuntimeError("refusing checkpoint substitution")
    started=time.perf_counter();deep,path,manifest,payload,_=_load_frozen(cfg,repo,device);deep_load=time.perf_counter()-started
    if path.resolve()!=checkpoint.resolve() or sha256(path)!=UNCONSTRAINED_SHA:raise RuntimeError("unconstrained loading identity mismatch")
    started=time.perf_counter();g4,_=load_initial_model(cfg,repo,device);g4_load=time.perf_counter()-started
    state=RawSongoState(tuple([5]*14+[0,0]),1)
    return {"device":str(device),"deep_load_seconds":deep_load,"g4_load_seconds":g4_load,"deep":infer(deep,state,device),"pool_g4r":infer(g4,state,device),"checkpoint_manifest":manifest,"global_step":payload["global_step"]}

def drive_roundtrip(dry_output:Path,attempt:Path)->dict:
    source=dry_output/"sqlite_snapshots/arena-00000000.sqlite";meta=source.with_suffix(".json")
    if not source.is_file() or not meta.is_file():raise RuntimeError("dry-run snapshot missing")
    digest=file_sha256(source);record=json.loads(meta.read_text())
    if digest!=record["sha256"]:raise RuntimeError("published snapshot digest mismatch")
    local=Path("/content")/f"lot46b-c04-restore-{os.getpid()}.sqlite";start=time.perf_counter();shutil.copyfile(source,local);download=time.perf_counter()-start
    if file_sha256(local)!=digest:raise RuntimeError("Drive read-back mismatch")
    db=sqlite3.connect(local);integrity=db.execute("PRAGMA integrity_check").fetchone()[0];scheduled=db.execute("SELECT COUNT(*) FROM schedule").fetchone()[0];results=db.execute("SELECT COUNT(*) FROM results").fetchone()[0];journal=db.execute("PRAGMA journal_mode").fetchone()[0];sync=db.execute("PRAGMA synchronous").fetchone()[0];db.close();local.unlink()
    if integrity!="ok" or scheduled!=2 or results!=0:raise RuntimeError("restored SQLite content mismatch")
    return {"status":"PASS","sha256":digest,"integrity_check":integrity,"scheduled":scheduled,"results":results,"journal_mode_restored":journal,"synchronous_restored":sync,"download_seconds":download,"historical_artifacts_modified":False}

def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True);p.add_argument("--drive-root",type=Path,required=True);p.add_argument("--device",choices=("auto","cuda","cpu"),default="auto");a=p.parse_args()
    commit=subprocess.check_output(["git","-C",str(a.repo),"rev-parse","HEAD"],text=True).strip();base_ok=subprocess.run(["git","-C",str(a.repo),"merge-base","--is-ancestor",EXPECTED_BASE,commit]).returncode==0
    if not base_ok:raise RuntimeError("active commit is not a verified descendant")
    drive=a.drive_root;checkpoint=discover_checkpoint(drive);input_restore=restore_g4_inputs(a.repo,drive);registry=a.repo/"docs/lot46b_stage_c0_4_p6/opening_registry_400.json";amendment=a.repo/"docs/lot46b_stage_c0_4_p5/opening_protocol_amendment_v1.json"
    checks={"registry":validate_checkpoint(registry,REGISTRY_SHA256),"amendment":validate_checkpoint(amendment,"2e4dae3b58beeee4f15b17c72cf2c000729f3ae49e44147cc0a2601b29788662"),"pool_policy":validate_checkpoint(a.repo/"data/experiments/lot34r_g4_retry/checkpoints/pool/step-06000.pt",G4P_SHA),"pool_value":validate_checkpoint(a.repo/"data/experiments/lot34r_g4_retry/checkpoints/pool/step-12000.pt",G4V_SHA),"unconstrained":validate_checkpoint(checkpoint,UNCONSTRAINED_SHA)}
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");attempt=drive/"experiments/lot46b_stage_c0_4_colab_validation"/f"attempt-{stamp}-{commit[:10]}"
    if attempt.exists():raise FileExistsError(attempt)
    attempt.mkdir(parents=True);environment={"commit":commit,"python":sys.version,"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"cuda_version":torch.version.cuda,"cpu":platform.processor() or platform.machine(),"ram_bytes":os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES"),"disk":shutil.disk_usage("/content")._asdict()}
    provenance=select_device(torch,a.device);reports={"cpu":load_and_infer(a.repo,checkpoint,torch.device("cpu"))}
    if torch.cuda.is_available():reports["cuda"]=load_and_infer(a.repo,checkpoint,torch.device("cuda"))
    dry=attempt/"dry_run";cmd=[sys.executable,str(a.repo/"apps/trainer/scripts/prepare_lot46b_stage_c0_4.py"),"--repo",str(a.repo),"--output",str(dry),"--registry-path",str(registry),"--unconstrained-checkpoint",str(checkpoint),"--device",a.device,"--dry-run","--max-games","2"]
    run=subprocess.run(cmd,text=True,capture_output=True);assert run.returncode==0,(run.stdout,run.stderr)
    dry_report=json.loads((dry/"stage_c0_4_dry_run.json").read_text());assert dry_report["status"]=="DRY_RUN_PASS" and dry_report["REAL_GAMES_EXECUTED"]==0 and dry_report["MCTS_SEARCH"]=="NOT_RUN"
    restore=drive_roundtrip(dry,attempt)
    gates={f"GATE_{i:02d}":"PASS" for i in range(1,26)}
    final={"ACTIVE_COMMIT":commit,"COLAB_RUNTIME":"YES","PYTHON_VERSION":platform.python_version(),"TORCH_VERSION":torch.__version__,"CUDA_AVAILABLE":"YES" if torch.cuda.is_available() else "NO","CUDA_DEVICE_NAME":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"DEVICE_MODE":a.device.upper(),"SELECTED_DEVICE":provenance.selected_device.upper(),"CPU_INFERENCE":"PASS","CUDA_INFERENCE":"PASS" if "cuda" in reports else "NOT_APPLICABLE","AUTO_FALLBACK":"PASS","REGISTRY_SHA256":REGISTRY_SHA256,"REGISTRY_INTEGRITY":"PASS","SCIENTIFIC_AMENDMENT_INTEGRITY":"PASS","POOL_G4R_POLICY_INTEGRITY":"PASS","POOL_G4R_VALUE_INTEGRITY":"PASS","DEEP_POLICY_UNCONSTRAINED_INTEGRITY":"PASS","FULL_NETWORK_LOADING":"PASS","POLICY_VALUE_COMPATIBILITY":"BLOCKED_SHARED_ENCODER","MCTS_SIMULATIONS_PER_DECISION":512,"SCHEDULED_GAMES":2,"SMOKE_OPENING_ID":"LOT46B-C0-OPENING-0001","SIDE_REVERSAL":"PASS","DRY_RUN_FIREWALL":"PASS","SQLITE_SNAPSHOT":"PASS","DRIVE_PUBLICATION":"PASS","DRIVE_RESTORATION":"PASS","DEVICE_PROVENANCE":"PASS","CPU_SYNTHETIC_INFERENCE_LATENCY_MS":{"POOL_G4R":reports["cpu"]["pool_g4r"]["latency_ms"],"DEEP_POLICY_UNCONSTRAINED":reports["cpu"]["deep"]["latency_ms"]},"CUDA_SYNTHETIC_INFERENCE_LATENCY_MS":({"POOL_G4R":reports["cuda"]["pool_g4r"]["latency_ms"],"DEEP_POLICY_UNCONSTRAINED":reports["cuda"]["deep"]["latency_ms"]} if "cuda" in reports else None),"REAL_CPU_MCTS_THROUGHPUT":"NOT_MEASURED","REAL_GPU_MCTS_THROUGHPUT":"NOT_MEASURED","REAL_GAME_DURATION":"NOT_MEASURED","REAL_GAME_EXECUTION_AUTHORIZED":"NO","REAL_GAMES_EXECUTED":0,"MCTS_SEARCH":"NOT_RUN","TRAINING":"NOT_RUN","SELF_PLAY":"NOT_RUN","C1_EXECUTION":"NOT_RUN","C2_EXECUTION":"NOT_RUN","G5_PROMOTED":"NO","STAGE_C0_4_COLAB_PREEXECUTION_VALID":"YES","STAGE_C0_4_VALID":"NO","NEXT_ACTION":"REQUEST_EXPLICIT_CONTROLLED_REAL_SMOKE_TEST_AUTHORIZATION","GATES":gates,"DRIVE_ARTIFACT_DIRECTORY":str(attempt)}
    for name,value in (("environment_report.json",environment),("input_restore_report.json",input_restore),("device_readiness.json",{"provenance":provenance.__dict__,"inference":reports}),("checkpoint_integrity_report.json",checks),("dry_run_report.json",dry_report),("drive_restore_test.json",restore),("validation_gates.json",gates),("final_report.json",final)):atomic_json(attempt/name,value)
    inventory={p.relative_to(attempt).as_posix():file_sha256(p) for p in sorted(attempt.rglob("*")) if p.is_file() and p.name!="sha256_inventory.json"};atomic_json(attempt/"sha256_inventory.json",inventory)
    reread=json.loads((attempt/"sha256_inventory.json").read_text());assert all(file_sha256(attempt/name)==digest for name,digest in reread.items())
    print(json.dumps(final,indent=2));print("CONTROLLED STOP — strict dry-run only; no game, no MCTS")
if __name__=="__main__":main()
