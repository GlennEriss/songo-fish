import json
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from songo_ai.model import load_srn_checkpoint
from songo_ai.training.lot46 import Lot46Error, group_aware_split
from songo_ai.training.lot46a import (DatasetSource, DurableCheckpointStore,
    ExperimentConfig, WeightedStatefulSampler, canonical_hash,
    configure_trainable, make_optimizer_scheduler, resume_into, run_training,
    validate_bundle_manifest, validate_rows, validate_scientific_inputs)

ROOT=Path(__file__).resolve().parents[3]
CONTROL=ROOT/"configs/lot46a/control_smoke.json"


def config(tmp_path:Path,name="LOT46A_TEST_A",steps=2):
    base=ExperimentConfig.load(CONTROL)
    return replace(base,experiment_id=name,output_directory=str(tmp_path/name),max_steps=steps,
                   validation_interval=steps,checkpoint_interval=1,batch_size=8,device="cpu")


def test_all_candidate_contracts_are_explicitly_validated(tmp_path):
    base=config(tmp_path)
    assert base.candidate_family=="CONTROL"
    with pytest.raises(Lot46Error): replace(base,candidate_family="DECORATIVE").validate()
    with pytest.raises(Lot46Error): replace(base,candidate_family="CONTROL",dataset_sources=(DatasetSource("lot45","x",1,"deep",32768),)).validate()
    with pytest.raises(Lot46Error): replace(base,candidate_family="DEEP_POLICY_REPLAY",dataset_sources=(DatasetSource("lot45","x",1,"deep",32768),)).validate()


def test_weighted_sampler_resume_is_exact():
    rows=[{"target_source":"deep"}]*3+[{"target_source":"replay"}]*2
    a=WeightedStatefulSampler(rows,{"deep":.8,"replay":.2},7,4); a.next();state=a.state_dict();expected=a.next()
    b=WeightedStatefulSampler(rows,{"deep":.8,"replay":.2},7,4);b.load_state_dict(state)
    assert b.next()==expected and b.epoch==a.epoch and b.position==a.position


def test_policy_and_value_head_freezing_contracts(tmp_path):
    checkpoint=ROOT/"data/experiments/lot34r_g4_retry/checkpoints/pool/step-06000.pt"
    policy=load_srn_checkpoint(checkpoint).model; names=configure_trainable(policy,"DEEP_POLICY","HEAD_ONLY")
    assert names and all(n.startswith("policy_mlp.") for n in names)
    assert all(not p.requires_grad for n,p in policy.named_parameters() if n.startswith("value_mlp."))
    value=load_srn_checkpoint(checkpoint).model; names=configure_trainable(value,"VALUE_INDEPENDENT","HEAD_ONLY")
    assert names and all(n.startswith("value_mlp.") for n in names)
    shared=load_srn_checkpoint(checkpoint).model; names=configure_trainable(shared,"DEEP_POLICY","SHARED_TRUNK_TRAINABLE")
    assert any(n.startswith("node_encoder.") for n in names) and not any(n.startswith("value_mlp.") for n in names)


def test_optimizer_and_scheduler_are_historical_and_deterministic(tmp_path):
    cfg=config(tmp_path);model=load_srn_checkpoint(ROOT/cfg.policy_checkpoint).model;configure_trainable(model,cfg.candidate_family,cfg.training_mode)
    opt,scheduler=make_optimizer_scheduler(model,cfg)
    assert isinstance(opt,torch.optim.AdamW) and scheduler.get_last_lr()==[cfg.learning_rate]
    opt.step();scheduler.step();assert scheduler.get_last_lr()==[cfg.learning_rate]


def test_value_contract_rejects_non_terminal_and_missing_fake_zero():
    row={"visit_counts":[1,0,0,0,0,0,0],"legal_mask":[1]*7,"policy_target":[1.,0,0,0,0,0,0],
         "value_target_available":True,"z_mean":.5,"target_budget":1,"target_source":"x"}
    with pytest.raises(Lot46Error):validate_rows([row])
    row["value_target_available"]=False;row["z_mean"]=0
    with pytest.raises(Lot46Error):validate_rows([row])


def test_value_contract_accepts_only_proven_terminal_aggregate():
    row={"visit_counts":[1,0,0,0,0,0,0],"legal_mask":[1]*7,"policy_target":[1.,0,0,0,0,0,0],
         "value_target_available":True,"z_mean":1/3,"z_counts":{"win":2,"draw":0,"loss":1,"unknown":0},
         "z_perspective":"player_to_move","target_budget":32768,"target_source":"LOT45_MCTS32768"}
    assert validate_rows([row])["value_labeled"]==1
    row["z_mean"]=.5
    with pytest.raises(Lot46Error,match="inconsistent"):validate_rows([row])


def test_global_split_prevents_cross_corpus_state_leakage():
    rows=[{"fingerprint":"same","split_group":"game-a","holdout":True},{"fingerprint":"same","split_group":"game-a"},{"fingerprint":"other","split_group":"game-b"}]
    split=group_aware_split(rows);assert all(x["fingerprint"]=="same" for x in split["strategic_holdout"])
    assert not ({x["fingerprint"] for x in split["train"]}&{x["fingerprint"] for x in split["strategic_holdout"]})


def test_durable_store_ignores_corruption_and_incomplete_manifest(tmp_path):
    store=DurableCheckpointStore(tmp_path/"local",tmp_path/"drive")
    payload={"experiment_id":"X","value":torch.tensor([1])};m=store.publish(payload,1)
    assert store.latest("X")[1]["step"]==1
    Path(m["durable_path"]).write_bytes(b"corrupt")
    (tmp_path/"drive/checkpoint-00000002.pt").write_bytes(b"partial")
    assert store.latest("X") is None


def test_real_checkpoint_dataset_backward_checkpoint_and_resume(tmp_path):
    cfg=config(tmp_path,"LOT46A_INTEGRATION",4)
    first=run_training(cfg,ROOT,stop_after=2);assert first["status"]=="RESUMABLE" and first["checkpoint_durable"]
    final=run_training(cfg,ROOT,resume=True);assert final["status"]=="COMPLETED" and final["global_step"]==4
    assert final["head_invariance_pass"] is True


def test_multi_experiment_has_no_artifact_collision(tmp_path):
    a=config(tmp_path,"LOT46A_MULTI_A",1);b=replace(a,experiment_id="LOT46A_MULTI_B",output_directory=str(tmp_path/"LOT46A_MULTI_B"),seed=a.seed+1)
    ra=run_training(a,ROOT);rb=run_training(b,ROOT)
    assert ra["experiment_id"]!=rb["experiment_id"]
    assert (Path(a.output_directory)/"training_history.jsonl").is_file()
    assert (Path(b.output_directory)/"training_history.jsonl").is_file()
    assert Path(a.output_directory)!=Path(b.output_directory)


def test_resume_refuses_a_different_code_commit(tmp_path,monkeypatch):
    cfg=config(tmp_path,"LOT46A_COMMIT_PIN",2)
    run_training(cfg,ROOT,stop_after=1)
    monkeypatch.setattr("songo_ai.training.lot46a.git_commit",lambda root:"different-commit")
    with pytest.raises(Lot46Error,match="REFUSE_RESUME code_commit mismatch"):
        run_training(cfg,ROOT,resume=True)


def test_real_strategic_battery_is_present_valid_and_separate(tmp_path):
    report=validate_scientific_inputs(config(tmp_path),ROOT,stage="integration-test")
    battery=next(x for x in report["dependencies"] if x["name"]=="strategic_preservation_battery")
    assert battery["validation_status"]=="PASS" and battery["positions"]==2000
    assert battery["role"]=="TRAINING_PRESERVATION_PAIRWISE_ONLY"
    assert report["TRAINING_HOLDOUT_SEPARATION_VALID"]=="YES"


def test_missing_battery_has_complete_diagnostic(tmp_path):
    cfg=replace(config(tmp_path),objective={**config(tmp_path).objective,"strategic_battery":str(tmp_path/"missing.jsonl")})
    with pytest.raises(Lot46Error,match="MISSING_INPUTS=.*recovery_action"):
        validate_scientific_inputs(cfg,ROOT,stage="train")


def test_wrong_battery_sha_is_rejected(tmp_path):
    bad=tmp_path/"qdiag256.jsonl";bad.write_text("{}\n")
    cfg=replace(config(tmp_path),objective={**config(tmp_path).objective,"strategic_battery":str(bad)})
    with pytest.raises(Lot46Error,match="MISSING_INPUTS"):
        validate_scientific_inputs(cfg,ROOT,stage="train")


def test_incompatible_battery_schema_is_rejected(tmp_path,monkeypatch):
    bad=tmp_path/"qdiag256.jsonl";bad.write_text("{}\n")
    monkeypatch.setattr("songo_ai.training.lot46a.QDIAG256_SHA256",hashlib.sha256(bad.read_bytes()).hexdigest())
    cfg=replace(config(tmp_path),objective={**config(tmp_path).objective,"strategic_battery":str(bad)})
    with pytest.raises(Lot46Error,match="missing"):
        validate_scientific_inputs(cfg,ROOT,stage="train")


def test_wrong_battery_path_is_not_searched_implicitly(tmp_path):
    cfg=replace(config(tmp_path),objective={**config(tmp_path).objective,"strategic_battery":"wrong/qdiag256.jsonl"})
    with pytest.raises(Lot46Error) as error:validate_scientific_inputs(cfg,ROOT,stage="train")
    assert str(ROOT/"wrong/qdiag256.jsonl") in str(error.value)


def test_incomplete_bundle_is_rejected():
    with pytest.raises(Lot46Error,match="incomplete scientific bundle"):
        validate_bundle_manifest({"files":{"a":"x","b":"y"}},{"a"})


def test_prepare_stage_does_not_require_future_training_battery(tmp_path):
    cfg=replace(config(tmp_path),objective={**config(tmp_path).objective,"strategic_battery":"missing.jsonl"})
    report=validate_scientific_inputs(cfg,ROOT,stage="prepare-a")
    assert report["SCIENTIFIC_INPUTS_READY"]=="YES"
    assert all(x["name"]!="strategic_preservation_battery" for x in report["dependencies"])


def test_cuda_main_training_is_valid_smoke_evidence():
    from run_srn_lot46 import cuda_training_evidence
    status={"status":"COMPLETED","experiment_id":"X","device":"cuda","checkpoint_durable":True,
            "global_step":40,"code_commit":"abc","validation":{"no_grad_parameter_invariance":True}}
    latest=(Path("checkpoint.pt"),{"status":"COMMITTED","step":40})
    passed,evidence=cuda_training_evidence(status,latest,None)
    assert passed and evidence["source"]=="main_training_status" and evidence["global_step"]==40
    status["device"]="cpu"
    assert cuda_training_evidence(status,latest,None)[0] is False


def test_final_checkpoint_audit_loads_finite_and_is_idempotent(tmp_path):
    from run_srn_lot46 import audit_final_checkpoint
    cfg=config(tmp_path,"LOT46A_FINAL_AUDIT",1);run_training(cfg,ROOT)
    out=Path(cfg.output_directory);status=json.loads((out/"status.json").read_text());assignment=json.loads((out/"experiment_assignment.json").read_text())
    latest=DurableCheckpointStore(out/"local_checkpoints",out/"durable_checkpoints").latest(cfg.experiment_id)
    first=audit_final_checkpoint(cfg,status,assignment,latest);second=audit_final_checkpoint(cfg,status,assignment,latest)
    assert first["status"]==second["status"]=="PASS"
    assert first["sha256_before"]==first["sha256_after"]==second["sha256_after"]
    assert all(first["checks"].values())


def test_final_checkpoint_audit_rejects_missing_and_wrong_identity(tmp_path):
    from run_srn_lot46 import audit_final_checkpoint
    cfg=config(tmp_path,"LOT46A_FINAL_BAD",1)
    assert audit_final_checkpoint(cfg,None,None,None)["status"]=="FAIL"
    run_training(cfg,ROOT);out=Path(cfg.output_directory);status=json.loads((out/"status.json").read_text());assignment=json.loads((out/"experiment_assignment.json").read_text())
    latest=DurableCheckpointStore(out/"local_checkpoints",out/"durable_checkpoints").latest(cfg.experiment_id)
    status["experiment_id"]="OTHER"
    audit=audit_final_checkpoint(cfg,status,assignment,latest)
    assert audit["status"]=="FAIL" and audit["checks"]["experiment_id"] is False


def test_final_checkpoint_audit_rejects_corrupt_checkpoint(tmp_path):
    from run_srn_lot46 import audit_final_checkpoint
    cfg=config(tmp_path,"LOT46A_FINAL_CORRUPT",1);run_training(cfg,ROOT)
    out=Path(cfg.output_directory);status=json.loads((out/"status.json").read_text());assignment=json.loads((out/"experiment_assignment.json").read_text())
    latest=DurableCheckpointStore(out/"local_checkpoints",out/"durable_checkpoints").latest(cfg.experiment_id)
    path,manifest=latest;path.write_bytes(b"corrupt")
    audit=audit_final_checkpoint(cfg,status,assignment,(path,manifest))
    assert audit["status"]=="FAIL" and audit["error_type"]
