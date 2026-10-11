import inspect
from pathlib import Path

import pytest

from songo_ai.training.lot46 import Lot46Error
from songo_ai.training.lot46a import ExperimentConfig, evaluate, step_model
from songo_ai.training.lot46b_b2 import ALLOWED, TARGETS, _comparable_payload, _decision

ROOT=Path(__file__).resolve().parents[3]


def configs():
    stage=ExperimentConfig.load(ROOT/"configs/lot46b_b2/stage_a_deep_policy.json")
    standard=ExperimentConfig.load(ROOT/"configs/lot46b_b2/stage_b_deep_policy_standard.json")
    unconstrained=ExperimentConfig.load(ROOT/"configs/lot46b_b2/stage_b_deep_unconstrained.json")
    return stage,standard,unconstrained


def test_only_two_policy_families_are_authorized():
    assert ALLOWED=={"DEEP_POLICY","DEEP_POLICY_UNCONSTRAINED"}
    assert "VALUE_INDEPENDENT" not in ALLOWED and "DEEP_POLICY_REPLAY" not in ALLOWED


def test_exact_progressive_targets_are_frozen():
    assert TARGETS==(1000,2000,3000,5000)


def test_unconstrained_starts_new_and_targets_5000():
    _,_,config=configs();assert config.max_steps==5000
    assert config.experiment_id=="LOT46B_B2_STAGE_B_DEEP_UNCONSTRAINED_001"
    assert config.objective["name"]=="DEEP_POLICY_CE_ONLY"


def test_standard_keeps_historical_scientific_settings():
    stage,standard,_=configs()
    payload={"training_config_fingerprint":__import__("songo_ai.training.lot46a",fromlist=["canonical_hash"]).canonical_hash(stage.payload())}
    assert _comparable_payload(payload,stage,standard)
    assert standard.objective==stage.objective and standard.seed==stage.seed


def test_standard_change_is_not_accepted():
    from dataclasses import replace
    stage,standard,_=configs();payload={"training_config_fingerprint":"bad"}
    assert not _comparable_payload(payload,stage,standard)
    assert not _comparable_payload({"training_config_fingerprint":__import__("songo_ai.training.lot46a",fromlist=["canonical_hash"]).canonical_hash(stage.payload())},stage,replace(standard,learning_rate=.01))


def test_training_loss_exposes_ce_kl_entropy_and_legal_mask():
    source=inspect.getsource(step_model)
    assert "mask_policy_logits" in source and "train_policy_kl" in source and "train_policy_entropy" in source
    assert "unconstrained" in source and "policy if unconstrained" in source


def test_validation_exposes_policy_entropy():
    assert "policy_entropy" in inspect.getsource(evaluate)


def test_two_consecutive_validation_degradations_stop(tmp_path):
    _,_,config=configs();config=__import__("dataclasses").replace(config,output_directory=str(tmp_path/config.experiment_id))
    out=Path(config.output_directory);out.mkdir(parents=True)
    (out/"stage_b2_baseline.json").write_text('{"validation":{"policy_ce":1.0,"policy_entropy":1.0}}')
    (out/"stage_b2_evaluation-00001000.json").write_text('{"step":1000,"validation":{"policy_ce":1.1,"policy_entropy":1.0},"correction":{"POTENTIAL_REGRESSION_RATE":0.1}}')
    current={"step":2000,"validation":{"policy_ce":1.1,"policy_kl":.2,"policy_entropy":1.0},"correction":{"POTENTIAL_REGRESSION_RATE":.1}}
    assert _decision(config,ROOT,current)["decision"]=="STOP"


def test_legacy_agreement_never_vetoes(tmp_path):
    _,_,config=configs();config=__import__("dataclasses").replace(config,output_directory=str(tmp_path/config.experiment_id))
    out=Path(config.output_directory);out.mkdir(parents=True)
    (out/"stage_b2_baseline.json").write_text('{"validation":{"policy_ce":1.0,"policy_entropy":1.0}}')
    current={"step":1000,"validation":{"policy_ce":1.0,"policy_kl":.2,"policy_entropy":1.0},"correction":{"POTENTIAL_REGRESSION_RATE":.1},"legacy_agreement":{"value":0.0}}
    decision=_decision(config,ROOT,current);assert decision["decision"]=="CONTINUE" and not decision["legacy_agreement_is_veto"]


def test_b2_module_has_no_arena_promotion_or_target_generation_entrypoint():
    import songo_ai.training.lot46b_b2 as module
    assert not any(name in vars(module) for name in ("arena","promote","generate_targets","train_value"))
