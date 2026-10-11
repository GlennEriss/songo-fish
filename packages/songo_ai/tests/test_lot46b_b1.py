import inspect
import json
from pathlib import Path

import pytest

from songo_ai.training.lot46 import Lot46Error
from songo_ai.training.lot46a import ExperimentConfig, evaluate, step_model
from songo_ai.training.lot46b_b1 import (GENERATIONAL_KNOWLEDGE_IS_PROVISIONAL,
    audit_policy_spec, audit_value_spec, validate_budget)

ROOT=Path(__file__).resolve().parents[3]


def policy_spec(**updates):
    value={"objective":"DEEP_POLICY_CE_ONLY","target_sources":["LOT45_MCTS32768"],
        "legal_action_mask":True,"distillation_teacher":None,"lambda_preservation":None}
    value.update(updates); return value


def test_unconstrained_has_no_g4_labels_or_distillation():
    report=audit_policy_spec(policy_spec())
    assert report["status"]=="PASS"
    assert report["checks"]["no_g4_labels"] and report["checks"]["no_distillation"]
    assert audit_policy_spec(policy_spec(target_sources=["G4R_LOGITS"]))["status"]=="FAIL"


def test_minimax_and_mcts_q_are_forbidden_policy_labels():
    assert audit_policy_spec(policy_spec(target_sources=["MINIMAX_BIDOUA"]))["status"]=="FAIL"
    assert audit_policy_spec(policy_spec(target_sources=["MCTS_Q"]))["status"]=="FAIL"


def test_legal_mask_and_deep_provenance_are_mandatory():
    assert audit_policy_spec(policy_spec(legal_action_mask=False))["status"]=="FAIL"
    assert audit_policy_spec(policy_spec(target_sources=["AUTONOMOUS_MCTS32768_RELABELED"]))["status"]=="PASS"


def test_unconstrained_config_is_new_isolated_identity():
    config=ExperimentConfig.load(ROOT/"configs/lot46b_b2/stage_b_deep_unconstrained.json")
    assert config.experiment_id==Path(config.output_directory).name
    assert config.experiment_id!="LOT46B_B2_STAGE_A_DEEP_POLICY_001"
    assert config.objective=={"name":"DEEP_POLICY_CE_ONLY","legal_action_mask":True}


def test_unconstrained_config_rejects_indirect_legacy_labels(tmp_path):
    raw=json.loads((ROOT/"configs/lot46b_b2/stage_b_deep_unconstrained.json").read_text())
    raw["dataset_sources"][0]["target_source"]="HISTORICAL_AUTONOMOUS_MCTS"
    path=tmp_path/"bad.json"; path.write_text(json.dumps(raw))
    with pytest.raises(Lot46Error,match="autonomous deep targets"): ExperimentConfig.load(path)


def test_value_uses_true_terminal_z_and_masks_missing_targets():
    assert audit_value_spec({"target_source":"TRUE_TERMINAL_Z","missing_target_mask":True})["status"]=="PASS"
    assert audit_value_spec({"target_source":"MCTS_Q","missing_target_mask":True})["status"]=="FAIL"
    source=inspect.getsource(step_model)
    assert 'if vm.any()' in source and 'optimization_skipped_no_value_labels' in source


def test_value_validation_persists_predictions_calibration_and_groups():
    source=inspect.getsource(evaluate)
    for field in ("value_predictions","value_calibration","value_errors_by_phase","value_errors_by_terminal_result"):
        assert field in source


def test_budget_prevents_false_resume_for_changed_loss():
    matrix=[{"name":"NEW","category":"NEW_EXPERIMENT","already_consumed":0,"initial_step":0,
        "additional_budget":4000,"cumulative_budget":4000}]
    assert validate_budget(matrix)["status"]=="PASS"
    matrix[0]["already_consumed"]=1000
    assert validate_budget(matrix)["status"]=="FAIL"


def test_historical_artifacts_are_not_mutated_by_b1_module():
    import songo_ai.training.lot46b_b1 as module
    source=inspect.getsource(module)
    assert "unlink(" not in source and "rmtree(" not in source and "torch.save(" not in source
    assert not any(name in vars(module) for name in ("run_training","arena","promote"))


def test_generation_knowledge_is_explicitly_provisional():
    assert GENERATIONAL_KNOWLEDGE_IS_PROVISIONAL is True
