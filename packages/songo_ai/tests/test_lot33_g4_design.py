import ast,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path("apps/trainer/scripts").resolve()))
from songo_ai.training import DesignError,validate_g4_design,promotion_decision,generator_admission_decision
import design_srn_lot33 as lot33

def load(name):return json.loads((lot33.OUT/name).read_text())

def test_dataset_composition_provenance_balancing_and_terminal_contracts():
 lot33.main();report=load("report.json");d=report["design"];assert validate_g4_design(d)
 assert sum(d["generation_ratio"].values())==100 and sum(d["training_source_ratio"].values())==100
 comp=load("dataset_composition.json");assert comp["D_TEACHER_role"].endswith("annotations forbidden")
 replay=load("replay_strategy.json");assert replay["name"]=="GENERATION_BALANCED_SOURCE_STRATIFIED" and not replay["strategic_disagreement_oversampling"]
 value=load("value_protocol.json");assert "truncated games" in value["forbidden_sources"] and "G2_G3 terminal" in value["allowed_sources"]

def test_split_by_game_component_identity_and_checkpoint_contract():
 strategic=load("strategic_test_design.json");assert strategic["split_unit"]=="game_id" and strategic["crossplay_game_atomic"]
 training=load("training_protocol.json");assert training["initial_policy"]["component"]=="G3_STRATEGIC" and training["initial_value"]["component"]=="V28_A"
 selection=load("component_selection.json");assert selection["max_policy_candidates"]==selection["max_value_candidates"]==2 and selection["max_combinations"]==4
 checkpoint=load("checkpoint_selection.json");assert checkpoint["selection_data"].startswith("VALIDATION") and checkpoint["no_posthoc_epoch_arena_selection"]

def test_opponent_search_promotion_and_generator_rules():
 assert load("opponent_battery.json")["opponents"]==["G2","G3_VALUE_REWORK"]
 search=load("search_robustness_rule.json");assert search["primary_budgets"]==[64,128] and not search["strict_monotonicity_required"]
 good={"champion_pooled_score":.53,"champion_bootstrap_p_gt_50":.97,"population_score":.50,"population_ci_low":.44,"max_side_gap":.06,"max_replicated_budget_drop":.04,"search_collapse_reproduced":False,"data_or_model_pathology":False,"strategic_preservation":.9}
 assert promotion_decision(good)["promoted"];good["population_score"]=.40;assert not promotion_decision(good)["promoted"]
 assert generator_admission_decision({"champion_pooled_score":.45,"strategic_diversity_gain":.06,"marginal_data_gain":0.,"data_or_model_pathology":False,"compute_cost_ratio":1.5})["admitted"]

def test_control_is_causal_and_no_training_code_exists():
 control=load("experiment_control.json");assert control["causal_variable"]=="data-generation paradigm" and control["CONTROL_G4"]["updates"]==control["POOL_G4"]["updates"]
 tree=ast.parse(Path(lot33.__file__).read_text());calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
 assert {"backward","step","train","save_checkpoint"}.isdisjoint(calls) and load("report.json")["training_performed"] is False

def test_invalid_design_is_rejected():
 d=load("report.json")["design"];d["training_source_ratio"]["NEW_GENERATION"]=71
 with pytest.raises(DesignError):validate_g4_design(d)
