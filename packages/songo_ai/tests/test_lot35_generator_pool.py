import ast,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path("apps/trainer/scripts").resolve()))
import run_srn_lot35 as lot35
from songo_ai.generation.population import ModelRole,RegisteredModel,validate_generator_pool

ROOT=Path("data/experiments/lot35_generator_pool")

def load(name):return json.loads((ROOT/name).read_text())

def test_g4_identity_is_honestly_unresolved_and_candidates_are_fingerprinted():
 identity=load("g4_champion_identity.json")
 assert identity["G4_CHAMPION_IDENTITY_RESOLVED"]=="NO" and identity["no_posthoc_choice"]
 assert set(identity["candidates"])=={"CONTROL","POOL"}
 for row in identity["candidates"].values():
  assert len(row["policy_fingerprint"])==len(row["value_fingerprint"])==64
  assert Path(row["policy_checkpoint"]).exists() and Path(row["value_checkpoint"]).exists()

def test_lot35_performs_no_training_or_massive_generation_and_freezes_stack():
 config=load("configuration.json");source=Path(lot35.__file__).read_text();tree=ast.parse(source)
 assert not config["training_performed"] and not config["massive_g5_generation_performed"]
 assert not config["engine_modified"] and not config["srn_modified"] and not config["mcts_modified"]
 calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
 assert {"backward","step","generate_selfplay","run_population_selfplay"}.isdisjoint(calls)

def test_thresholds_are_pre_registered_and_diagnostic_generation_is_blocked():
 config=load("configuration.json");plan=load("diagnostic_generation_plan.json")
 assert config["thresholds"]["pre_registered_before_diagnostic_trajectories"]
 assert plan["status"]=="NOT_EXECUTED_IDENTITY_GATE_CLOSED" and plan["side_balanced"]
 assert plan["not_d_g5"] and plan["diagnostic_games_per_source"]==512

def test_registry_roles_and_provenance_are_explicit():
 registry=load("model_registry.json")
 assert registry["models"]["G2"]["roles"]==["HISTORICAL_OPPONENT"]
 assert registry["models"]["G3_VALUE_REWORK"]["roles"]==["HISTORICAL_OPPONENT"]
 assert registry["models"]["CONTROL_G4R"]["roles"]==["PROBATIONARY_GENERATOR"]
 assert registry["models"]["POOL_G4R"]["roles"]==["PROBATIONARY_GENERATOR"]
 assert len(registry["training_protocol_fingerprint"])==64

def test_generator_pool_contract_requires_unique_active_champion_and_maximum_three():
 champion=RegisteredModel("g4","a",frozenset({ModelRole.CHAMPION,ModelRole.ACTIVE_GENERATOR}),"p","v")
 historical=RegisteredModel("g2","b",frozenset({ModelRole.HISTORICAL_OPPONENT}),"p","v")
 assert validate_generator_pool([champion,historical],max_active_generators=3)
 with pytest.raises(ValueError):validate_generator_pool([historical],max_active_generators=3)
 with pytest.raises(ValueError):validate_generator_pool([champion,*[RegisteredModel(f"g{i}",str(i),frozenset({ModelRole.ACTIVE_GENERATOR}),"p","v") for i in range(3)]],max_active_generators=3)

def test_g5_contract_is_safe_but_not_executable_until_identity_resolution():
 design=load("g5_generation_design.json");pool=load("generator_pool_v2.json");report=load("report.json")["verdict"]
 assert design["G5_POLICY_OBJECTIVE"]=="CORRECT_AND_PRESERVE"
 assert design["G5_VALUE_OBJECTIVE"]=="MSE_TO_TRUE_TERMINAL_Z_ONLY"
 assert design["POLICY_VALUE_SELECTION_DECOUPLED"]=="YES"
 assert not pool["valid"] and pool["ACTIVE_GENERATOR_POOL"]==[]
 assert report["NEXT_ACTION"]=="G4_CHAMPION_IDENTITY_RESOLUTION"
