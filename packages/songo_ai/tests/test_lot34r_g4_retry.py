import ast,json,sys
from pathlib import Path
sys.path.insert(0,str(Path("apps/trainer/scripts").resolve()))
import run_srn_lot34r as retry

ROOT=Path("data/experiments/lot34r_g4_retry")

def test_reuses_exact_lot34_corpora_and_generates_no_games():
 identity=json.loads((ROOT/"lot34_dataset_identity.json").read_text())
 assert identity["LOT34_DATA_REUSED_EXACTLY"]=="YES" and identity["NEW_GENERATION_PERFORMED"]=="NO"
 assert identity["CONTROL"]["games"]==identity["POOL"]["games"]==8000
 assert identity["CONTROL"]["positions"]==704816 and identity["POOL"]["positions"]==714700
 source=Path(retry.__file__).read_text();tree=ast.parse(source)
 assert "generate_selfplay" not in source and "run_population_selfplay" not in source
 assert not any(isinstance(n,ast.FunctionDef) and "generation" in n.name for n in ast.walk(tree))

def test_forensics_are_ordered_and_include_update_zero():
 report=json.loads((ROOT/"checkpoint_forensics.json").read_text())
 for arm in ("CONTROL","POOL"):
  updates=[x["update"] for x in report["curves"][arm]]
  assert updates==[0,4000,8000,12000,16000,20000,24000,28000,32000]
  assert report["curves"][arm][0]["preservation_rate"]==1
 assert report["SAFE_CHECKPOINT_EXISTS_CONTROL"]==report["SAFE_CHECKPOINT_EXISTS_POOL"]=="NO"

def test_gate_and_single_intervention_are_pre_registered():
 config=json.loads((ROOT/"configuration.json").read_text());intervention=json.loads((ROOT/"retry_intervention.json").read_text())
 assert config["strategic_preservation_gate"]==.85
 assert intervention["name"]=="CORRECT_AND_PRESERVE" and intervention["common_to_both_arms"]
 assert (intervention["lambda_correction"],intervention["lambda_preservation"],intervention["rho"])==(.1,1.,.5)
 assert intervention["no_lambda_sweep"] and intervention["max_updates"]==32000

def test_frozen_sources_initialization_objectives_and_candidate_limits():
 source=Path(retry.__file__).read_text()
 assert "LOT34/\"control_data\"" in source and "LOT34/\"pool_data\"" in source
 assert "load_historical()" in source and "load_reanalysis_rows(holder)" in source
 assert "make_initial_model()" in source and "MSE_TO_TRUE_TERMINAL_Z_ONLY" in source
 assert "value_true_terminal_z_only\":True" in source and "truncated_value_excluded\":True" in source
 assert '"max_per_arm":2' in source and '"max_combinations_per_arm":4' in source
 assert '"teacher_labels":False' in source and '"minimax_labels":False' in source

def test_main_arena_requires_both_strategic_gates():
 source=Path(retry.__file__).read_text()
 assert '"MAIN_ARENAS_ALLOWED":"YES" if all(x["pass"] for x in gate.values()) else "NO"' in source
 assert "preservation_rate\"]>=GATE" in source
