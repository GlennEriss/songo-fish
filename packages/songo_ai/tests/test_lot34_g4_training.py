import ast,json,sys
from pathlib import Path
sys.path.insert(0,str(Path("apps/trainer/scripts").resolve()))
import run_srn_lot34_generation as gen
import run_srn_lot34_training as training
import run_srn_lot34_evaluation as evaluation

def test_exact_8k_generation_and_pool_ratio_side_balance():
 schedules=gen.schedules();assert all(len(v)==8000 for v in schedules.values())
 pool=schedules["POOL"];assert sum(x.p1_model_id==x.p2_model_id=="G2" for x in pool)==800
 assert sum(x.p1_model_id==x.p2_model_id=="G3_VALUE_REWORK" for x in pool)==800
 assert sum(x.p1_model_id=="G2" and x.p2_model_id=="G3_VALUE_REWORK" for x in pool)==3200
 assert sum(x.p1_model_id=="G3_VALUE_REWORK" and x.p2_model_id=="G2" for x in pool)==3200

def test_independent_predeclared_seeds_and_equal_search():
 schedules=gen.schedules();assert not ({x.seed for x in schedules["CONTROL"]}&{x.seed for x in schedules["POOL"]})
 source=Path(gen.__file__).read_text();assert "num_simulations=64" in source and '"training_updates":32000' in source

def test_generation_has_no_training_teacher_or_engine_mutation():
 source=Path(gen.__file__).read_text();tree=ast.parse(source);calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
 assert {"backward","step","train","save_checkpoint"}.isdisjoint(calls)
 assert '"teacher_labels":False' in source and '"minimax_labels":False' in source and '"engine_modified":False' in source and '"srn_modified":False' in source

def test_common_reanalysis_is_identical_and_value_free():
 source=Path(training.__file__).read_text();assert '"same_states":True' in source and '"shared_between_training_arms":True' in source
 assert '"value_targets":False' in source and '"teacher_labels":False' in source and '"minimax_labels":False' in source

def test_training_budget_optimizer_candidate_and_combination_limits():
 source=Path(training.__file__).read_text();assert "range(1,32001)" in source and "batch_size=256" in source
 assert "AdamW" in source and "max_per_arm\":2" in source and "max_combinations_per_arm\":4" in source
 assert "split_code(x.metadata[\"game_id\"])" in source

def test_independent_arena_seeds_paired_side_swap_and_fixed_sizes(tmp_path):
 class A:output=tmp_path
 seeds=evaluation.prepare(A)["main"];assert len({x for row in seeds.values() for x in row.values()})==20
 source=Path(evaluation.__file__).read_text();assert "count=256" in source and '"precommitted_games":512' in source
 assert "run_paired_arena" in source and '"side_swapped":True' in source

def test_training_contract_terminal_z_teacher_free_and_frozen_implementation():
 source=Path(training.__file__).read_text()
 assert "vmask" in source and "value_target" in source and "pred.reshape(-1)[vmask]" in source
 assert '"teacher_labels":False' in source and '"minimax_labels":False' in source
 assert '"engine_modified":False' in Path(gen.__file__).read_text()
 assert '"srn_modified":False' in Path(gen.__file__).read_text()

def test_completed_run_observed_ratios_updates_and_initialization_are_equal():
 root=Path("data/experiments/lot34_g4_training")
 if not (root/"control_training_report.json").exists():return
 rows=[json.loads((root/f"{arm}_training_report.json").read_text()) for arm in ("control","pool")]
 assert all(x["updates"]==32000 for x in rows)
 assert all(x["initialization_parameter_delta_vs_reference"]==0 for x in rows)
 assert rows[0]["observed_ratio"]==rows[1]["observed_ratio"]
 assert rows[0]["optimizer"]==rows[1]["optimizer"]=="AdamW"

def test_strategic_gate_prevents_main_arena_when_no_policy_is_eligible():
 root=Path("data/experiments/lot34_g4_training")
 if not (root/"policy_strategic_gate.json").exists():return
 gate=json.loads((root/"policy_strategic_gate.json").read_text());report=json.loads((root/"report.json").read_text())
 assert gate["threshold"]==.85 and gate["any_eligible"]=={"CONTROL":False,"POOL":False}
 assert report["status"]=="ABORTED_BEFORE_MAIN_ARENAS"
 assert report["verdict"]["EXPERIMENT_VALID"]=="NO"
 assert report["verdict"]["OFFICIAL_CHAMPION"]=="G2"
