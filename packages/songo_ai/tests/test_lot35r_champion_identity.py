import ast,json,sys
from pathlib import Path
sys.path.insert(0,str(Path("apps/trainer/scripts").resolve()))
import run_srn_lot35r as lot35r

def test_exactly_two_fingerprinted_candidates_and_frozen_models(tmp_path):
 class A:output=tmp_path
 identity=lot35r.prepare(A);assert set(identity)=={"CONTROL","POOL"}
 for row in identity.values():assert len(row["policy_fingerprint"])==len(row["value_fingerprint"])==64
 assert json.loads((tmp_path/"candidate_identity.json").read_text())["fingerprints_frozen_before_arena"]

def test_phase_contracts_are_predeclared_and_independent(tmp_path):
 class A:output=tmp_path
 lot35r.prepare(A);p1=json.loads((tmp_path/"phase1_seed_manifest.json").read_text());p2=json.loads((tmp_path/"phase2_seed_manifest.json").read_text())
 assert (p1["games"],p1["opening_pairs"],p1["budget"])==(1024,512,128)
 assert (p2["games"],p2["opening_pairs"],p2["budget"])==(1024,512,256)
 assert not ({p1["openings_seed"],p1["arena_seed"],p1["bootstrap_seed"]}&{p2["openings_seed"],p2["arena_seed"],p2["bootstrap_seed"]})

def test_paired_bootstrap_and_immutable_probability_threshold():
 source=Path(lot35r.__file__).read_text()
 assert "THRESHOLD=.95" in source and '"bootstrap_unit":"opening_pair"' in source
 assert "count=512" in source and "run_paired_arena" in source
 assert 'summary["by_a_side"]["P1"]["games"]==summary["by_a_side"]["P2"]["games"]==512' in source

def test_phase2_is_conditional_and_no_additional_tiebreak_exists():
 source=Path(lot35r.__file__).read_text();tree=ast.parse(source)
 assert 'if p1["PHASE1_DECISION"]!="INCONCLUSIVE":raise RuntimeError' in source
 assert "MCTS512" not in source and "2048" not in source and "4096" not in source
 assert "preservation" not in source.lower() and "vs_g2" not in source.lower()

def test_no_training_generation_pruning_or_model_mutation():
 source=Path(lot35r.__file__).read_text();tree=ast.parse(source);calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
 assert {"backward","step","train","generate_selfplay","run_population_selfplay"}.isdisjoint(calls)
 assert '"no_pruning":True' in source and '"G5_TRAINING_PERFORMED":"NO"' in source
 assert "load_state_dict" not in source and "torch.save" not in source
