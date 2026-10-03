from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[3]
SCRIPT=(ROOT/"apps/trainer/scripts/run_srn_lot40.py").read_text()

def test_lot40_fixed_scientific_contract():
 assert "HISTORICAL=3668.0900137098834" in SCRIPT
 assert "POSITIONS*BUDGET" in SCRIPT and '"parallel_searches":256' in SCRIPT
 assert "POOL_G4R" in SCRIPT

def test_lot40_has_isolated_optimization_arms():
 for stage in ("00_baseline","01_profile","02_coordination","03_tree","04_engine","05_graph","06_final","07_single_search","08_finalize"):
  assert stage in SCRIPT
 for flag in ("profile_runtime","compact_tree_ops","fast_engine_rebuild","vectorized_graph"):
  assert flag in SCRIPT

def test_lot40_enforces_exact_correctness_and_no_training():
 assert "semantic divergence" in SCRIPT and '"exact"' in SCRIPT
 assert '"training_performed":False' in SCRIPT
 assert ".backward(" not in SCRIPT and "torch.optim" not in SCRIPT
 assert "model_weights_changed" in SCRIPT

def test_lot40_persistent_stages_and_artifacts():
 assert "stage_valid" in SCRIPT and "checksum.json" in SCRIPT
 for artifact in ("baseline.json","coordination_profile.json","ablation.json","final_benchmark.json","single_search_latency.json","decision.json","report.json","experiment_manifest.json","checksums.json"):
  assert artifact in SCRIPT

def test_lot40_notebook_contract():
 path=ROOT/"notebooks/songo_colab_lot40_cpu_pipeline.ipynb"
 if path.exists():
  notebook=json.loads(path.read_text());text=json.dumps(notebook)
  assert notebook["nbformat"]==4 and "supervised_run" in text and "lot40_inputs.tar.gz" in text
