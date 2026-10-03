from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[3]
SCRIPT=(ROOT/"apps/trainer/scripts/run_srn_lot39.py").read_text()
MCTS=(ROOT/"packages/songo_ai/search/mcts.py").read_text()

def test_lot39_fixed_work_and_concurrency_contract():
 assert "CONCURRENCIES=(16,32,64,128,256)" in SCRIPT
 assert "POSITIONS=256" in SCRIPT and "BUDGET=4096" in SCRIPT
 assert "POSITIONS*BUDGET" in SCRIPT

def test_lot39_uses_independent_batched_trees_and_exact_budget():
 assert "search_many(wave" in SCRIPT
 assert "per-tree simulation budget mismatch" in SCRIPT
 assert "requested_simulations_per_tree" in SCRIPT

def test_lot39_is_measurement_only():
 assert '"training_performed":False' in SCRIPT
 assert ".backward(" not in SCRIPT and "torch.optim" not in SCRIPT
 assert '"MODEL_WEIGHTS_CHANGED"' in SCRIPT

def test_lot39_resume_and_checksums():
 assert "valid_stage" in SCRIPT and '"status":"COMPLETE"' in SCRIPT
 assert "stage_state.json" in SCRIPT and "artifact_checksums" in SCRIPT
 assert ".partial.json" in SCRIPT and "completed_waves" in SCRIPT
 assert "checkpoint=" in SCRIPT and "flush=True" in SCRIPT
 assert "resume_code_compatible" in SCRIPT and "RESUME_CRITICAL_FILES" in SCRIPT

def test_lot39_profiles_required_runtime_categories():
 for key in ("engine_s","tree_selection_s","graph_construction_s","tensor_preparation_s","host_to_device_s","model_forward_wall_s","device_to_host_s","backup_s","batch_coordination_s"):
  assert key in MCTS or key in SCRIPT

def test_lot39_notebook_is_valid_and_has_independent_stages():
 notebook=json.loads((ROOT/"notebooks/songo_colab_lot39_gpu_scaling.ipynb").read_text())
 text=json.dumps(notebook)
 for concurrency in (16,32,64,128,256):assert f"'{concurrency}'" in text
 assert notebook["nbformat"]==4 and "lot39_inputs.tar.gz" in text
 assert "supervised_run" in text and "code de sortie" in text and "actif depuis" in text

def test_lot39_report_is_artifact_driven():
 assert "raw_scaling_results.json" in SCRIPT
 assert "svg_chart(rows" in SCRIPT and "srn_gpu_parallel_scaling_lot39.md" in SCRIPT
