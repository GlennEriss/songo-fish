from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[3]
SCRIPT=(ROOT/"apps/trainer/scripts/run_srn_colab_benchmark.py").read_text()
IMPORT=(ROOT/"apps/trainer/scripts/import_remote_experiment.py").read_text()
def test_notebook_is_minimal_valid_json():
 d=json.loads((ROOT/"notebooks/songo_colab_compute_benchmark.ipynb").read_text());assert d["nbformat"]==4;assert len(d["cells"])<=9
def test_environment_device_and_cpu_fallback_contract():
 assert "torch.cuda.is_available()" in SCRIPT and 'requested=="cuda"' in SCRIPT and 'torch.device("cpu")' in SCRIPT
 assert all(x in SCRIPT for x in ("python_version","torch_version","numpy_version","numba_version","GPU_name","GPU_memory_bytes"))
def test_deterministic_positions_and_budget_contract():
 assert "POSITIONS=16" in SCRIPT and "SEED=20263801" in SCRIPT
 assert 'choices=(1024,4096)' in SCRIPT and "MCTS65536" not in SCRIPT
def test_single_state_mcts_and_batched_srn_only():
 assert '"SINGLE_STATE"' in SCRIPT and "BATCHES=(1,8,16,32,64,128,256)" in SCRIPT
 assert "SongoMCTS" in SCRIPT and "build_batch" in SCRIPT
def test_no_training_or_mutation_contract():
 assert '"training_performed":False' in SCRIPT and '"optimizer_created":False' in SCRIPT and '"backward_called":False' in SCRIPT
 assert ".backward(" not in SCRIPT and "torch.optim" not in SCRIPT
def test_bundle_checksums_and_strict_import():
 assert "artifact_checksums" in SCRIPT and "experiment_manifest.json" in SCRIPT
 assert "cannot export before successful finalize" in SCRIPT
 assert "checksum mismatch" in IMPORT and "engine fingerprint mismatch" in IMPORT and "model fingerprint mismatch" in IMPORT
 assert "git commit mismatch" in IMPORT and "unsafe bundle path" in IMPORT
def test_fingerprint_preservation():
 assert "model_parameter_fingerprint" in SCRIPT and 'before["unchanged"]' in SCRIPT

def test_gpu_only_finalize_is_supported_and_not_misclassified():
 assert "efficient=None if speed is None" in SCRIPT
 assert 'under="YES" if efficient is False and beneficial is True' in SCRIPT
 assert '"INCONCLUSIVE"' in SCRIPT

def test_notebook_stops_export_when_finalize_fails():
 notebook=(ROOT/"notebooks/songo_colab_compute_benchmark.ipynb").read_text()
 assert "subprocess.run" in notebook and "check=True" in notebook
