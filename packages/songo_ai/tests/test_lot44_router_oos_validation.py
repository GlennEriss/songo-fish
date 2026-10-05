import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "apps/trainer/scripts/run_lot44_router_oos_validation.py"
SCRIPT = SCRIPT_PATH.read_text(encoding="utf-8")


def test_lot44_script_parses_and_preserves_negative_lot43():
    ast.parse(SCRIPT)
    assert '"LOT43_VALID_IS_NO"' in SCRIPT
    assert 'd.get("LOT43_VALID") == "NO"' in SCRIPT
    assert "fb248b86" in SCRIPT
    assert "TRAINING_PERFORMED" in SCRIPT
    assert "MINIMAX_LABELS_USED" in SCRIPT


def test_lot44_has_strict_oos_split_and_test_lock():
    for artifact in (
        "independent_corpus_manifest.json", "overlap_audit.json", "split_manifest.json",
        "train_fingerprints.json", "calibration_fingerprints.json", "test_fingerprints.json",
        "test_lock.json", "router_frozen_manifest.json", "leakage_audit.json",
    ):
        assert artifact in SCRIPT
    assert "deterministic_group_aware" in SCRIPT
    assert "include_label=part != \"test\"" in SCRIPT
    assert "test already evaluated" in SCRIPT


def test_lot44_label_and_features_are_frozen_pre_65536():
    assert "MATERIAL_LATE_BIFURCATION_V1" in SCRIPT
    assert '"uses_65536": False' in SCRIPT
    assert "action_flip_count" in SCRIPT
    assert "ranking_volatility" in SCRIPT
    assert "distribution_drift" in SCRIPT
    assert "regularized_logistic_regression" in SCRIPT
    assert "Platt scaling" in SCRIPT


def test_lot44_reports_restart_resume_and_uncertainty():
    assert "classifier_restart" in SCRIPT
    assert "classifier_resume" in SCRIPT
    assert "COUNTERFACTUAL_ESTIMATE" in SCRIPT
    assert "recall_wilson_95" in SCRIPT
    assert '"MCTS_RESUME_CURRENTLY_SUPPORTED": "NO"' in SCRIPT
    assert '"EARLY_STOP_8192_ALLOWED": "NO"' in SCRIPT


def test_lot44_notebook_is_valid_and_calls_runner():
    notebook = json.loads((ROOT / "notebooks/lot44_router_oos_validation.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert notebook["nbformat"] == 4
    assert "run_lot44_router_oos_validation.py" in source
    for stage in ("validate", "corpus", "split", "search", "features", "train", "calibrate", "freeze", "evaluate", "finalize", "export"):
        assert stage in source
