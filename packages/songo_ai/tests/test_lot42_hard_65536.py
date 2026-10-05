from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (ROOT / "apps/trainer/scripts/run_srn_lot42.py").read_text()


def test_lot42_targets_hard_positions_only():
    assert "hard_positions.json" in SCRIPT
    assert "HARD_POSITION_COUNT" in SCRIPT
    assert "hard_set_sha" in SCRIPT
    assert "Do not" not in SCRIPT  # script contains implementation, not protocol prose
    assert "NEW_MCTS_SEARCH_ON_FULL_256" in SCRIPT


def test_lot42_budget_and_mcts_contract():
    assert "BUDGET = 65536" in SCRIPT
    assert "REF_BUDGET = 32768" in SCRIPT
    assert "add_root_noise=False" in SCRIPT
    assert "LOT40_FLAGS" in SCRIPT
    assert "actual_simulations" in SCRIPT
    assert "requested_simulations" in SCRIPT


def test_lot42_no_training_or_minimax():
    assert ".backward(" not in SCRIPT
    assert "torch.optim" not in SCRIPT
    assert "MINIMAX_LABELS_USED" in SCRIPT
    assert "TRAINING_PERFORMED" in SCRIPT
    assert '"NO"' in SCRIPT


def test_lot42_resume_and_artifacts():
    for token in (
        "memory_pilot.json",
        "execution_plan.json",
        "shards",
        "checksum",
        "search_65536_results.json",
        "comparison_32768_65536.json",
        "ultra_hard_positions.json",
        "hard_but_stable_at_32768.json",
        "hard_position_65536_summary.csv",
        "decision.json",
        "experiment_manifest.json",
    ):
        assert token in SCRIPT


def test_lot42_scientific_decision_outputs():
    for token in (
        "TOP1_AGREEMENT_32768_65536",
        "ACTION_FLIP_RATE_32768_65536",
        "MEDIAN_JS_32768_65536",
        "P90_JS_32768_65536",
        "MCTS65536_ADDITIONAL_INFORMATION",
        "MCTS65536_ROLE",
        "MULTI_FIDELITY_CONFIRMED",
        "NEXT_LOT",
    ):
        assert token in SCRIPT
