from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (ROOT / "apps/trainer/scripts/analyze_srn_lot41b.py").read_text()


def test_lot41b_is_analysis_only():
    assert "SongoMCTS" not in SCRIPT
    assert "load_model" not in SCRIPT
    assert ".backward(" not in SCRIPT
    assert "torch.optim" not in SCRIPT
    assert "NEW_MCTS_SEARCH_PERFORMED" in SCRIPT
    assert '"NO"' in SCRIPT


def test_lot41b_uses_completed_budget_chain_only():
    assert "BUDGETS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)" in SCRIPT
    assert "STAGE_BUDGETS" not in SCRIPT
    assert "MCTS65536_SCIENTIFICALLY_USEFUL" in SCRIPT
    assert "131072" not in SCRIPT
    assert "B_REF = 32768" in SCRIPT


def test_lot41b_outputs_required_artifacts():
    for artifact in (
        "input_validation.json",
        "position_alignment.json",
        "action_trajectories.json",
        "top1_adjacent.json",
        "top1_vs_32768.json",
        "action_flips.json",
        "action_stability_budget.json",
        "js_adjacent.json",
        "js_vs_32768.json",
        "ranking_adjacent.json",
        "ranking_vs_32768.json",
        "top2_stability.json",
        "q_validity.json",
        "deep_search_regret.json",
        "flip_severity.json",
        "stability_summary.json",
        "compute_value_tradeoff.json",
        "difficulty_distribution.json",
        "uncertainty_signals.json",
        "hard_positions.json",
        "outliers.json",
        "teacher_budget_decision.json",
        "decision.json",
        "report.json",
        "budget_convergence_summary.csv",
        "position_stability.csv",
    ):
        assert artifact in SCRIPT


def test_lot41b_has_multilevel_metrics():
    for metric in (
        "top1_agreement",
        "final_action_agreement",
        "jensen_shannon",
        "kendall_agreement",
        "same_top2_set_rate",
        "Q_REGRET_AVAILABLE",
        "high_severity_disagreement_rate",
        "action_stability_budget",
        "MULTI_FIDELITY_RECOMMENDED",
        "ADAPTIVE_DEEPENING_FEASIBLE",
    ):
        assert metric in SCRIPT
