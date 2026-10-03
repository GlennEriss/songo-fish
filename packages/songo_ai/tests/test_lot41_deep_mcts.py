from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (ROOT / "apps/trainer/scripts/run_srn_lot41.py").read_text()
PREPARE = (ROOT / "apps/trainer/scripts/prepare_colab_inputs.py").read_text()


def test_lot41_fixed_scientific_contract():
    for budget in ("256", "512", "1024", "2048", "4096", "8192", "16384", "32768", "65536"):
        assert budget in SCRIPT
    assert "POOL_G4R" in SCRIPT
    assert '"dirichlet_for_convergence": "OFF"' in SCRIPT
    assert "add_root_noise=False" in SCRIPT
    assert "ANALYSIS_TEMPERATURE = 1.0" in SCRIPT


def test_lot41_no_training_or_teacher_labels():
    assert '"training_performed": False' in SCRIPT
    assert '"optimizer_created": False' in SCRIPT
    assert '"backward_called": False' in SCRIPT
    assert '"teacher_labels": False' in SCRIPT
    assert '"minimax_labels": False' in SCRIPT
    assert ".backward(" not in SCRIPT
    assert "torch.optim" not in SCRIPT


def test_lot41_records_root_contract_and_raw_visits():
    for field in (
        "visit_counts",
        "visit_distribution",
        "selected_action",
        "root_q_values",
        "root_value",
        "policy_prior",
        "legal_mask",
        "effective_batch_mean",
        "target_temperature",
    ):
        assert field in SCRIPT


def test_lot41_convergence_metrics_are_not_argmax_only():
    for metric in (
        "top1_agreement",
        "jensen_shannon",
        "kendall_distance",
        "root_q_abs_delta",
        "best_action_margin_delta",
        "action_stability_budget",
        "has_reversal",
        "practically_stable",
    ):
        assert metric in SCRIPT


def test_lot41_staged_execution_and_latency_are_separate():
    for stage in ("stage_a", "stage_b", "stage_c", "stage_d", "stage_e", "latency", "pilot", "finalize", "export"):
        assert stage in SCRIPT
    assert "SINGLE_SEARCH_LATENCY" not in SCRIPT
    assert "latency.json" in SCRIPT
    assert "estimated_total_compute.json" in SCRIPT


def test_lot41_colab_bundle_is_supported():
    assert '"lot41"' in PREPARE
    assert "LOT41_REFERENCE_FILES" in PREPARE
