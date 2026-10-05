from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (ROOT / "apps/trainer/scripts/run_srn_lot43.py").read_text()


def test_lot43_is_dry_run_and_training_free():
    assert "SongoMCTS" not in SCRIPT
    assert "load_model" not in SCRIPT
    assert ".backward(" not in SCRIPT
    assert "torch.optim" not in SCRIPT
    assert "TRAINING_PERFORMED" in SCRIPT
    assert "OPTIMIZER_CREATED" in SCRIPT
    assert "BACKWARD_CALLED" in SCRIPT
    assert "MODEL_WEIGHTS_CHANGED" in SCRIPT


def test_lot43_budget_hierarchy_and_no_extreme_search():
    assert "ROUTINE_HIERARCHY = (4096, 8192, 32768, 65536)" in SCRIPT
    assert "131072" not in SCRIPT
    assert "MCTS16384_ROUTING_STAGE" in SCRIPT
    assert '"REMOVE"' in SCRIPT
    assert "REF_ORDINARY = 32768" in SCRIPT
    assert "REF_ULTRA = 65536" in SCRIPT


def test_lot43_feature_schema_blocks_future_leakage():
    assert "uses_future_information" in SCRIPT
    assert "allowed_for_router" in SCRIPT
    assert "ultra_hard_label" in SCRIPT
    assert '"allowed_for_router": False' in SCRIPT
    assert "FUTURE_INFORMATION_LEAKAGE" in SCRIPT
    assert "forbidden_features" in SCRIPT


def test_lot43_outputs_required_artifacts():
    for artifact in (
        "input_validation.json",
        "source_artifact_manifest.json",
        "feature_schema.json",
        "oracle_required_budgets.json",
        "oracle_policy.json",
        "signal_analysis_4096.json",
        "signal_analysis_8192.json",
        "signal_analysis_32768.json",
        "candidate_routers.json",
        "router_validation.json",
        "routing_matrix_4096.json",
        "routing_matrix_8192.json",
        "routing_matrix_32768.json",
        "false_stops.json",
        "false_deepens.json",
        "false_early_stability.json",
        "ultra_hard_routing.json",
        "compute_comparison.json",
        "compute_projection.json",
        "oracle_vs_deployable.json",
        "multi_fidelity_teacher_v1.json",
        "teacher_protocol_fingerprint.json",
        "decision.json",
        "report.json",
        "experiment_manifest.json",
        "checksums.json",
        "router_position_results.csv",
        "router_comparison.csv",
    ):
        assert artifact in SCRIPT


def test_lot43_protocol_serialization_and_target_semantics():
    assert "MULTI_FIDELITY_TEACHER_V1" in SCRIPT
    assert "sha256(proto_blob)" in SCRIPT
    assert "pi_teacher = visit_distribution at final selected budget" in SCRIPT
    assert "raw visit counts retained" in SCRIPT
    assert "terminal z only" in SCRIPT
    assert "RESTART" in SCRIPT
