"""Integration avec les vraies APIs (moteur, POOL_G4R, SongoMCTS) a petit budget."""
import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from lot44.artifacts import ErrorLog, Lot44FatalError, read_json, sidecar, write_json
from lot44.corpus import state_from_dict
from lot44.finalize import finalize
from lot44.pipeline import Context, evaluate, prepare, run, search_step
from lot44.preflight import run_preflight
from lot44.search import SearchIdentity, run_search, validate_result
from lot44.smoke import run_smoke, smoke_inputs
from run_srn_lot39 import fingerprint_state

from conftest import ORIGINAL_POSITIONS, make_context, require_inputs

DECISION_KEYS = (
    "LOT44_VALID", "INDEPENDENT_CORPUS_VALID", "OOS_OVERLAP_WITH_ORIGINAL_256", "TRAIN_SIZE", "CALIBRATION_SIZE", "TEST_SIZE",
    "OOS_POSITIVE_COUNT", "OOS_NEGATIVE_COUNT", "ROUTER_VALIDATED", "OOS_RECALL", "OOS_RECALL_CI95", "OOS_PRECISION", "OOS_PR_AUC",
    "OOS_ROC_AUC", "OOS_FALSE_NEGATIVES", "WORST_FALSE_NEGATIVE_REGRET", "OOS_65536_ROUTING_RATE",
    "CLASSIC_CONFIDENCE_ONLY_PERFORMANCE", "TRAJECTORY_FEATURES_PERFORMANCE", "COMBINED_FEATURES_PERFORMANCE", "TRAJECTORY_SIGNALS_ADD_VALUE",
    "UNIFORM_32768_TOTAL_COST", "CONSERVATIVE_RESTART_COST", "CLASSIFIER_RESTART_COST", "CONSERVATIVE_RESUME_COST", "CLASSIFIER_RESUME_COST", "UNIFORM_65536_TOTAL_COST",
    "MCTS_RESUME_CURRENTLY_SUPPORTED", "MULTIFIDELITY_RESTART_ECONOMICALLY_VIABLE", "MULTIFIDELITY_RESUME_ECONOMICALLY_VIABLE",
    "UNIFORM_32768_DEFENSIBLE", "EARLY_STOP_8192_ALLOWED", "MODEL_WEIGHTS_CHANGED", "NEXT_ACTION",
)


def records(n):
    require_inputs()
    states = json.loads(ORIGINAL_POSITIONS.read_text())["states"][:n]
    return [{"fingerprint": fingerprint_state(state_from_dict(s)), "state": {"board": s["board"], "player_to_move": s["player_to_move"]}, "game_id": str(i)} for i, s in enumerate(states)]


def identity(budget=16, seed=1):
    return SearchIdentity(budget=budget, seed=seed, model_fingerprints={"m": "x"}, engine_fingerprint="e")


def search(out, model, recs, ident, **kw):
    return run_search(out, partition="train", records=recs, identity=ident, concurrency=2, model_loader=lambda: model, device=torch.device("cpu"), errors=ErrorLog(out / "errors.jsonl"), log=lambda _: None, **kw)


def test_search_resume_idempotence_and_seed_independence(tmp_path, model):
    recs = records(6)
    first = search(tmp_path, model, recs, identity(), max_shards=1)
    assert first["status"] == "PARTIAL" and first["computed"] == 2
    shard = next((tmp_path / "search/train/16").glob("shard_*.json"))
    before = (shard.read_bytes(), shard.stat().st_mtime_ns)
    second = search(tmp_path, model, recs, identity())
    assert second["status"] == "COMPLETE" and second["computed"] == 4 and second["already_complete"] == 2
    assert (shard.read_bytes(), shard.stat().st_mtime_ns) == before
    assert search(tmp_path, model, recs, identity())["computed"] == 0
    rows = {r["state_fingerprint"]: r for p in (tmp_path / "search/train/16").glob("shard_*_*.json") if not p.name.endswith(".sha256.json") for r in json.loads(p.read_text())["rows"]}
    other = tmp_path / "other"
    run_search(other, partition="train", records=recs, identity=identity(), concurrency=3, model_loader=lambda: model, device=torch.device("cpu"), errors=ErrorLog(other / "e.jsonl"), log=lambda _: None)
    rows3 = {r["state_fingerprint"]: r for p in (other / "search/train/16").glob("shard_*_*.json") if not p.name.endswith(".sha256.json") for r in json.loads(p.read_text())["rows"]}
    assert {k: v["visit_counts"] for k, v in rows.items()} == {k: v["visit_counts"] for k, v in rows3.items()}


def test_incomplete_shard_is_recomputed_and_corrupt_shard_is_fatal(tmp_path, model):
    recs = records(4)
    search(tmp_path, model, recs, identity())
    shards = sorted(p for p in (tmp_path / "search/train/16").glob("shard_*.json") if not p.name.endswith(".sha256.json"))
    sidecar(shards[0]).unlink()
    redo = search(tmp_path, model, recs, identity())
    assert redo["computed"] == 2
    payload = json.loads(shards[1].read_text())
    payload["rows"][0]["visit_counts"][0] += 1
    shards[1].write_text(json.dumps(payload))
    with pytest.raises(Lot44FatalError) as err:
        search(tmp_path, model, recs, identity())
    assert err.value.code == "CHECKSUM_MISMATCH"


def test_search_identity_change_is_fatal(tmp_path, model):
    recs = records(2)
    search(tmp_path, model, recs, identity(seed=1))
    with pytest.raises(Lot44FatalError) as err:
        search(tmp_path, model, recs, identity(seed=2))
    assert err.value.code == "MCTS_CONFIG_INCOMPATIBLE"


def test_validate_result_rejects_bad_results():
    good = SimpleNamespace(num_simulations=16, visit_counts=(8, 8, 0, 0, 0, 0, 0), selected_action=0, legal_mask=(True,) * 7, policy=(0.5, 0.5, 0, 0, 0, 0, 0), root_q_values=(0.0,) * 7, root_value=0.1)
    validate_result("f" * 64, good, 16)
    cases = {
        "SIMULATION_COUNT": dict(num_simulations=15),
        "ILLEGAL_ACTION": dict(legal_mask=(False,) + (True,) * 6),
        "NAN_CRITICAL": dict(root_value=float("nan")),
    }
    for code, change in cases.items():
        with pytest.raises(Lot44FatalError) as err:
            validate_result("f" * 64, SimpleNamespace(**{**good.__dict__, **change}), 16)
        assert err.value.code == code


def smoke_ctx(tmp_path, n=48) -> Context:
    require_inputs()
    return make_context(tmp_path / "exp", max_positions=n)


def test_prepare_is_idempotent_and_detects_modifications(tmp_path):
    ctx = smoke_ctx(tmp_path)
    cands, ref = smoke_inputs(ORIGINAL_POSITIONS, 48)
    first = prepare(ctx, candidates=cands, reference=ref)
    assert first["overlap"] == 0 and sum(first["sizes"].values()) == 46
    lock = read_json(ctx.out / "test_lock.json")
    assert lock["locked"] and not lock["evaluated"] and lock["split_method"] and lock["code_commit"]
    prepare(ctx, candidates=cands, reference=ref)
    assert read_json(ctx.out / "test_lock.json") == lock
    with pytest.raises(Lot44FatalError) as err:
        prepare(dataclasses.replace(ctx, max_positions=40), candidates=cands, reference=ref)
    assert err.value.code == "CORPUS_MODIFIED"
    # Corpus identique (46 disponibles <= plafond) mais graine de split differente.
    with pytest.raises(Lot44FatalError) as err:
        prepare(dataclasses.replace(ctx, seed=ctx.seed + 1), candidates=cands, reference=ref)
    assert err.value.code == "SPLIT_MODIFIED"
    label = read_json(ctx.out / "label_definition.json")
    write_json(ctx.out / "label_definition.json", {**label, "js_threshold": 0.04, "sha256": "tampered"})
    with pytest.raises(Lot44FatalError) as err:
        prepare(ctx, candidates=cands, reference=ref)
    assert err.value.code == "LABEL_DEFINITION_CHANGED"


def test_test_reference_search_is_blocked_before_freeze(tmp_path):
    ctx = smoke_ctx(tmp_path)
    cands, ref = smoke_inputs(ORIGINAL_POSITIONS, 48)
    prepare(ctx, candidates=cands, reference=ref)
    with pytest.raises(Lot44FatalError) as err:
        search_step(ctx, "test", "ref", torch.device("cpu"))
    assert err.value.code == "TEST_LEAKAGE"
    assert not (ctx.out / "search/test/256").exists()


def test_full_pipeline_freeze_evaluate_once_and_finalize(tmp_path):
    ctx = smoke_ctx(tmp_path, n=96)
    cands, ref = smoke_inputs(ORIGINAL_POSITIONS, 96)
    prepare(ctx, candidates=cands, reference=ref)
    result = run(ctx)
    assert result["status"] == "COMPLETE" and result["develop"]["status"] == "FROZEN"
    frozen = read_json(ctx.out / "router_frozen_manifest.json")
    assert frozen["test_reference_search_existed_at_freeze"] is False
    for key in ("features", "transformations", "models", "label_definition_sha256", "checkpoint_fingerprints", "engine_fingerprint", "mcts_identity_fingerprints", "code_commit", "dataset_fingerprint", "split_partition_sha256"):
        assert key in frozen
    assert "test" not in read_json(ctx.out / "labels_development.json")["rows"]
    assert evaluate(ctx)["status"] == "ALREADY_EVALUATED"
    lock = read_json(ctx.out / "test_lock.json")
    assert lock["evaluated"] and lock["evaluation_attempts"] == 1
    decision = finalize(ctx)
    for key in DECISION_KEYS:
        assert key in decision
    assert decision["EARLY_STOP_8192_ALLOWED"] == "NO" and decision["MCTS_RESUME_CURRENTLY_SUPPORTED"] == "NO"
    assert decision["CLASSIFIER_RESUME_COST"]["semantics"] == "COUNTERFACTUAL_ESTIMATE"
    assert read_json(ctx.out / "leakage_audit.json")["LEAKAGE_DETECTED"] == "NO"
    (ctx.out / "oos_metrics.json").unlink()
    with pytest.raises(Lot44FatalError) as err:
        evaluate(ctx)
    assert err.value.code == "TEST_LEAKAGE"


def test_smoke_end_to_end(tmp_path):
    ctx = smoke_ctx(tmp_path)
    report = run_smoke(ctx, tmp_path / "lot44_smoke")
    assert report["SMOKE_TEST"] == "PASS"
    assert report["steps"]["idempotent_rerun"]["recomputed_positions"] == 0
    assert report["steps"]["bundle_reverified"]["bundle_valid"]
    assert len(report["steps"]["classifier"]["fitted_models"]) == 3
    assert (tmp_path / "lot44_smoke_results.tar.gz").is_file()


def test_preflight_passes_locally_on_cpu(tmp_path):
    ctx = smoke_ctx(tmp_path)
    report = run_preflight(dataclasses.replace(ctx, smoke=False), require_lot_inputs=False)
    assert report["PREFLIGHT_STATUS"] == "PASS", report["critical_failures"]
    for name in ("PYTHON_IMPORTS", "PROJECT_IMPORTS", "CUDA", "DRIVE", "CHECKPOINT", "MODEL_FORWARD", "ENGINE", "MCTS16", "MCTS64", "BATCHED_MCTS", "ARTIFACT_WRITE_READ", "MANIFEST", "CHECKSUM", "RESUME", "FINALIZE", "EXPORT"):
        assert name in report["checks"]
    assert (ctx.out / "lot44_api_audit.json").is_file() and not (ctx.out / ".preflight_tmp").exists()


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CUDA-unavailable path")
def test_preflight_fails_cleanly_when_cuda_requested_but_absent(tmp_path):
    ctx = dataclasses.replace(smoke_ctx(tmp_path), device_name="cuda")
    report = run_preflight(ctx, require_lot_inputs=False)
    assert report["PREFLIGHT_STATUS"] == "FAIL" and report["SCIENTIFIC_RUN_ALLOWED"] == "NO"
    assert report["checks"]["CUDA"]["details"]["CUDA_AVAILABLE"] == "NO"
    assert "CUDA" in report["critical_failures"]
