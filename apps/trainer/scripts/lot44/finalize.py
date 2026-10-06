"""Finalize (verifications, comptabilite, decision pre-enregistree) et export."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from .accounting import compute_accounting, mcts_resume_audit
from .artifacts import (
    Lot44FatalError,
    build_checksums,
    canonical_hash,
    make_bundle,
    read_checked_json,
    read_csv,
    read_json,
    update_stage_state,
    utc_now,
    verify_checksums,
    write_json,
)
from .config import BUNDLE_NAME, EXPERIMENT_DIR_NAME, LABEL_DEFINITION, PARTITIONS, PRODUCTION_BUDGETS, PROTOCOL, ROLES
from .evaluation import wilson
from .features import COMBINED, FORBIDDEN_COLUMNS, compute_features
from .pipeline import Context, partition_records, role_rows
from .search import load_search_rows, shard_dir

TECHNICAL_ARTIFACTS = ("preflight_report.json", "lot44_api_audit.json", "test_report.json", "smoke_test_report.json", "errors.jsonl", "stage_state.json")
SCIENTIFIC_INPUTS = (
    "lot43_input_validation.json", "independent_corpus_manifest.json", "overlap_audit.json", "split_manifest.json",
    "train_fingerprints.json", "calibration_fingerprints.json", "test_fingerprints.json", "test_lock.json",
    "label_definition.json", "protocol.json", "power_analysis.json", "feature_schema.json", "feature_dataset.csv",
    "labels_development.json", "classifier_training.json", "classifier_calibration.json", "router_frozen_manifest.json",
    "oos_predictions.csv", "oos_metrics.json", "confidence_intervals.json", "false_negatives.json", "router_counterexamples.json",
)
FINAL_OUTPUTS = (
    "reference_32768_manifest.json", "reference_65536_manifest.json", "leakage_audit.json",
    "baseline_uniform_32768.json", "baseline_conservative_router.json", "baseline_uniform_65536.json",
    "restart_accounting.json", "resume_accounting.json", "compute_frontier.json",
    "ablation_instantaneous.json", "ablation_trajectory.json", "ablation_combined.json",
    "decision.json", "report.json", "checksums.json", "experiment_manifest.json",
)


def _bool(value: str) -> bool:
    if value not in ("True", "False"):
        raise Lot44FatalError("ARTIFACT_INCONSISTENT", f"unexpected boolean literal {value!r} in CSV")
    return value == "True"


def _float_or_none(value: str) -> float | None:
    return None if value == "" else float(value)


def load_predictions(out: Path) -> list[dict]:
    rows = read_csv(out / "oos_predictions.csv")
    parsed = []
    for r in rows:
        parsed.append({
            **r,
            "label": int(r["label"]),
            "deeper_regret": _float_or_none(r["deeper_regret"]),
            "high_severity": _bool(r["high_severity"]),
            **{k: _bool(v) for k, v in r.items() if k.startswith("route_") or k == "false_negative_primary" or k == "top1_flip"},
        })
    return parsed


def reference_manifest(ctx: Context, role: str) -> dict:
    budget = ctx.budgets[role]
    out = {"budget": budget, "role": role, "search_identity_fingerprint": ctx.identity(role).fingerprint(), "partitions": {}}
    for partition in PARTITIONS:
        records = partition_records(ctx, partition)
        rows = role_rows(ctx, partition, role, records)
        shards = [read_json(p) for p in sorted(shard_dir(ctx.out, partition, budget).glob("shard_*.json")) if not p.name.endswith(".sha256.json")]
        out["partitions"][partition] = {
            "positions": len(rows),
            "complete": len(rows) == len(records),
            "actual_simulations": int(sum(r["actual_simulations"] for r in rows.values())),
            "shards": len(shards),
            "wall_time_s": float(sum(s["wall_time_s"] for s in shards)),
            "devices": sorted({s["device"] for s in shards}),
        }
    out["complete"] = all(p["complete"] for p in out["partitions"].values())
    out["ground_truth_only"] = role == "ref"
    return out


def leakage_audit(ctx: Context) -> dict:
    overlap = read_json(ctx.out / "overlap_audit.json")
    split = read_json(ctx.out / "split_manifest.json")
    frozen = read_checked_json(ctx.out / "router_frozen_manifest.json")
    lock = read_json(ctx.out / "test_lock.json")
    metrics = read_checked_json(ctx.out / "oos_metrics.json")
    header = (ctx.out / "feature_dataset.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    params = list(inspect.signature(compute_features).parameters)
    checks = {
        "physical_overlap_with_original_256": overlap["OOS_OVERLAP_WITH_ORIGINAL_256"],
        "game_overlap_with_original_256": overlap["OOS_GAME_OVERLAP_WITH_ORIGINAL_256"],
        "same_game_across_partitions": split["leakage"]["same_game_across_partitions"],
        "same_state_across_partitions": split["leakage"]["same_state_across_partitions"],
        "feature_columns_include_label_or_reference": sorted(set(header) & set(FORBIDDEN_COLUMNS)),
        "compute_features_parameters": params,
        "compute_features_receives_reference_search": "ref" in params or any("ref" in p for p in params),
        "feature_names_mention_reference": [n for n in COMBINED if "ref" in n or str(ctx.budgets["ref"]) in n],
        "test_reference_search_existed_at_freeze": frozen["test_reference_search_existed_at_freeze"],
        "frozen_before_evaluation": frozen["frozen_utc"] <= lock.get("evaluated_utc", ""),
        "evaluation_attempts": lock.get("evaluation_attempts"),
        "evaluated_with_frozen_manifest": lock.get("evaluated_with_frozen_manifest_sha256") == canonical_hash(frozen) == metrics["frozen_manifest_sha256"],
        "calibration_partition_only": read_json(ctx.out / "classifier_calibration.json")["partition"] == "calibration",
        "training_partition_only": read_json(ctx.out / "classifier_training.json")["partition"] == "train",
        "label_definition_unchanged": frozen["label_definition_sha256"] == canonical_hash(LABEL_DEFINITION),
        "protocol_unchanged": frozen["protocol_sha256"] == canonical_hash(PROTOCOL),
    }
    clean = (
        checks["physical_overlap_with_original_256"] == 0 and checks["game_overlap_with_original_256"] == 0
        and checks["same_game_across_partitions"] == 0 and checks["same_state_across_partitions"] == 0
        and not checks["feature_columns_include_label_or_reference"] and not checks["compute_features_receives_reference_search"]
        and not checks["feature_names_mention_reference"] and not checks["test_reference_search_existed_at_freeze"]
        and checks["frozen_before_evaluation"] and checks["evaluation_attempts"] == 1 and checks["evaluated_with_frozen_manifest"]
        and checks["calibration_partition_only"] and checks["training_partition_only"]
        and checks["label_definition_unchanged"] and checks["protocol_unchanged"]
    )
    return {**checks, "LEAKAGE_DETECTED": "NO" if clean else "YES", "TEST_SET_MUTATED": "NO", "threshold_tuning_on_test": False}


def _sims(ctx: Context, records: list[dict]) -> dict[str, np.ndarray]:
    return {role: np.asarray([role_rows(ctx, "test", role, records)[r["fingerprint"]]["actual_simulations"] for r in records], dtype=float) for role in ROLES}


def decide(ctx: Context, metrics: dict, ci: dict, acc: dict, leak: dict, resume: dict, predictions: list[dict], sizes: dict) -> dict:
    primary = metrics["primary"]
    minimum = PROTOCOL["min_oos_positives_for_verdict"]
    positives = primary["positives"]
    recall_ci = ci["wilson95"][metrics["primary_router"]]["recall_wilson95"]
    fn_rows = [p for p in predictions if p["false_negative_primary"]]
    high_fn = sum(p["high_severity"] for p in fn_rows)
    if positives < minimum:
        validated = "INCONCLUSIVE"
    elif (primary["recall"] or 0) >= PROTOCOL["router_validated_min_recall"] and (recall_ci[0] or 0) >= PROTOCOL["router_validated_min_recall_ci_low"] and high_fn == 0 and primary["routing_rate"] <= PROTOCOL["router_validated_max_routing_rate"]:
        validated = "YES"
    else:
        validated = "NO"
    diff = ci["bootstrap"]["ap_combined_minus_classic_ci95"]
    if positives < minimum or diff[0] is None:
        trajectory = "INCONCLUSIVE"
    elif diff[0] > 0:
        trajectory = "YES"
    elif diff[1] <= 0:
        trajectory = "NO"
    else:
        trajectory = "INCONCLUSIVE"
    strategies = acc["strategies"]
    router = strategies["CALIBRATED_ROUTER_32768_TO_65536"]
    uniform_ref = strategies["UNIFORM_65536"]["restart_total_simulations"]

    def viable(cost: float) -> str:
        if validated == "INCONCLUSIVE":
            return "INCONCLUSIVE"
        return "YES" if validated == "YES" and cost <= PROTOCOL["viability_cost_ratio"] * uniform_ref else "NO"

    restart_viable = viable(router["restart_total_simulations"])
    resume_viable = viable(router["resume_total_simulations"])
    n = primary["n"]
    high = sum(p["high_severity"] for p in predictions)
    high_ci = wilson(high, n)
    limit = PROTOCOL["uniform_32768_high_severity_rate_limit"]
    uniform_defensible = "YES" if high_ci[1] is not None and high_ci[1] <= limit else "NO" if high_ci[0] is not None and high_ci[0] > limit else "INCONCLUSIVE"
    valid = leak["LEAKAGE_DETECTED"] == "NO"
    if not valid:
        next_action = "MORE_ANALYSIS_REQUIRED"
    elif validated == "YES" and restart_viable == "YES":
        next_action = "LOT45_ROUTED_AUTONOMOUS_TARGET_GENERATION"
    elif validated == "YES" and resume_viable == "YES":
        next_action = "LOT45_MCTS_TREE_RESUME_ENGINEERING"
    else:
        next_action = "LOT45_G5_UNIFORM_32768_TARGET_GENERATION"

    def perf(name: str) -> dict:
        m = metrics["models"][name]
        return {k: m[k] for k in ("pr_auc", "roc_auc", "recall", "precision", "routing_rate", "fn", "brier", "ece")}

    def cost(name: str, kind: str) -> dict:
        s = strategies[name]
        return {"simulations": s[f"{kind}_total_simulations"], "semantics": s[f"{kind}_semantics"]}

    return {
        "SMOKE_RUN_NOT_SCIENTIFIC": ctx.smoke,
        "LOT44_VALID": "YES" if valid else "NO",
        "INDEPENDENT_CORPUS_VALID": "YES" if leak["physical_overlap_with_original_256"] == 0 and leak["game_overlap_with_original_256"] == 0 else "NO",
        "OOS_OVERLAP_WITH_ORIGINAL_256": leak["physical_overlap_with_original_256"],
        "OOS_GAME_OVERLAP_WITH_ORIGINAL_256": leak["game_overlap_with_original_256"],
        "TRAIN_SIZE": sizes["train"], "CALIBRATION_SIZE": sizes["calibration"], "TEST_SIZE": sizes["test"],
        "OOS_POSITIVE_COUNT": positives, "OOS_NEGATIVE_COUNT": n - positives,
        "ROUTER_VALIDATED": validated,
        "OOS_RECALL": primary["recall"], "OOS_RECALL_CI95": recall_ci,
        "OOS_PRECISION": primary["precision"], "OOS_PR_AUC": primary["pr_auc"], "OOS_ROC_AUC": primary["roc_auc"],
        "OOS_FALSE_NEGATIVES": primary["fn"], "OOS_HIGH_SEVERITY_FALSE_NEGATIVES": high_fn,
        "WORST_FALSE_NEGATIVE_REGRET": max((p["deeper_regret"] for p in fn_rows if p["deeper_regret"] is not None), default=None),
        "OOS_65536_ROUTING_RATE": primary["routing_rate"],
        "CLASSIC_CONFIDENCE_ONLY_PERFORMANCE": perf("classic"),
        "TRAJECTORY_FEATURES_PERFORMANCE": perf("trajectory"),
        "COMBINED_FEATURES_PERFORMANCE": perf("combined"),
        "CONSERVATIVE_ROUTER_PERFORMANCE": perf("conservative_router"),
        "TRAJECTORY_SIGNALS_ADD_VALUE": trajectory,
        "UNIFORM_32768_TOTAL_COST": cost("UNIFORM_32768", "restart"),
        "CONSERVATIVE_RESTART_COST": cost("CONSERVATIVE_ROUTER", "restart"),
        "CLASSIFIER_RESTART_COST": cost("CALIBRATED_ROUTER_32768_TO_65536", "restart"),
        "CONSERVATIVE_RESUME_COST": cost("CONSERVATIVE_ROUTER", "resume"),
        "CLASSIFIER_RESUME_COST": cost("CALIBRATED_ROUTER_32768_TO_65536", "resume"),
        "UNIFORM_65536_TOTAL_COST": cost("UNIFORM_65536", "restart"),
        "MCTS_RESUME_CURRENTLY_SUPPORTED": resume["MCTS_RESUME_CURRENTLY_SUPPORTED"],
        "MULTIFIDELITY_RESTART_ECONOMICALLY_VIABLE": restart_viable,
        "MULTIFIDELITY_RESUME_ECONOMICALLY_VIABLE": resume_viable,
        "UNIFORM_32768_DEFENSIBLE": uniform_defensible,
        "OOS_HIGH_SEVERITY_LATE_BIFURCATIONS": high, "OOS_HIGH_SEVERITY_RATE_CI95": high_ci,
        "EARLY_STOP_8192_ALLOWED": "NO",
        "SRN_TRAINING": "NO", "G5_TRAINING": "NO", "POLICY_UPDATE": "NO", "VALUE_UPDATE": "NO", "MODEL_WEIGHTS_CHANGED": "NO",
        "MINIMAX_LABELS_USED": "NO", "TEACHER_LABELS_USED": "NO", "ROOT_NOISE": "OFF",
        "NEXT_ACTION": next_action,
    }


def finalize(ctx: Context) -> dict:
    required = SCIENTIFIC_INPUTS if ctx.smoke else SCIENTIFIC_INPUTS + TECHNICAL_ARTIFACTS
    if ctx.smoke:
        required = tuple(r for r in required if r != "lot43_input_validation.json")
    missing = [name for name in required if not (ctx.out / name).is_file()]
    if missing:
        raise Lot44FatalError("ARTIFACT_MISSING", f"cannot finalize, missing: {missing}")
    if not ctx.smoke:
        preflight = read_json(ctx.out / "preflight_report.json")
        if preflight.get("PREFLIGHT_STATUS") != "PASS":
            raise Lot44FatalError("PREFLIGHT_FAILED", "preflight_report.json is not PASS")
        if read_json(ctx.out / "test_report.json").get("failed", 1) != 0:
            raise Lot44FatalError("TESTS_FAILED", "test_report.json reports failures")
        if read_json(ctx.out / "smoke_test_report.json").get("SMOKE_TEST") != "PASS":
            raise Lot44FatalError("SMOKE_FAILED", "smoke_test_report.json is not PASS")
    for name in required:
        path = ctx.out / name
        if name.endswith(".json"):
            read_json(path)
        elif name.endswith(".csv"):
            read_csv(path)
    lock = read_json(ctx.out / "test_lock.json")
    if not lock.get("evaluated"):
        raise Lot44FatalError("ARTIFACT_INCONSISTENT", "TEST not evaluated: refusing to present partial results as complete")
    refs = {role: reference_manifest(ctx, role) for role in ("l3", "ref")}
    if not all(r["complete"] for r in refs.values()):
        raise Lot44FatalError("INCOMPLETE_SEARCH", "reference searches incomplete")
    for role in ("l1", "l2"):
        for partition in PARTITIONS:
            records = partition_records(ctx, partition)
            load_search_rows(ctx.out, partition, ctx.budgets[role], ctx.identity(role).fingerprint(), {r["fingerprint"] for r in records})
    write_json(ctx.out / "reference_32768_manifest.json", refs["l3"])
    write_json(ctx.out / "reference_65536_manifest.json", refs["ref"])
    leak = leakage_audit(ctx)
    write_json(ctx.out / "leakage_audit.json", leak)
    metrics = read_checked_json(ctx.out / "oos_metrics.json")
    ci = read_json(ctx.out / "confidence_intervals.json")
    predictions = load_predictions(ctx.out)
    records = partition_records(ctx, "test")
    if [p["fingerprint"] for p in predictions] != [r["fingerprint"] for r in records]:
        raise Lot44FatalError("ARTIFACT_INCONSISTENT", "oos_predictions.csv does not match the TEST partition")
    routes = {"classifier": np.asarray([p[f"route_{metrics['primary_router']}"] for p in predictions], dtype=bool), "conservative": np.asarray([p["route_conservative"] for p in predictions], dtype=bool)}
    acc = compute_accounting(_sims(ctx, records), predictions, routes)
    resume = mcts_resume_audit()
    strategies = acc["strategies"]
    write_json(ctx.out / "restart_accounting.json", {"semantics": "MEASURED actual simulation counts; every budget is an independent search from the root", "strategies": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("resume")} for k, v in strategies.items()}})
    write_json(ctx.out / "resume_accounting.json", {"semantics": "COUNTERFACTUAL_ESTIMATE except UNIFORM strategies (MEASURED): l3 tree continued to ref", "mcts_resume_audit": resume, "caveat": "with a real resume the l1/l2 snapshots would come from the same tree, not from independent searches; feature values could differ", "strategies": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("restart")} for k, v in strategies.items()}})
    write_json(ctx.out / "compute_frontier.json", {"points": [{"strategy": k, "restart": v["restart_total_simulations"], "resume": v["resume_total_simulations"], "resume_semantics": v["resume_semantics"], "recall": v["recall"], "routing_rate": v["routing_rate"], "false_negatives": v["false_negatives"]} for k, v in strategies.items()], "note": "single frozen operating point per router; no threshold sweep on TEST"})
    write_json(ctx.out / "baseline_uniform_32768.json", strategies["UNIFORM_32768"])
    write_json(ctx.out / "baseline_conservative_router.json", {**strategies["CONSERVATIVE_ROUTER"], "metrics": metrics["models"]["conservative_router"], "rule": PROTOCOL["conservative_router"]})
    write_json(ctx.out / "baseline_uniform_65536.json", strategies["UNIFORM_65536"])
    for name, file in (("classic", "ablation_instantaneous.json"), ("trajectory", "ablation_trajectory.json"), ("combined", "ablation_combined.json")):
        write_json(ctx.out / file, {"model": name, "features": read_checked_json(ctx.out / "router_frozen_manifest.json")["features"][name], "test_metrics": metrics["models"][name], "wilson95": ci["wilson95"][name], "bootstrap95": ci["bootstrap"]["models"][name], "pre_registered_as": "primary" if name == PROTOCOL["primary_router"] else "ablation"})
    sizes = read_json(ctx.out / "split_manifest.json")["sizes"]
    decision = decide(ctx, metrics, ci, acc, leak, resume, predictions, sizes)
    write_json(ctx.out / "decision.json", decision)
    write_json(ctx.out / "report.json", {"lot": 44, "smoke": ctx.smoke, "decision": decision, "power_analysis": read_json(ctx.out / "power_analysis.json"), "protocol": PROTOCOL, "label_definition": LABEL_DEFINITION, "lot43_preserved_as_valid_negative": True, "controlled_stop": True, "finalized_utc": utc_now()})
    update_stage_state(ctx.out, "finalize", {"status": "COMPLETE"})
    checksums = build_checksums(ctx.out)
    write_json(ctx.out / "checksums.json", checksums)
    if verify_checksums(ctx.out, checksums):
        raise Lot44FatalError("CHECKSUM_MISMATCH", "checksums changed immediately after writing")
    fps = ctx.fingerprints()
    write_json(ctx.out / "experiment_manifest.json", {"lot": 44, "smoke": ctx.smoke, "code_commit": fps["code_commit"], "engine_fingerprint": fps["engine"], "model_fingerprints": fps["model"], "budgets": ctx.budgets, "production_budgets": ctx.budgets == PRODUCTION_BUDGETS, "seed": ctx.seed, "artifacts": len(checksums), "checksums_sha256": canonical_hash(checksums), "finalized_utc": utc_now()})
    missing_final = [n for n in FINAL_OUTPUTS if not (ctx.out / n).is_file()]
    if missing_final:
        raise Lot44FatalError("ARTIFACT_MISSING", f"finalize did not produce {missing_final}")
    return decision


def export(ctx: Context, bundle: Path | None) -> dict:
    checksums_path = ctx.out / "checksums.json"
    if not checksums_path.is_file() or not (ctx.out / "decision.json").is_file():
        raise Lot44FatalError("ARTIFACT_MISSING", "finalize Lot44 before export")
    bad = verify_checksums(ctx.out, read_json(checksums_path))
    if bad:
        raise Lot44FatalError("CHECKSUM_MISMATCH", f"artifacts changed after finalize: {bad[:5]}")
    target = bundle or ctx.out.parent.parent / "exports" / BUNDLE_NAME
    report = make_bundle(ctx.out, target, EXPERIMENT_DIR_NAME)
    return {"EXPORT": "PASS", **report}
