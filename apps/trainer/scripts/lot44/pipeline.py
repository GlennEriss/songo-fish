"""Orchestration des stages Lot44 : prepare -> run (searches, develop/freeze,
TEST ref, evaluate) -> finalize -> export.

Ordre impose : la recherche de reference (65536) du TEST n'est autorisee
qu'apres ecriture du ``router_frozen_manifest.json`` ; les labels TEST
n'existent donc physiquement pas avant le gel du routeur.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from run_srn_colab_benchmark import engine_fingerprint, git_commit, pool_fingerprints
from run_srn_lot39 import fingerprint_state, load_model

from .artifacts import (
    ErrorLog,
    Lot44FatalError,
    canonical_hash,
    checked_status,
    read_checked_json,
    read_json,
    total_ram_bytes,
    update_stage_state,
    utc_now,
    write_checked_json,
    write_csv,
    write_json,
)
from .classifier import apply_calibrator, choose_threshold, decision_function, fit_calibrator, fit_logistic, out_of_fold_scores
from .config import (
    COUNTEREXAMPLE_FINGERPRINT,
    DEFAULT_CONCURRENCY,
    LABEL_DEFINITION,
    MAX_RAM_FRACTION,
    PARTITIONS,
    PRE_ROUTING_ROLES,
    PRODUCTION_BUDGETS,
    PROTOCOL,
    ROLES,
    SPLIT_FRACTIONS,
)
from .corpus import build_corpus, load_original_reference, state_from_dict
from .evaluation import bootstrap_ci, classification_metrics, proportion_intervals
from .features import COMBINED, FEATURE_SETS, compute_features, feature_schema
from .labels import material_label
from .routers import conservative_route, conservative_thresholds
from .search import SearchIdentity, estimated_peak_bytes, load_search_rows, run_search
from .splits import SPLIT_METHOD, group_split, leakage_report, partition_payload

# Debits T4 mesures (Lot41, concurrence 256) ; 65536 a 64 arbres non mesure.
ASSUMED_T4_SIMS_PER_S = {4096: 4855.0, 8192: 4661.0, 32768: 4022.0, 65536: 3000.0}
LOT42_MATERIAL_COUNT = 3
LOT43_REQUIRE_65536_COUNT = 11
ORIGINAL_POSITION_COUNT = 256


@dataclass
class Context:
    out: Path
    budgets: dict[str, int]
    seed: int
    device_name: str
    max_positions: int
    candidates_file: Path
    original_positions: Path
    lot41: Path | None
    lot42: Path | None
    lot43: Path | None
    smoke: bool = False
    concurrency_override: int | None = None
    max_shards: int | None = None
    allow_high_ram: bool = False
    log: Callable[[str], None] = print
    _fingerprints: dict = field(default_factory=dict)

    @property
    def errors(self) -> ErrorLog:
        return ErrorLog(self.out / "errors.jsonl")

    def concurrency(self, budget: int) -> int:
        return self.concurrency_override or DEFAULT_CONCURRENCY.get(budget, 8)

    def fingerprints(self) -> dict:
        if not self._fingerprints:
            self._fingerprints = {"model": pool_fingerprints(), "engine": engine_fingerprint(), "code_commit": git_commit()}
        return self._fingerprints

    def identity(self, role: str) -> SearchIdentity:
        fps = self.fingerprints()
        return SearchIdentity(budget=self.budgets[role], seed=self.seed, model_fingerprints=fps["model"], engine_fingerprint=fps["engine"])


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise Lot44FatalError("CUDA_UNAVAILABLE", "CUDA_AVAILABLE = NO but --device cuda was requested; use --device cpu or a GPU runtime")
    return torch.device("cuda" if name == "cuda" or (name == "auto" and torch.cuda.is_available()) else "cpu")


# --------------------------------------------------------------------- prepare


def validate_lot43(ctx: Context) -> dict:
    d43 = read_json(ctx.lot43 / "decision.json")
    d42 = read_json(ctx.lot42 / "decision.json")
    early = read_json(ctx.lot43 / "false_early_stability.json").get("rows", [])
    counter = next((r for r in early if r.get("fingerprint") == COUNTEREXAMPLE_FINGERPRINT), None)
    checks = {
        "LOT43_VALID_IS_NO": d43.get("LOT43_VALID") == "NO",
        "LOT43_NO_FUTURE_LEAKAGE": d43.get("FUTURE_INFORMATION_LEAKAGE") == "NO",
        "LOT43_NO_TRAINING": d43.get("TRAINING_PERFORMED") == "NO",
        "LOT43_NEXT_ACTION_IS_LOT44": d43.get("NEXT_ACTION") == "LOT44_ROUTER_OUT_OF_SAMPLE_VALIDATION",
        "LOT43_COUNTEREXAMPLE_PRESENT": counter is not None,
        "LOT43_COUNTEREXAMPLE_HIGH_SEVERITY": bool(counter and counter.get("high_severity_false_stop") and (counter.get("regret_vs_oracle") or 0) > 0.19),
        "LOT42_VALID": d42.get("LOT42_VALID") == "YES",
        "LOT42_ULTRA_HARD_COUNT_IS_3": d42.get("ULTRA_HARD_POSITION_COUNT") == LOT42_MATERIAL_COUNT,
    }
    payload = {"checks": checks, "VALID": "YES" if all(checks.values()) else "NO", "lot43_result_preserved_as_valid_negative": True, "counterexample": counter, "lot43_decision": d43, "lot42_decision_excerpt": {k: d42.get(k) for k in ("LOT42_VALID", "ULTRA_HARD_POSITION_COUNT", "HARD_POSITION_COUNT", "MCTS65536_ROLE")}}
    write_json(ctx.out / "lot43_input_validation.json", payload)
    if payload["VALID"] != "YES":
        raise Lot44FatalError("LOT43_INPUT_INVALID", f"failed checks: {[k for k, v in checks.items() if not v]}")
    return payload


def freeze_definitions(ctx: Context) -> None:
    for name, payload in (("label_definition.json", LABEL_DEFINITION), ("protocol.json", PROTOCOL)):
        path = ctx.out / name
        record = {**payload, "sha256": canonical_hash(payload)}
        if path.is_file():
            if read_json(path).get("sha256") != record["sha256"]:
                raise Lot44FatalError("LABEL_DEFINITION_CHANGED", f"{name} differs from the frozen definition in lot44/config.py")
        else:
            write_json(path, {**record, "frozen_utc": utc_now(), "frozen_before_any_reference_label": True})


def load_candidates(ctx: Context) -> list[dict]:
    payload = read_json(ctx.candidates_file)
    rows = payload["rows"]
    if canonical_hash([r["fingerprint"] for r in rows]) != payload["fingerprints_sha256"]:
        raise Lot44FatalError("CORPUS_CORRUPT", f"candidate file fingerprint list does not match its manifest: {ctx.candidates_file}")
    return rows


def binomial_tail(n: int, p: float, k: int) -> float:
    """P(X >= k) pour X ~ Binomial(n, p)."""

    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))) if n >= k else 0.0


def power_analysis(n_total: int, n_test: int) -> dict:
    minimum = PROTOCOL["min_oos_positives_for_verdict"]
    scenarios = {}
    for name, count in (("lot42_material_lower_bound", LOT42_MATERIAL_COUNT), ("lot43_require_65536_rate", LOT43_REQUIRE_65536_COUNT)):
        p = count / ORIGINAL_POSITION_COUNT
        scenarios[name] = {
            "positive_rate": p,
            "expected_test_positives": n_test * p,
            "p_test_positives_at_least_minimum": binomial_tail(n_test, p, minimum),
            "test_size_for_expected_minimum": math.ceil(minimum / p),
            "total_corpus_for_expected_minimum": math.ceil(minimum / p / SPLIT_FRACTIONS["test"]),
        }
    return {
        "corpus_size": n_total,
        "test_size": n_test,
        "min_oos_positives_for_verdict": minimum,
        "scenarios": scenarios,
        "caveat": "Lot42 ran 65536 only on the 91 hard positions: 3/256 is a lower bound of the material late-bifurcation rate; 11/256 is the Lot43 oracle rate of positions needing 65536 (looser criterion).",
        "computed_before_any_reference_search": True,
    }


def workload(ctx: Context, sizes: dict[str, int]) -> dict:
    n = sum(sizes.values())
    steps = []
    for partition in PARTITIONS:
        for role in ROLES:
            budget = ctx.budgets[role]
            sims = sizes[partition] * budget
            steps.append({"partition": partition, "role": role, "budget": budget, "positions": sizes[partition], "simulations": sims, "concurrency": ctx.concurrency(budget), "estimated_peak_ram_gb": estimated_peak_bytes(budget, ctx.concurrency(budget)) / 1e9, "estimated_t4_hours": sims / ASSUMED_T4_SIMS_PER_S.get(budget, 4000.0) / 3600})
    return {"positions": n, "budgets": ctx.budgets, "total_simulations": sum(s["simulations"] for s in steps), "estimated_t4_hours": sum(s["estimated_t4_hours"] for s in steps), "throughput_assumption": ASSUMED_T4_SIMS_PER_S, "steps": steps}


def prepare(ctx: Context, *, candidates: list[dict] | None = None, reference: dict | None = None) -> dict:
    ctx.out.mkdir(parents=True, exist_ok=True)
    (ctx.out / "errors.jsonl").touch()
    lot43 = validate_lot43(ctx) if not ctx.smoke else {"VALID": "SKIPPED_SMOKE"}
    freeze_definitions(ctx)
    candidates = candidates if candidates is not None else load_candidates(ctx)
    reference = reference if reference is not None else load_original_reference(ctx.original_positions, ctx.lot41, ctx.lot42)
    corpus = build_corpus(candidates, reference, max_positions=ctx.max_positions, seed=ctx.seed)
    corpus_fp = canonical_hash([r["fingerprint"] for r in corpus["rows"]])
    manifest_path = ctx.out / "independent_corpus_manifest.json"
    if checked_status(manifest_path) == "VALID":
        if read_json(manifest_path)["fingerprint_sha256"] != corpus_fp:
            raise Lot44FatalError("CORPUS_MODIFIED", "recomputed corpus differs from the existing independent_corpus_manifest.json")
    else:
        write_checked_json(manifest_path, {
            "count": len(corpus["rows"]), "available_after_exclusion": corpus["available"], "compute_cap": ctx.max_positions, "seed": ctx.seed,
            "source": str(ctx.candidates_file) if not ctx.smoke else "SMOKE_ORIGINAL_POSITIONS", "source_sha256": canonical_hash([r["fingerprint"] for r in candidates]),
            "positions_per_game": 1, "selection": "seeded hash order over eligible candidates (non-terminal, >= 2 legal actions)",
            "teacher_labels_used": False, "minimax_labels_used": False, "identity_key": corpus["overlap"]["identity_key"],
            "fingerprint_sha256": corpus_fp, "rows": corpus["rows"],
        })
        write_json(ctx.out / "overlap_audit.json", corpus["overlap"])
    rows = read_checked_json(manifest_path)["rows"]
    parts = group_split(rows, seed=ctx.seed)
    leak = leakage_report(parts)
    split_path = ctx.out / "split_manifest.json"
    split = {"method": SPLIT_METHOD, "seed": ctx.seed, "fractions": SPLIT_FRACTIONS, "sizes": {k: len(v) for k, v in parts.items()}, "partition_sha256": {k: partition_payload(k, v)["sha256"] for k, v in parts.items()}, "leakage": leak, "dataset_fingerprint": corpus_fp}
    if checked_status(split_path) == "VALID":
        if read_json(split_path)["partition_sha256"] != split["partition_sha256"]:
            raise Lot44FatalError("SPLIT_MODIFIED", "recomputed split differs from split_manifest.json")
    else:
        if any(leak.values()):
            raise Lot44FatalError("TEST_LEAKAGE", f"split leakage: {leak}")
        for name in PARTITIONS:
            write_checked_json(ctx.out / f"{name}_fingerprints.json", partition_payload(name, parts[name]))
        write_checked_json(split_path, {**split, "created_utc": utc_now()})
    lock_path = ctx.out / "test_lock.json"
    if not lock_path.is_file():
        write_json(lock_path, {"locked": True, "evaluated": False, "evaluation_attempts": 0, "test_sha256": split["partition_sha256"]["test"], "test_fingerprints_file": "test_fingerprints.json", "dataset_fingerprint": corpus_fp, "split_seed": ctx.seed, "split_method": SPLIT_METHOD, "created_utc": utc_now(), "code_commit": ctx.fingerprints()["code_commit"], "TEST_SET_MUTATED": "NO"})
    elif read_json(lock_path)["test_sha256"] != split["partition_sha256"]["test"]:
        raise Lot44FatalError("SPLIT_MODIFIED", "test_lock.json does not match the TEST partition")
    sizes = split["sizes"]
    write_json(ctx.out / "power_analysis.json", power_analysis(sum(sizes.values()), sizes["test"]))
    plan = workload(ctx, sizes)
    write_json(ctx.out / "workload_plan.json", plan)
    update_stage_state(ctx.out, "prepare", {"status": "COMPLETE", "sizes": sizes})
    return {"lot43": lot43["VALID"], "sizes": sizes, "overlap": read_json(ctx.out / "overlap_audit.json")["OOS_OVERLAP_WITH_ORIGINAL_256"], "workload": {k: plan[k] for k in ("positions", "total_simulations", "estimated_t4_hours")}}


def partition_records(ctx: Context, name: str) -> list[dict]:
    corpus = {r["fingerprint"]: r for r in read_checked_json(ctx.out / "independent_corpus_manifest.json")["rows"]}
    payload = read_checked_json(ctx.out / f"{name}_fingerprints.json")
    if read_checked_json(ctx.out / "split_manifest.json")["partition_sha256"][name] != payload["sha256"]:
        raise Lot44FatalError("SPLIT_MODIFIED", f"{name}_fingerprints.json does not match split_manifest.json")
    return [corpus[fp] for fp in payload["fingerprints"]]


# ------------------------------------------------------------------------ run


def frozen_manifest_valid(ctx: Context) -> bool:
    return checked_status(ctx.out / "router_frozen_manifest.json") == "VALID"


def search_step(ctx: Context, partition: str, role: str, device: torch.device) -> dict:
    if partition == "test" and role == "ref" and not frozen_manifest_valid(ctx):
        raise Lot44FatalError("TEST_LEAKAGE", "TEST reference search requested before router_frozen_manifest.json")
    budget = ctx.budgets[role]
    concurrency = ctx.concurrency(budget)
    ram = total_ram_bytes()
    if ram and estimated_peak_bytes(budget, concurrency) > MAX_RAM_FRACTION * ram and not ctx.allow_high_ram:
        raise Lot44FatalError("RAM_BUDGET", f"budget {budget} x concurrency {concurrency} needs ~{estimated_peak_bytes(budget, concurrency) / 1e9:.1f} GB > {MAX_RAM_FRACTION:.0%} of {ram / 1e9:.1f} GB; lower --concurrency")
    loader = lambda: load_model(device)  # noqa: E731
    return run_search(ctx.out, partition=partition, records=partition_records(ctx, partition), identity=ctx.identity(role), concurrency=concurrency, model_loader=loader, device=device, errors=ctx.errors, max_shards=ctx.max_shards, log=ctx.log)


def run(ctx: Context) -> dict:
    device = choose_device(ctx.device_name)
    summary: dict = {"device": str(device), "steps": []}
    pre_freeze = [(p, r) for p in ("train", "calibration") for r in ROLES] + [("test", r) for r in PRE_ROUTING_ROLES]
    for partition, role in pre_freeze:
        result = search_step(ctx, partition, role, device)
        summary["steps"].append({k: result[k] for k in ("partition", "budget", "status", "computed", "already_complete")})
        if result["status"] != "COMPLETE":
            return {**summary, "status": "PARTIAL"}
    summary["develop"] = develop(ctx)
    result = search_step(ctx, "test", "ref", device)
    summary["steps"].append({k: result[k] for k in ("partition", "budget", "status", "computed", "already_complete")})
    if result["status"] != "COMPLETE":
        return {**summary, "status": "PARTIAL"}
    summary["evaluate"] = evaluate(ctx)
    return {**summary, "status": "COMPLETE"}


def role_rows(ctx: Context, partition: str, role: str, records: list[dict]) -> dict[str, dict]:
    return load_search_rows(ctx.out, partition, ctx.budgets[role], ctx.identity(role).fingerprint(), {r["fingerprint"] for r in records})


def partition_features(ctx: Context, partition: str) -> tuple[list[dict], list[dict]]:
    records = partition_records(ctx, partition)
    rows = {role: role_rows(ctx, partition, role, records) for role in PRE_ROUTING_ROLES}
    feats = []
    for record in records:
        fp = record["fingerprint"]
        values = compute_features(rows["l1"][fp], rows["l2"][fp], rows["l3"][fp], record["state"])
        feats.append({"fingerprint": fp, "partition": partition, "game_id": record["game_id"], **values})
    return records, feats


def partition_labels(ctx: Context, partition: str, records: list[dict]) -> list[dict]:
    l3 = role_rows(ctx, partition, "l3", records)
    ref = role_rows(ctx, partition, "ref", records)
    return [{"fingerprint": r["fingerprint"], **material_label(l3[r["fingerprint"]], ref[r["fingerprint"]])} for r in records]


def matrix(feats: list[dict], names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[f[n] for n in names] for f in feats], dtype=float).reshape(len(feats), len(names))


def develop(ctx: Context) -> dict:
    """Features, entrainement (TRAIN), calibration (CALIBRATION), gel du routeur."""

    manifest_path = ctx.out / "router_frozen_manifest.json"
    if frozen_manifest_valid(ctx):
        ctx.log("[Lot44][develop] SKIP: router already frozen")
        return {"status": "ALREADY_FROZEN"}
    if read_json(ctx.out / "test_lock.json").get("evaluated"):
        raise Lot44FatalError("TEST_LEAKAGE", "TEST already evaluated: the router cannot be refitted")
    freeze_definitions(ctx)
    data = {p: partition_features(ctx, p) for p in PARTITIONS}
    all_feats = [f for p in PARTITIONS for f in data[p][1]]
    write_json(ctx.out / "feature_schema.json", feature_schema(ctx.budgets))
    write_csv(ctx.out / "feature_dataset.csv", all_feats, ["fingerprint", "partition", "game_id", *COMBINED])
    labels = {p: partition_labels(ctx, p, data[p][0]) for p in ("train", "calibration")}
    write_json(ctx.out / "labels_development.json", {"partitions": ["train", "calibration"], "test_labels_present": False, "rows": {p: labels[p] for p in labels}})
    y = {p: np.asarray([r["label"] for r in labels[p]], dtype=float) for p in labels}
    fps = {p: [f["fingerprint"] for f in data[p][1]] for p in labels}
    training, calibration, models = {}, {}, {}
    for name, names in FEATURE_SETS.items():
        x_train = matrix(data["train"][1], names)
        x_cal = matrix(data["calibration"][1], names)
        model = fit_logistic(x_train, y["train"], l2=PROTOCOL["l2_strength"], balanced=True, names=list(names))
        oof, folds = out_of_fold_scores(x_train, y["train"], fps["train"], names=list(names), seed=ctx.seed)
        raw_cal = decision_function(model, x_cal)
        calibrator = fit_calibrator(raw_cal, y["calibration"])
        cal_probs = apply_calibrator(calibrator, raw_cal)
        dev_scores = np.concatenate([apply_calibrator(calibrator, oof) if oof is not None else np.zeros(0), cal_probs])
        dev_y = np.concatenate([y["train"] if oof is not None else np.zeros(0), y["calibration"]])
        threshold = choose_threshold(dev_scores, dev_y)
        training[name] = {"model": model, "oof_folds": folds, "oof_available": oof is not None, "train_metrics_in_sample": classification_metrics(y["train"], apply_calibrator(calibrator, decision_function(model, x_train)), threshold["threshold"])}
        calibration[name] = {"calibrator": calibrator, "threshold": threshold, "calibration_metrics": classification_metrics(y["calibration"], cal_probs, threshold["threshold"]), "development_positives": int(dev_y.sum()), "development_size": int(len(dev_y))}
        models[name] = {"model": model, "calibrator": calibrator, "threshold": threshold["threshold"]}
    write_json(ctx.out / "classifier_training.json", {"partition": "train", "size": int(len(y["train"])), "positives": int(y["train"].sum()), "models": training, "test_accessed": False})
    write_json(ctx.out / "classifier_calibration.json", {"partition": "calibration", "size": int(len(y["calibration"])), "positives": int(y["calibration"].sum()), "models": calibration, "test_accessed": False})
    fps_all = ctx.fingerprints()
    lock = read_json(ctx.out / "test_lock.json")
    manifest = {
        "frozen": True,
        "frozen_utc": utc_now(),
        "primary_router": PROTOCOL["primary_router"],
        "models": models,
        "features": {k: list(v) for k, v in FEATURE_SETS.items()},
        "transformations": "per-feature standardization (train mean/std), logistic decision function, calibrator, threshold on calibrated score",
        "feature_schema_sha256": canonical_hash(feature_schema(ctx.budgets)),
        "label_definition_sha256": canonical_hash(LABEL_DEFINITION),
        "protocol_sha256": canonical_hash(PROTOCOL),
        "conservative_router": conservative_thresholds(),
        "checkpoint_fingerprints": fps_all["model"],
        "engine_fingerprint": fps_all["engine"],
        "mcts_identity_fingerprints": {role: ctx.identity(role).fingerprint() for role in ROLES},
        "budgets": ctx.budgets,
        "code_commit": fps_all["code_commit"],
        "dataset_fingerprint": read_json(ctx.out / "split_manifest.json")["dataset_fingerprint"],
        "split_partition_sha256": read_json(ctx.out / "split_manifest.json")["partition_sha256"],
        "test_sha256": lock["test_sha256"],
        "test_reference_search_existed_at_freeze": any((ctx.out / "search" / "test" / str(ctx.budgets["ref"])).glob("shard_*.json")),
    }
    write_checked_json(manifest_path, manifest)
    update_stage_state(ctx.out, "develop", {"status": "FROZEN"})
    return {"status": "FROZEN", "train_positives": int(y["train"].sum()), "calibration_positives": int(y["calibration"].sum()), "thresholds": {k: v["threshold"] for k, v in models.items()}}


# ------------------------------------------------------------------- evaluate


def score_models(frozen: dict, feats: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for name, item in frozen["models"].items():
        raw = decision_function(item["model"], matrix(feats, tuple(item["model"]["features"])))
        prob = apply_calibrator(item["calibrator"], raw)
        out[name] = {"probability": prob, "route": prob >= item["threshold"]}
    return out


def historical_counterexample(ctx: Context, frozen: dict) -> dict:
    """fb248b86 : diagnostic seulement, jamais compte comme preuve OOS."""

    base = {"fingerprint": COUNTEREXAMPLE_FINGERPRINT, "counts_as_oos_evidence": False}
    if ctx.smoke or ctx.budgets != PRODUCTION_BUDGETS or ctx.lot41 is None:
        return {**base, "status": "NOT_APPLICABLE_SMOKE"}
    rows = {}
    for role in PRE_ROUTING_ROLES:
        payload = read_json(ctx.lot41 / "search" / f"budget_{ctx.budgets[role]}.json")
        rows[role] = next(r for r in payload["rows"] if r["state_fingerprint"] == COUNTEREXAMPLE_FINGERPRINT)
    states = read_json(ctx.original_positions)["states"]
    state = next(s for s in states if fingerprint_state(state_from_dict(s)) == COUNTEREXAMPLE_FINGERPRINT)
    values = compute_features(rows["l1"], rows["l2"], rows["l3"], state)
    scored = score_models(frozen, [values])
    ref = next((r for r in read_json(ctx.lot42 / "search_65536_results.json")["rows"] if r["state_fingerprint"] == COUNTEREXAMPLE_FINGERPRINT), None)
    return {
        **base,
        "status": "EVALUATED",
        "source": "Lot41 4096/8192/32768 rows (different seeds) + Lot42 65536 row",
        "features": values,
        "router_probability": {k: float(v["probability"][0]) for k, v in scored.items()},
        "routed_to_65536": {k: bool(v["route"][0]) for k, v in scored.items()},
        "conservative_routed_to_65536": conservative_route(values),
        "label_32768_65536": material_label(rows["l3"], ref) if ref else None,
        "note": "Lot43 regret 0.1937 was measured 8192 -> 65536; the 32768 -> 65536 label may differ.",
    }


def evaluate(ctx: Context) -> dict:
    lock_path = ctx.out / "test_lock.json"
    lock = read_json(lock_path)
    if lock.get("evaluated"):
        if checked_status(ctx.out / "oos_metrics.json") == "VALID":
            ctx.log("[Lot44][evaluate] SKIP: TEST already evaluated once (no re-evaluation)")
            return {"status": "ALREADY_EVALUATED"}
        raise Lot44FatalError("TEST_LEAKAGE", "test_lock says evaluated but oos_metrics.json is missing/corrupt; refusing a second evaluation")
    frozen = read_checked_json(ctx.out / "router_frozen_manifest.json")
    if not lock["locked"] or lock["test_sha256"] != frozen["test_sha256"]:
        raise Lot44FatalError("SPLIT_MODIFIED", "test lock does not match the frozen router")
    if frozen["label_definition_sha256"] != canonical_hash(LABEL_DEFINITION) or frozen["protocol_sha256"] != canonical_hash(PROTOCOL):
        raise Lot44FatalError("LABEL_DEFINITION_CHANGED", "label/protocol changed after freeze")
    write_json(lock_path, {**lock, "evaluation_attempts": lock.get("evaluation_attempts", 0) + 1, "evaluation_started_utc": utc_now()})
    records, feats = partition_features(ctx, "test")
    labels = partition_labels(ctx, "test", records)
    y = np.asarray([r["label"] for r in labels], dtype=int)
    scored = score_models(frozen, feats)
    conservative = np.asarray([conservative_route(f) for f in feats], dtype=bool)
    primary = frozen["primary_router"]
    metrics = {name: classification_metrics(y, s["probability"], frozen["models"][name]["threshold"], ece_bins=PROTOCOL["ece_bins"]) for name, s in scored.items()}
    metrics["conservative_router"] = classification_metrics(y, conservative.astype(float), 0.5, ece_bins=PROTOCOL["ece_bins"])
    predictions = []
    for i, (record, label) in enumerate(zip(records, labels)):
        row = {"fingerprint": record["fingerprint"], "game_id": record["game_id"], **{k: label[k] for k in ("label", "label_reason", "top1_flip", "js_l3_ref", "deeper_regret", "flip_severity", "high_severity", "action_l3", "action_ref")}}
        for name, s in scored.items():
            row[f"probability_{name}"] = float(s["probability"][i])
            row[f"route_{name}"] = bool(s["route"][i])
        row["route_conservative"] = bool(conservative[i])
        row["false_negative_primary"] = bool(label["label"] and not scored[primary]["route"][i])
        predictions.append(row)
    write_csv(ctx.out / "oos_predictions.csv", predictions)
    boot = bootstrap_ci(y, {k: v["probability"] for k, v in scored.items()}, resamples=PROTOCOL["bootstrap_resamples"], seed=ctx.seed)
    write_checked_json(ctx.out / "oos_metrics.json", {"primary_router": primary, "primary": metrics[primary], "models": metrics, "evaluated_utc": utc_now(), "frozen_manifest_sha256": canonical_hash(frozen)})
    write_json(ctx.out / "confidence_intervals.json", {"primary_router": primary, "wilson95": {name: proportion_intervals(m) for name, m in metrics.items()}, "bootstrap": boot, "zero_false_negative_warning": "zero observed false negatives never proves recall = 1; read the Wilson interval"})
    misses = [p for p in predictions if p["false_negative_primary"]]
    write_json(ctx.out / "false_negatives.json", {"router": primary, "count": len(misses), "high_severity_count": sum(bool(p["high_severity"]) for p in misses), "rows": misses})
    counter = historical_counterexample(ctx, frozen)
    write_json(ctx.out / "router_counterexamples.json", {
        "oos_false_negatives": misses,
        "oos_positives_missed_by_conservative": [p for p in predictions if p["label"] and not p["route_conservative"]],
        "historical_counterexample_fb248b86": counter,
    })
    write_json(lock_path, {**read_json(lock_path), "locked": False, "evaluated": True, "evaluated_utc": utc_now(), "evaluated_with_frozen_manifest_sha256": canonical_hash(frozen), "evaluated_at_commit": ctx.fingerprints()["code_commit"], "TEST_SET_MUTATED": "NO"})
    update_stage_state(ctx.out, "evaluate", {"status": "COMPLETE"})
    return {"status": "EVALUATED", "positives": int(y.sum()), "primary": {k: metrics[primary][k] for k in ("recall", "precision", "routing_rate", "fn")}}
