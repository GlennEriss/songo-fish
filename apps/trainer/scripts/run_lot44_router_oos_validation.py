#!/usr/bin/env python3
"""Lot 44 : validation hors echantillon du routeur MCTS 32768 -> 65536.

Stages (dans l'ordre d'usage) :
  prepare-inputs  LOCAL : candidats OOS (Lot34 pool) + bundle inputs Colab
  selftest        pytest Lot44 -> test_report.json
  preflight       verifications reelles -> preflight_report.json (hard gate)
  smoke           pipeline complet miniature -> smoke_test_report.json
  prepare         corpus OOS, split par partie, test_lock, label/protocole geles
  plan            charge de calcul et analyse de puissance
  run             recherches -> gel du routeur -> 65536 TEST -> evaluation unique
                  (reprenable : relancer la meme commande ; exige --confirm)
  status          etat des stages
  finalize        verifications, comptabilite RESTART/RESUME, decision
  export          lot44_results.tar.gz verifie

Aucun entrainement SRN/G5, aucune modification moteur/MCTS, aucun label
Minimax/teacher. Le seul modele entraine est le petit classifieur de routage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[2]
for entry in (str(SCRIPTS), str(REPO / "packages")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from lot44.artifacts import ErrorLog, Lot44FatalError, append_execution, read_json, utc_now  # noqa: E402
from lot44.config import CANDIDATES_FILE, DEFAULT_MAX_POSITIONS, DEFAULT_SEED, ORIGINAL_POSITIONS_FILE, PRODUCTION_BUDGETS, SOURCE_POOL_DIR  # noqa: E402
from lot44.paths import resolve_paths  # noqa: E402

STAGES = ("prepare-inputs", "selftest", "preflight", "smoke", "prepare", "plan", "run", "status", "finalize", "export")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--drive-root", help="Google Drive root (contains songo-ai/); default: SONGO_DRIVE_ROOT, /content/drive/MyDrive, Drive desktop")
    p.add_argument("--output", type=Path, help="experiment directory (default: <drive>/songo-ai/experiments/lot44_router_out_of_sample_validation)")
    p.add_argument("--lot41", type=Path)
    p.add_argument("--lot42", type=Path)
    p.add_argument("--lot43", type=Path)
    p.add_argument("--exports", type=Path, help="exports directory (default: <drive>/songo-ai/exports)")
    p.add_argument("--bundle", type=Path, help="explicit export bundle path")
    p.add_argument("--candidates", type=Path, default=REPO / CANDIDATES_FILE)
    p.add_argument("--original-positions", type=Path, default=REPO / ORIGINAL_POSITIONS_FILE)
    p.add_argument("--source-pool", type=Path, default=REPO / SOURCE_POOL_DIR, help="prepare-inputs only: Lot34 pool self-play shards")
    p.add_argument("--sync-to-drive", action="store_true", help="prepare-inputs only: copy the input bundle to <drive>/songo-ai/inputs")
    p.add_argument("--max-positions", type=int, default=DEFAULT_MAX_POSITIONS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--concurrency", type=int, help="override per-budget defaults (4096/8192: 256, 32768: 128, 65536: 64)")
    p.add_argument("--max-shards", type=int, help="stop each search step after N new shards (time-sliced sessions)")
    p.add_argument("--allow-high-ram", action="store_true", help="bypass the estimated RAM guard")
    p.add_argument("--confirm", action="store_true", help="run: non-interactive confirmation for the long compute")
    p.add_argument("--skip-lot-inputs", action="store_true", help="preflight only: do not require Lot41-43 artifacts / candidates (local engineering check)")
    return p


def make_context(a: argparse.Namespace):
    from lot44.pipeline import Context

    paths = resolve_paths(drive_root=a.drive_root, output=a.output, lot41=a.lot41, lot42=a.lot42, lot43=a.lot43, exports=a.exports)
    ctx = Context(
        out=paths.output, budgets=dict(PRODUCTION_BUDGETS), seed=a.seed, device_name=a.device, max_positions=a.max_positions,
        candidates_file=a.candidates.resolve(), original_positions=a.original_positions.resolve(),
        lot41=paths.lot41, lot42=paths.lot42, lot43=paths.lot43,
        concurrency_override=a.concurrency, max_shards=a.max_shards, allow_high_ram=a.allow_high_ram,
    )
    return ctx, paths


def require_gates(ctx) -> None:
    from run_srn_colab_benchmark import git_commit

    preflight = ctx.out / "preflight_report.json"
    if not preflight.is_file() or read_json(preflight).get("PREFLIGHT_STATUS") != "PASS":
        raise Lot44FatalError("PREFLIGHT_FAILED", "run requires preflight_report.json with PREFLIGHT_STATUS = PASS")
    if read_json(preflight).get("code_commit") != git_commit():
        raise Lot44FatalError("PREFLIGHT_STALE", "code changed since preflight; rerun --stage preflight")
    smoke = ctx.out / "smoke_test_report.json"
    if not smoke.is_file() or read_json(smoke).get("SMOKE_TEST") != "PASS":
        raise Lot44FatalError("SMOKE_FAILED", "run requires smoke_test_report.json with SMOKE_TEST = PASS")


def dispatch(a: argparse.Namespace, ctx, paths) -> dict:
    if a.stage == "prepare-inputs":
        from lot44.inputs import build_candidates, build_input_bundle

        lot41 = paths.lot41 if (paths.lot41 / "search/budget_32768.json").is_file() else None
        lot42 = paths.lot42 if (paths.lot42 / "search_65536_results.json").is_file() else None
        candidates = build_candidates(a.source_pool, ctx.original_positions, ctx.candidates_file, seed=ctx.seed, lot41=lot41, lot42=lot42)
        bundle = build_input_bundle(REPO, REPO / "data/colab_bridge", sync_to_drive=a.sync_to_drive, drive_root=a.drive_root)
        return {"candidates": candidates, "bundle": bundle}
    if a.stage == "selftest":
        from lot44.selftest import run_selftest

        report = run_selftest(ctx.out)
        if report["failed"]:
            raise Lot44FatalError("TESTS_FAILED", f"{report['failed']} Lot44 tests failed; see test_report.json")
        return {k: report[k] for k in ("command", "return_code", "total", "passed", "failed", "skipped")}
    if a.stage == "preflight":
        from lot44.preflight import run_preflight

        report = run_preflight(ctx, require_lot_inputs=not a.skip_lot_inputs)
        if report["PREFLIGHT_STATUS"] != "PASS":
            raise Lot44FatalError("PREFLIGHT_FAILED", f"critical failures: {report['critical_failures']}")
        return {k: report[k] for k in ("PREFLIGHT_STATUS", "SCIENTIFIC_RUN_ALLOWED", "CUDA_AVAILABLE", "device", "summary")}
    if a.stage == "smoke":
        from lot44.smoke import run_smoke

        report = run_smoke(ctx)
        return {"SMOKE_TEST": report["SMOKE_TEST"], "seconds": report["seconds"], "finalize": report["steps"]["finalize"]}
    if a.stage == "prepare":
        from lot44.pipeline import prepare

        return prepare(ctx)
    if a.stage == "plan":
        return plan(ctx)
    if a.stage == "run":
        from lot44.pipeline import run

        require_gates(ctx)
        overview = plan(ctx)
        if not a.confirm:
            return {"status": "CONFIRMATION_REQUIRED", "plan": overview, "hint": "re-run with --confirm to start/resume the compute"}
        return run(ctx)
    if a.stage == "status":
        return status(ctx)
    if a.stage == "finalize":
        from lot44.finalize import finalize

        return finalize(ctx)
    if a.stage == "export":
        from lot44.finalize import export

        return export(ctx, a.bundle or paths.exports / "lot44_results.tar.gz")
    raise ValueError(a.stage)


def plan(ctx) -> dict:
    from lot44.pipeline import choose_device

    split = read_json(ctx.out / "split_manifest.json")
    workload = read_json(ctx.out / "workload_plan.json")
    power = read_json(ctx.out / "power_analysis.json")
    lock = read_json(ctx.out / "test_lock.json")
    done = {}
    state = ctx.out / "stage_state.json"
    if state.is_file():
        done = {k: v.get("status") for k, v in read_json(state)["stages"].items()}
    overview = {
        "number_of_positions": workload["positions"], "partition_sizes": split["sizes"], "budgets": workload["budgets"],
        "total_simulations": workload["total_simulations"], "estimated_t4_hours": round(workload["estimated_t4_hours"], 2),
        "device": str(choose_device(ctx.device_name)), "checkpoint_fingerprint": ctx.fingerprints()["model"]["policy_fingerprint"][:16] + "/" + ctx.fingerprints()["model"]["value_fingerprint"][:16],
        "output_directory": str(ctx.out), "resume_status": done, "test_locked": lock["locked"], "test_evaluated": lock["evaluated"],
        "power": {k: {"expected_test_positives": round(v["expected_test_positives"], 2), "p_at_least_min": round(v["p_test_positives_at_least_minimum"], 3)} for k, v in power["scenarios"].items()},
    }
    print(json.dumps(overview, indent=2), flush=True)
    return overview


def status(ctx) -> dict:
    state = ctx.out / "stage_state.json"
    heartbeat = ctx.out / "heartbeat.json"
    return {"stages": read_json(state)["stages"] if state.is_file() else {}, "heartbeat": read_json(heartbeat) if heartbeat.is_file() else None}


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    os.chdir(REPO)
    ctx, paths = make_context(a)
    ctx.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    entry = {"stage": a.stage, "argv": sys.argv[1:] if argv is None else argv, "started_utc": utc_now(), "paths": paths.as_dict()}
    print(f"[Lot44] stage={a.stage} output={ctx.out} drive={paths.drive_method}", flush=True)
    try:
        result = dispatch(a, ctx, paths)
    except BaseException as exc:
        ErrorLog(ctx.out / "errors.jsonl").record(stage=a.stage, exc=exc)
        append_execution(ctx.out, {**entry, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "seconds": time.time() - started})
        raise
    append_execution(ctx.out, {**entry, "status": "OK", "seconds": time.time() - started})
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
