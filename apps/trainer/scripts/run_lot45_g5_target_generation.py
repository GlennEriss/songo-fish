#!/usr/bin/env python3
"""Lot 45 : generation des cibles G5 par MCTS32768 uniforme (G5_DEEP_AUTONOMOUS_REANALYSIS_V1).

Stages :
  prepare-inputs  LOCAL : audit des sources, selection ordonnee, bundle inputs
  selftest        pytest Lot44 + Lot45 -> test_report.json
  preflight       verifications reelles + RUN_LOCK -> preflight_report.json
  smoke           pipeline complet a budget reduit (format final)
  prepare         validation Lot44, plan de shards (--target-positions), compute_plan
  pilot           premiers shards du plan (protocole final, reutilisables)
  plan            affiche compute_plan et l'avancement
  generate        calcul MCTS32768 sous verrou mono-ecrivain (exige --confirm ; reprenable)
  status          avancement, verrou, heartbeat
  finalize        validation, dataset, audits, decision
  export          lot45_results.tar.gz verifie

Aucun entrainement (optimizer/backward/checkpoint G5 = NONE), aucun label
Minimax/teacher historique, aucun routeur, aucune position a 65536.
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
from lot44.paths import resolve_drive_root  # noqa: E402
from lot45.config import DEFAULT_CONCURRENCY, DEFAULT_SEED, EXPERIMENT_DIR_NAME, PILOT_SHARDS, PROBE_POSITIONS_FILE, SELECTION_FILE  # noqa: E402

STAGES = ("prepare-inputs", "selftest", "preflight", "smoke", "prepare", "pilot", "plan", "generate", "status", "finalize", "export")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--drive-root", help="Google Drive root containing songo-ai/ (default: SONGO_DRIVE_ROOT, /content/drive/MyDrive, Drive desktop)")
    p.add_argument("--output", type=Path, help="experiment directory (default: <drive>/songo-ai/experiments/lot45_g5_uniform_mcts32768_targets)")
    p.add_argument("--lot44", type=Path, help="Lot44 experiment directory (default: <drive>/songo-ai/experiments/lot44_router_out_of_sample_validation)")
    p.add_argument("--exports", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--selection", type=Path, default=REPO / SELECTION_FILE)
    p.add_argument("--probe-positions", type=Path, default=REPO / PROBE_POSITIONS_FILE)
    p.add_argument("--target-positions", type=int, help="prepare: number of positions to reanalyze (prefix of the ordered selection)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--max-shards", type=int, help="generate: stop after N newly computed shards")
    p.add_argument("--pilot-shards", type=int, default=PILOT_SHARDS)
    p.add_argument("--confirm", action="store_true", help="generate: non-interactive confirmation for the long compute")
    p.add_argument("--takeover-stale-lock", action="store_true", help="take over a lock whose heartbeat expired (logged in run_lock_history.json)")
    p.add_argument("--skip-inputs", action="store_true", help="preflight: do not require Lot45 selection/Lot44 inputs (engineering check)")
    p.add_argument("--sync-to-drive", action="store_true", help="prepare-inputs: copy the input bundle to <drive>/songo-ai/inputs")
    return p


def make_context(a: argparse.Namespace):
    from lot45.generation import Context

    root, method = resolve_drive_root(a.drive_root)
    experiments = (root / "songo-ai/experiments") if root else REPO / "data/experiments"
    out = (a.output or experiments / EXPERIMENT_DIR_NAME).expanduser().resolve()
    lot44 = (a.lot44 or experiments / "lot44_router_out_of_sample_validation").expanduser()
    exports = (a.exports or (root / "songo-ai/exports" if root else REPO / "data/exports")).expanduser()
    ctx = Context(out=out, seed=a.seed, device_name=a.device, selection_file=a.selection.resolve(), original_positions=a.probe_positions.resolve(), lot44=lot44 if lot44.is_dir() else None, concurrency=a.concurrency, target_positions=a.target_positions, max_shards=a.max_shards, takeover_stale_lock=a.takeover_stale_lock)
    return ctx, {"drive_root": str(root) if root else None, "drive_method": method, "output": str(out), "lot44": str(lot44), "exports": str(exports)}, exports


def require_gates(ctx, *, pilot_needed: bool) -> None:
    from run_srn_colab_benchmark import git_commit

    preflight = ctx.out / "preflight_report.json"
    if not preflight.is_file() or read_json(preflight).get("PREFLIGHT_STATUS") != "PASS":
        raise Lot44FatalError("PREFLIGHT_FAILED", "requires preflight_report.json with PREFLIGHT_STATUS = PASS")
    if read_json(preflight).get("code_commit") != git_commit():
        raise Lot44FatalError("PREFLIGHT_STALE", "code changed since preflight; rerun --stage preflight")
    for name, key in (("smoke_test_report.json", "SMOKE_TEST"),) + ((("pilot_report.json", "PILOT"),) if pilot_needed else ()):
        path = ctx.out / name
        if not path.is_file() or read_json(path).get(key) != "PASS":
            raise Lot44FatalError("GATE_FAILED", f"requires {name} with {key} = PASS")


def overview(ctx) -> dict:
    from lot45.generation import shard_complete

    manifest = read_json(ctx.out / "shard_manifest.json")
    done = sum(shard_complete(ctx, s) for s in manifest["shards"])
    plan = read_json(ctx.out / "compute_plan.json")
    lock = ctx.out / "run_lock.json"
    return {"target_positions": manifest["target_positions"], "shards_complete": done, "shards_total": len(manifest["shards"]), "positions_complete_estimate": done * manifest["shard_size"], "compute_plan": plan["candidates"], "t4_simulations_per_second": plan["t4_simulations_per_second"], "pilot_simulations_per_second": plan["pilot_simulations_per_second"], "run_lock": read_json(lock) if lock.is_file() else None, "heartbeat": read_json(ctx.out / "heartbeat.json") if (ctx.out / "heartbeat.json").is_file() else None}


def dispatch(a: argparse.Namespace, ctx, paths: dict, exports: Path) -> dict:
    if a.stage == "prepare-inputs":
        from lot45.inputs import build_input_bundle, build_selection

        if ctx.lot44 is None:
            raise Lot44FatalError("ARTIFACT_MISSING", "Lot44 experiment directory required to exclude its diagnostic positions")
        selection = build_selection(REPO, ctx.lot44, seed=ctx.seed)
        return {"selection": selection, "bundle": build_input_bundle(REPO, REPO / "data/colab_bridge", sync_to_drive=a.sync_to_drive, drive_root=a.drive_root)}
    if a.stage == "selftest":
        from lot44.selftest import run_selftest

        report = run_selftest(ctx.out, ("packages/songo_ai/tests/lot44", "packages/songo_ai/tests/lot45"))
        if report["failed"]:
            raise Lot44FatalError("TESTS_FAILED", f"{report['failed']} tests failed; see test_report.json")
        return {k: report[k] for k in ("command", "return_code", "total", "passed", "failed", "skipped")}
    if a.stage == "preflight":
        from lot45.preflight import run_preflight

        report = run_preflight(ctx, require_inputs=not a.skip_inputs, repo=REPO)
        if report["PREFLIGHT_STATUS"] != "PASS":
            raise Lot44FatalError("PREFLIGHT_FAILED", f"critical failures: {report['critical_failures']}")
        return {k: report[k] for k in ("PREFLIGHT_STATUS", "SCIENTIFIC_RUN_ALLOWED", "CUDA_AVAILABLE", "device", "summary")}
    if a.stage == "smoke":
        from lot45.smoke import run_smoke

        report = run_smoke(ctx)
        return {"SMOKE_TEST": report["SMOKE_TEST"], "seconds": report["seconds"], "finalize": report["steps"]["finalize"]}
    if a.stage == "prepare":
        from lot45.generation import prepare
        from lot45.inputs import install_reports

        install_reports(REPO, ctx.out)
        return prepare(ctx)
    if a.stage == "pilot":
        from lot45.generation import pilot

        require_gates(ctx, pilot_needed=False)
        return pilot(ctx, shards=a.pilot_shards)
    if a.stage == "plan":
        return overview(ctx)
    if a.stage == "generate":
        from lot45.generation import generate

        require_gates(ctx, pilot_needed=True)
        info = overview(ctx)
        if not a.confirm:
            return {"status": "CONFIRMATION_REQUIRED", "plan": info, "hint": "re-run with --confirm to start/resume the generation"}
        return generate(ctx, owner="generate", max_shards=a.max_shards)
    if a.stage == "status":
        return overview(ctx)
    if a.stage == "finalize":
        from lot45.finalize import finalize

        return finalize(ctx)
    if a.stage == "export":
        from lot45.finalize import export

        return export(ctx, a.bundle or exports / "lot45_results.tar.gz")
    raise ValueError(a.stage)


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    os.chdir(REPO)
    ctx, paths, exports = make_context(a)
    ctx.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    entry = {"stage": a.stage, "argv": sys.argv[1:] if argv is None else argv, "started_utc": utc_now(), "paths": paths}
    print(f"[Lot45] stage={a.stage} output={ctx.out} drive={paths['drive_method']}", flush=True)
    try:
        result = dispatch(a, ctx, paths, exports)
    except BaseException as exc:
        ErrorLog(ctx.out / "errors.jsonl").record(stage=a.stage, exc=exc)
        append_execution(ctx.out, {**entry, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "seconds": time.time() - started})
        raise
    append_execution(ctx.out, {**entry, "status": "OK", "seconds": time.time() - started})
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
