"""Preflight obligatoire (sections 13-25). Chaque verification est reellement
executee ; un echec critique donne PREFLIGHT_STATUS = FAIL."""
from __future__ import annotations

import importlib
import platform
import shutil
import sys
import tarfile
import time
import traceback
from typing import Callable

import numpy as np
import torch

from songo_ai.dataset import RawSongoState
from songo_ai.evaluation import model_parameter_fingerprint
from songo_ai.model import SongoGraphBuilder
from songo_ai.search import MCTSConfig, SongoMCTS
from songo_ai.songo.rules import SongoLegacyGame
from run_srn_lot39 import fingerprint_state, load_model
from run_srn_lot41 import LOT40_FLAGS

from .api_audit import build_api_audit
from .artifacts import (
    ErrorLog,
    Lot44FatalError,
    build_checksums,
    checked_status,
    make_bundle,
    read_csv,
    read_json,
    read_jsonl,
    sidecar,
    utc_now,
    verify_bundle,
    write_checked_json,
    write_csv,
    write_json,
    write_jsonl,
)
from colab_drive import sha256
from .config import C_PUCT, POLICY_TEMPERATURE
from .corpus import state_from_dict
from .pipeline import Context, choose_device, load_candidates
from .search import SearchIdentity, run_search, validate_result

ORDER = ("PYTHON_IMPORTS", "PROJECT_IMPORTS", "INPUTS", "CUDA", "DRIVE", "CHECKPOINT", "MODEL_FORWARD", "ENGINE", "MCTS16", "MCTS64", "BATCHED_MCTS", "ARTIFACT_WRITE_READ", "MANIFEST", "CHECKSUM", "RESUME", "FINALIZE", "EXPORT", "MODEL_WEIGHTS_UNCHANGED")


class Preflight:
    def __init__(self, ctx: Context, *, require_lot_inputs: bool) -> None:
        self.ctx = ctx
        self.require_lot_inputs = require_lot_inputs
        self.checks: dict[str, dict] = {}
        self.tmp = ctx.out / ".preflight_tmp"
        self.model = None
        self.device: torch.device | None = None
        self.fingerprint_before: str | None = None

    def check(self, name: str, fn: Callable[[], dict], *, critical: bool = True) -> None:
        started = time.perf_counter()
        try:
            details = fn() or {}
            status = details.pop("_status", "PASS")
            self.checks[name] = {"status": status, "critical": critical, "details": details}
        except Exception as exc:  # enregistre l'echec : le statut global devient FAIL
            self.checks[name] = {"status": "FAIL", "critical": critical, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            ErrorLog(self.ctx.out / "errors.jsonl").record(stage=f"preflight/{name}", exc=exc, fatal=critical)
        self.checks[name]["seconds"] = round(time.perf_counter() - started, 3)
        print(f"[Lot44][preflight] {name:<22} {self.checks[name]['status']}", flush=True)

    def positions(self, count: int) -> list[RawSongoState]:
        states = read_json(self.ctx.original_positions)["states"]
        return [state_from_dict(s) for s in states[:count]]

    # -- checks ----------------------------------------------------------
    def python_imports(self) -> dict:
        stdlib = {name: importlib.import_module(name).__name__ == name for name in ("csv", "hashlib", "json", "tarfile", "xml.etree.ElementTree", "inspect", "resource")}
        if not all(stdlib.values()):
            raise ImportError(f"standard imports failed: {stdlib}")
        return {"stdlib": stdlib, "python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(), "pandas": "NOT_REQUIRED", "scikit_learn": "NOT_REQUIRED (numpy logistic regression)"}

    def project_imports(self) -> dict:
        audit = build_api_audit({"preflight_device": str(self.device)})
        write_json(self.ctx.out / "lot44_api_audit.json", audit)
        return {"apis_resolved": len(audit["apis"]), "audit_file": "lot44_api_audit.json"}

    def inputs(self) -> dict:
        details = {"original_positions": str(self.ctx.original_positions), "original_positions_exists": self.ctx.original_positions.is_file()}
        if not details["original_positions_exists"]:
            raise FileNotFoundError(self.ctx.original_positions)
        if self.require_lot_inputs:
            candidates = load_candidates(self.ctx)
            details["candidates"] = len(candidates)
            for label, root, names in (("lot41", self.ctx.lot41, ("search/budget_4096.json", "search/budget_8192.json", "search/budget_32768.json")), ("lot42", self.ctx.lot42, ("decision.json", "search_65536_results.json")), ("lot43", self.ctx.lot43, ("decision.json", "false_early_stability.json"))):
                missing = [n for n in names if root is None or not (root / n).is_file()]
                if missing:
                    raise FileNotFoundError(f"{label} missing {missing} under {root}")
                details[label] = str(root)
        return details

    def cuda(self) -> dict:
        available = torch.cuda.is_available()
        details = {"CUDA_AVAILABLE": "YES" if available else "NO", "requested_device": self.ctx.device_name}
        if not available:
            details["CPU_CONTINUATION_ALLOWED"] = "NO" if self.ctx.device_name == "cuda" else "YES"
            details["_status"] = "FAIL" if self.ctx.device_name == "cuda" else "UNAVAILABLE"
            return details
        x = torch.arange(16, dtype=torch.float32, device="cuda")
        back = (x * 2).cpu()
        details.update({"gpu_name": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda, "tensor_roundtrip_ok": bool(torch.equal(back, torch.arange(16, dtype=torch.float32) * 2)), "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory})
        if not details["tensor_roundtrip_ok"]:
            raise RuntimeError("CUDA tensor round-trip mismatch")
        return details

    def drive(self) -> dict:
        self.tmp.mkdir(parents=True, exist_ok=True)
        probe = self.tmp / "drive_probe.txt"
        payload = f"lot44 {utc_now()}"
        probe.write_text(payload, encoding="utf-8")
        ok = probe.read_text(encoding="utf-8") == payload
        probe.unlink()
        if not ok or probe.exists():
            raise OSError("write/read/delete probe failed")
        return {"output": str(self.ctx.out), "write_read_delete": True}

    def checkpoint(self) -> dict:
        fps = self.ctx.fingerprints()["model"]
        self.device = choose_device(self.ctx.device_name)
        self.model = load_model(self.device)
        self.fingerprint_before = model_parameter_fingerprint(self.model)
        state = self.model.state_dict()
        return {"model_fingerprints": fps, "parameter_fingerprint": self.fingerprint_before, "state_dict_tensors": len(state), "parameters": int(sum(t.numel() for t in self.model.parameters())), "device": str(self.device), "strict_load": "load_srn_checkpoint calls load_state_dict with the default strict=True: missing/unexpected keys or shape mismatches raise"}

    def model_forward(self) -> dict:
        graph = SongoGraphBuilder().build_batch(self.positions(2)).to(self.device)
        with torch.inference_mode():
            policy, value = self.model(graph)
        policy, value = policy.cpu(), value.cpu()
        if tuple(policy.shape) != (2, 7) or tuple(value.shape) != (2,):
            raise RuntimeError(f"unexpected shapes {tuple(policy.shape)} {tuple(value.shape)}")
        if not (torch.isfinite(policy).all() and torch.isfinite(value).all()):
            raise RuntimeError("non-finite model output")
        return {"policy_shape": list(policy.shape), "value_shape": list(value.shape), "dtype": str(policy.dtype), "device": str(self.device), "returned_to_cpu": True}

    def engine(self) -> dict:
        initial = RawSongoState(tuple([5] * 14 + [0, 0]), 1)
        game = SongoLegacyGame.from_state(initial.to_engine_state())
        game.normalize_terminal()
        legal = game.legal_local_actions()
        copy = game.clone()
        result = game.play_local(legal[0])
        conserved = sum(game.board) == 70
        turn_switched = result.finished or game.turn == 2
        copy_untouched = copy.board == [5] * 14 + [0, 0] and copy.turn == 1
        terminal = SongoLegacyGame.from_state(RawSongoState(tuple([0] * 7 + [5] * 7 + [35, 0]), 1).to_engine_state())
        terminal.normalize_terminal()
        details = {"initial_legal_actions": legal, "conservation": conserved, "turn_switched": turn_switched, "clone_independent": copy_untouched, "terminal_detected": terminal.finished, "fingerprint_initial": fingerprint_state(initial)}
        if not (len(legal) == 7 and conserved and turn_switched and copy_untouched and terminal.finished):
            raise RuntimeError(f"engine check failed: {details}")
        return details

    def mcts(self, budget: int) -> dict:
        rows = []
        for i, state in enumerate(self.positions(3)):
            search = SongoMCTS(self.model, config=MCTSConfig(num_simulations=budget, c_puct=C_PUCT, add_root_noise=False, seed=1000 + i), **LOT40_FLAGS)
            with torch.inference_mode():
                result = search.search(state, policy_temperature=POLICY_TEMPERATURE)
            validate_result(fingerprint_state(state), result, budget)
            rows.append({"selected_action": result.selected_action, "simulations": result.num_simulations, "root_value": result.root_value})
        return {"positions": len(rows), "rows": rows}

    def batched(self) -> dict:
        states = self.positions(4)
        seeds = [2000 + i for i in range(4)]
        search = SongoMCTS(self.model, config=MCTSConfig(num_simulations=32, c_puct=C_PUCT, add_root_noise=False, seed=seeds[0]), **LOT40_FLAGS)
        started = time.perf_counter()
        with torch.inference_mode():
            many = search.search_many(states, policy_temperature=POLICY_TEMPERATURE, seeds=seeds)
        elapsed = time.perf_counter() - started
        for state, result in zip(states, many):
            validate_result(fingerprint_state(state), result, 32)
        sequential = []
        for state, seed in zip(states, seeds):
            with torch.inference_mode():
                sequential.append(SongoMCTS(self.model, config=MCTSConfig(num_simulations=32, c_puct=C_PUCT, add_root_noise=False, seed=seed), **LOT40_FLAGS).search(state, policy_temperature=POLICY_TEMPERATURE))
        equal = [a.visit_counts == b.visit_counts for a, b in zip(many, sequential)]
        write_json(self.tmp / "batched.json", {"rows": [{"visits": list(r.visit_counts), "q": list(r.root_q_values)} for r in many]})
        return {"positions": 4, "budget": 32, "elapsed_s": elapsed, "serializable": True, "visits_equal_sequential": equal, "note": "visit equality with sequential search is diagnostic (GPU batched numerics may differ)"}

    def artifacts(self) -> dict:
        self.tmp.mkdir(parents=True, exist_ok=True)
        obj = {"a": 1, "b": [1.5, None, "x"], "c": {"d": True}}
        write_json(self.tmp / "x.json", obj)
        rows = [{"k": "1", "v": "a"}, {"k": "2", "v": "b"}]
        write_jsonl(self.tmp / "x.jsonl", [{"i": 1}, {"i": 2}])
        write_csv(self.tmp / "x.csv", rows)
        ok = read_json(self.tmp / "x.json") == obj and read_jsonl(self.tmp / "x.jsonl") == [{"i": 1}, {"i": 2}] and read_csv(self.tmp / "x.csv") == rows
        try:
            write_json(self.tmp / "nan.json", {"x": float("nan")})
            nan_rejected = False
        except ValueError:
            nan_rejected = True
        if not ok or not nan_rejected:
            raise RuntimeError(f"artifact round-trip ok={ok} nan_rejected={nan_rejected}")
        return {"json": True, "jsonl": True, "csv": True, "nan_rejected": True}

    def manifest(self) -> dict:
        path = self.tmp / "m.json"
        write_checked_json(path, {"rows": [1, 2, 3]})
        valid = checked_status(path) == "VALID"
        return {"sidecar": sidecar(path).name, "valid": valid, "_status": "PASS" if valid else "FAIL"}

    def checksum(self) -> dict:
        path = self.tmp / "m.json"
        path.write_text(path.read_text().replace("3", "4"))
        corrupt_detected = checked_status(path) == "CORRUPT"
        return {"tamper_detected": corrupt_detected, "_status": "PASS" if corrupt_detected else "FAIL"}

    def resume(self) -> dict:
        root = self.tmp / "resume"
        states = read_json(self.ctx.original_positions)["states"][:6]
        records = [{"fingerprint": fingerprint_state(state_from_dict(s)), "state": {"board": s["board"], "player_to_move": s["player_to_move"]}, "game_id": str(i)} for i, s in enumerate(states)]
        identity = SearchIdentity(budget=16, seed=7, model_fingerprints=self.ctx.fingerprints()["model"], engine_fingerprint=self.ctx.fingerprints()["engine"])
        errors = ErrorLog(root / "errors.jsonl")
        common = dict(partition="train", records=records, identity=identity, concurrency=2, model_loader=lambda: self.model, device=self.device, errors=errors, log=lambda _: None)
        first = run_search(root, max_shards=2, **common)
        shard_files = sorted((root / "search/train/16").glob("shard_*_2.json"))
        before = {p.name: (sha256(p), p.stat().st_mtime_ns) for p in shard_files}
        second = run_search(root, **common)
        after = {p.name: (sha256(p), p.stat().st_mtime_ns) for p in shard_files}
        third = run_search(root, **common)
        ok = first["status"] == "PARTIAL" and first["computed"] == 4 and second["computed"] == 2 and second["already_complete"] == 4 and before == after and third["computed"] == 0 and third["status"] == "COMPLETE"
        if not ok:
            raise RuntimeError(f"resume semantics broken: {first['computed']}/{second['computed']}/{third['computed']}")
        return {"interrupted_after_shards": 2, "recomputed_completed_shards": 0, "resumed_positions": second["computed"], "idempotent_rerun_computed": third["computed"]}

    def finalize_export(self) -> tuple[dict, dict]:
        root = self.tmp / "bundle_src"
        write_json(root / "decision.json", {"ok": True})
        write_csv(root / "t.csv", [{"a": "1"}])
        write_json(root / "checksums.json", build_checksums(root))
        bundle = self.tmp / "mini_results.tar.gz"
        report = make_bundle(root, bundle, "lot44_preflight")
        with tarfile.open(bundle, "r:gz") as archive:
            names = archive.getnames()
        tampered = self.tmp / "tampered.tar.gz"
        shutil.copy(bundle, tampered)
        shutil.copy(bundle.with_name(bundle.name + ".sha256"), tampered.with_name(tampered.name + ".sha256"))
        with tampered.open("ab") as stream:
            stream.write(b"x")
        try:
            verify_bundle(tampered, "lot44_preflight")
            tamper_detected = False
        except Lot44FatalError:
            tamper_detected = True
        finalize = {"checksums_written": True, "members": names}
        export = {**report, "tamper_detected": tamper_detected, "_status": "PASS" if tamper_detected and report["bundle_valid"] else "FAIL"}
        return finalize, export

    def weights_unchanged(self) -> dict:
        after = model_parameter_fingerprint(self.model)
        if after != self.fingerprint_before:
            raise Lot44FatalError("MODEL_WEIGHTS_CHANGED", "weights changed during preflight")
        return {"MODEL_WEIGHTS_CHANGED": "NO", "parameter_fingerprint": after}

    def extra_checks(self) -> list[tuple[str, Callable[[], dict], bool]]:
        """Verifications supplementaires des lots derives (nom, fonction, critique)."""

        return []

    def run(self) -> dict:
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
        self.check("PYTHON_IMPORTS", self.python_imports)
        self.check("INPUTS", self.inputs)
        self.check("CUDA", self.cuda, critical=self.ctx.device_name == "cuda")
        self.check("DRIVE", self.drive)
        self.check("CHECKPOINT", self.checkpoint)
        self.check("PROJECT_IMPORTS", self.project_imports)
        model_ready = self.model is not None
        for name, fn in (("MODEL_FORWARD", self.model_forward), ("ENGINE", self.engine), ("MCTS16", lambda: self.mcts(16)), ("MCTS64", lambda: self.mcts(64)), ("BATCHED_MCTS", self.batched)):
            if name == "ENGINE" or model_ready:
                self.check(name, fn)
            else:
                self.checks[name] = {"status": "FAIL", "critical": True, "error": "model unavailable (CHECKPOINT failed)"}
        self.check("ARTIFACT_WRITE_READ", self.artifacts)
        self.check("MANIFEST", self.manifest)
        self.check("CHECKSUM", self.checksum)
        if model_ready:
            self.check("RESUME", self.resume)
        else:
            self.checks["RESUME"] = {"status": "FAIL", "critical": True, "error": "model unavailable (CHECKPOINT failed)"}
        holder: dict = {}

        def fin() -> dict:
            holder["finalize"], holder["export"] = self.finalize_export()
            return holder["finalize"]

        def exported() -> dict:
            if "export" not in holder:
                raise RuntimeError("FINALIZE check failed, no bundle to verify")
            return dict(holder["export"])

        self.check("FINALIZE", fin)
        self.check("EXPORT", exported)
        extra = self.extra_checks()
        for name, fn, critical in extra:
            self.check(name, fn, critical=critical)
        if model_ready:
            self.check("MODEL_WEIGHTS_UNCHANGED", self.weights_unchanged)
        else:
            self.checks["MODEL_WEIGHTS_UNCHANGED"] = {"status": "FAIL", "critical": True, "error": "model unavailable"}
        shutil.rmtree(self.tmp, ignore_errors=False)
        order = ORDER + tuple(name for name, _, _ in extra)
        failed = [n for n in order if self.checks[n]["critical"] and self.checks[n]["status"] != "PASS"]
        status = "PASS" if not failed else "FAIL"
        report = {
            "lot": 44,
            "timestamp_utc": utc_now(),
            "code_commit": self.ctx.fingerprints()["code_commit"] if model_ready else None,
            "device": str(self.device) if self.device else self.ctx.device_name,
            "CUDA_AVAILABLE": "YES" if torch.cuda.is_available() else "NO",
            "checks": {n: self.checks[n] for n in order},
            "summary": {n: self.checks[n]["status"] for n in order},
            "critical_failures": failed,
            "PREFLIGHT_STATUS": status,
            "SCIENTIFIC_RUN_ALLOWED": "YES" if status == "PASS" else "NO",
        }
        write_json(self.ctx.out / "preflight_report.json", report)
        return report


def run_preflight(ctx: Context, *, require_lot_inputs: bool) -> dict:
    ctx.out.mkdir(parents=True, exist_ok=True)
    (ctx.out / "errors.jsonl").touch()
    return Preflight(ctx, require_lot_inputs=require_lot_inputs).run()
