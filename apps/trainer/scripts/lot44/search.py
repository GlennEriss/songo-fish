"""Recherches MCTS shardees, reprenables et idempotentes (sections 30-32, 85-87).

Le MCTS valide est utilise tel quel (``SongoMCTS.search_many`` + drapeaux
Lot40) : PUCT, backup, perspective Q, masquage, terminaux et semantique de la
racine sont inchanges ; le bruit de racine est desactive.
"""
from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from songo_ai.evaluation import model_parameter_fingerprint
from songo_ai.search import MCTSConfig, SongoMCTS
from run_srn_lot41 import LOT40_FLAGS
from run_srn_lot43 import rank_actions

from .artifacts import (
    ErrorLog,
    Lot44FatalError,
    canonical_hash,
    checked_status,
    memory_snapshot,
    read_json,
    sidecar,
    update_stage_state,
    utc_now,
    write_checked_json,
    write_heartbeat,
)
from .config import BASE_PROCESS_BYTES_ESTIMATE, BYTES_PER_NODE_ESTIMATE, C_PUCT, POLICY_TEMPERATURE, ROOT_NOISE
from .corpus import state_from_dict

SEARCH_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SearchIdentity:
    """Tout ce qui determine un resultat de recherche (hors concurrence)."""

    budget: int
    seed: int
    model_fingerprints: dict
    engine_fingerprint: str

    def payload(self) -> dict:
        return {
            "schema": SEARCH_SCHEMA_VERSION,
            "budget": self.budget,
            "seed_scheme": "sha256(seed,budget,fingerprint)[:8]",
            "seed": self.seed,
            "c_puct": C_PUCT,
            "policy_temperature": POLICY_TEMPERATURE,
            "root_noise": ROOT_NOISE,
            "lot40_flags": LOT40_FLAGS,
            "model_fingerprints": self.model_fingerprints,
            "engine_fingerprint": self.engine_fingerprint,
        }

    def fingerprint(self) -> str:
        return canonical_hash(self.payload())


def position_seed(seed: int, budget: int, fingerprint: str) -> int:
    """Graine par position : independante du sharding et de la concurrence."""

    return int(canonical_hash({"seed": seed, "budget": budget, "fp": fingerprint})[:8], 16)


def estimated_peak_bytes(budget: int, concurrency: int) -> int:
    return budget * concurrency * BYTES_PER_NODE_ESTIMATE + BASE_PROCESS_BYTES_ESTIMATE


def shard_dir(out: Path, partition: str, budget: int) -> Path:
    return out / "search" / partition / str(budget)


def validate_result(fp: str, result, budget: int) -> None:
    if result.num_simulations != budget:
        raise Lot44FatalError("SIMULATION_COUNT", f"{fp[:12]}: {result.num_simulations} simulations != {budget}")
    if sum(result.visit_counts) != budget:
        raise Lot44FatalError("SIMULATION_COUNT", f"{fp[:12]}: root visits {sum(result.visit_counts)} != {budget}")
    action = result.selected_action
    if action is None or not result.legal_mask[action]:
        raise Lot44FatalError("ILLEGAL_ACTION", f"{fp[:12]}: selected action {action} is not legal")
    if any(count > 0 and not legal for count, legal in zip(result.visit_counts, result.legal_mask)):
        raise Lot44FatalError("ILLEGAL_ACTION", f"{fp[:12]}: visits on an illegal action")
    values = [*result.policy, *result.root_q_values, result.root_value]
    if not all(math.isfinite(float(x)) for x in values):
        raise Lot44FatalError("NAN_CRITICAL", f"{fp[:12]}: non-finite policy/Q/value")


def result_row(fp: str, seed: int, budget: int, result) -> dict:
    return {
        "state_fingerprint": fp,
        "budget": budget,
        "seed": seed,
        "legal_mask": list(result.legal_mask),
        "visit_counts": list(result.visit_counts),
        "visit_distribution": list(result.policy),
        "selected_action": result.selected_action,
        "ranking": rank_actions({"legal_mask": list(result.legal_mask), "visit_counts": list(result.visit_counts)}),
        "root_q_values": list(result.root_q_values),
        "root_value": float(result.root_value),
        "root_priors": list(result.root_priors),
        "nodes": result.num_nodes,
        "network_evaluations": result.network_evaluations,
        "actual_simulations": result.num_simulations,
    }


def load_completed(directory: Path, identity_fp: str, expected: set[str], *, remove_incomplete: bool = True) -> tuple[dict[str, dict], list[str]]:
    """Lit les shards valides. Shard corrompu ou config differente = fatal."""

    rows: dict[str, dict] = {}
    shards = []
    if not directory.is_dir():
        return rows, shards
    for path in sorted(directory.glob("shard_*.json")):
        if path.name.endswith(".sha256.json"):
            continue
        status = checked_status(path)
        if status == "INCOMPLETE":
            # Ecrit sans sidecar : interruption entre les deux ecritures.
            if remove_incomplete:
                path.unlink()
            continue
        if status == "CORRUPT":
            raise Lot44FatalError("CHECKSUM_MISMATCH", f"corrupt shard {path}")
        payload = read_json(path)
        if payload.get("search_identity_fingerprint") != identity_fp:
            raise Lot44FatalError("MCTS_CONFIG_INCOMPATIBLE", f"{path.name} was produced with another search identity")
        if payload.get("status") != "COMPLETE":
            raise Lot44FatalError("ARTIFACT_INCONSISTENT", f"{path.name} has status {payload.get('status')}")
        for row in payload["rows"]:
            fp = row["state_fingerprint"]
            if fp not in expected:
                raise Lot44FatalError("SPLIT_MODIFIED", f"{path.name} contains {fp[:12]} outside the partition")
            if fp in rows:
                raise Lot44FatalError("ARTIFACT_INCONSISTENT", f"{fp[:12]} present in two shards")
            rows[fp] = row
        shards.append(path.name)
    return rows, shards


def run_search(
    out: Path,
    *,
    partition: str,
    records: list[dict],
    identity: SearchIdentity,
    concurrency: int,
    model_loader: Callable[[], torch.nn.Module],
    device: torch.device,
    errors: ErrorLog,
    max_shards: int | None = None,
    log: Callable[[str], None] = print,
    before_write: Callable[[], None] | None = None,
) -> dict:
    """Execute les positions manquantes de ``partition`` au budget donne.

    ``before_write`` est appele juste avant chaque ecriture de shard (ex. :
    verification du verrou mono-ecrivain de Lot45) et peut lever pour stopper.
    """

    budget = identity.budget
    directory = shard_dir(out, partition, budget)
    directory.mkdir(parents=True, exist_ok=True)
    identity_fp = identity.fingerprint()
    expected = {r["fingerprint"] for r in records}
    done, done_shards = load_completed(directory, identity_fp, expected)
    remaining = [r for r in sorted(records, key=lambda r: r["fingerprint"]) if r["fingerprint"] not in done]
    key = f"search/{partition}/{budget}"
    summary = {"partition": partition, "budget": budget, "total": len(records), "already_complete": len(done), "computed": 0, "shards_written": [], "skipped_shards": done_shards}
    if not remaining:
        update_stage_state(out, key, {"status": "COMPLETE", "completed_positions": len(done), "total_positions": len(records), "completed_shards": done_shards, "pending_shards": [], "failed_shards": []})
        log(f"[Lot44][{partition}][{budget}] SKIP COMPLETE {len(done)}/{len(records)}")
        return {**summary, "status": "COMPLETE"}
    model = model_loader()
    before = model_parameter_fingerprint(model)
    chunks = [remaining[i:i + concurrency] for i in range(0, len(remaining), concurrency)]
    started = time.time()
    completed = len(done)
    last_artifact = None
    for index, chunk in enumerate(chunks):
        if max_shards is not None and index >= max_shards:
            break
        name = f"shard_{chunk[0]['fingerprint'][:12]}_{chunk[-1]['fingerprint'][:12]}_{len(chunk)}.json"
        seeds = [position_seed(identity.seed, budget, r["fingerprint"]) for r in chunk]
        states = [state_from_dict(r["state"]) for r in chunk]
        shard_started = utc_now()
        t0 = time.perf_counter()
        try:
            search = SongoMCTS(model, config=MCTSConfig(num_simulations=budget, c_puct=C_PUCT, add_root_noise=ROOT_NOISE, seed=seeds[0]), **LOT40_FLAGS)
            with torch.inference_mode():
                results = search.search_many(states, policy_temperature=POLICY_TEMPERATURE, seeds=seeds)
            for record, result in zip(chunk, results):
                validate_result(record["fingerprint"], result, budget)
        except BaseException as exc:
            errors.record(stage=key, exc=exc, shard=name, position_fingerprint=",".join(r["fingerprint"][:12] for r in chunk), last_valid_checkpoint=last_artifact)
            update_stage_state(out, key, {"status": "FAILED", "completed_positions": completed, "total_positions": len(records), "failed_shards": [name]})
            raise
        elapsed = time.perf_counter() - t0
        rows = [result_row(r["fingerprint"], s, budget, res) for r, s, res in zip(chunk, seeds, results)]
        payload = {
            "status": "COMPLETE",
            "partition": partition,
            "budget": budget,
            "shard": name,
            "search_identity": identity.payload(),
            "search_identity_fingerprint": identity_fp,
            "input_fingerprints": [r["fingerprint"] for r in chunk],
            "input_sha256": canonical_hash([r["fingerprint"] for r in chunk]),
            "concurrency": len(chunk),
            "device": str(device),
            "start_utc": shard_started,
            "end_utc": utc_now(),
            "wall_time_s": elapsed,
            "simulations": budget * len(chunk),
            "simulations_per_second": budget * len(chunk) / elapsed if elapsed > 0 else None,
            "completed_positions": len(rows),
            "failed_positions": [],
            "memory": memory_snapshot(),
            "rows": rows,
        }
        if before_write is not None:
            before_write()
        write_checked_json(directory / name, payload)
        last_artifact = str((directory / name).relative_to(out))
        completed += len(rows)
        summary["computed"] += len(rows)
        summary["shards_written"].append(name)
        rate = summary["computed"] / (time.time() - started)
        eta = (len(records) - completed) / rate if rate > 0 else float("nan")
        mem = memory_snapshot()
        log(
            f"[Lot44][{partition}][{budget}] shard {index + 1:02d}/{len(chunks):02d} positions {completed}/{len(records)} "
            f"elapsed {time.time() - started:.0f}s ETA {eta:.0f}s sims/s {payload['simulations_per_second']:.0f} "
            f"RAM {mem['peak_rss_bytes'] / 1e9:.2f}GB GPU {mem.get('gpu_peak_bytes', 0) / 1e9:.2f}GB"
        )
        write_heartbeat(out, stage=key, completed=completed, total=len(records), current_shard=name, started=started, last_artifact=last_artifact)
        update_stage_state(out, key, {
            "status": "RUNNING" if completed < len(records) else "COMPLETE",
            "completed_positions": completed,
            "total_positions": len(records),
            "completed_shards": done_shards + summary["shards_written"],
            "pending_shards": len(chunks) - index - 1,
            "failed_shards": [],
            "last_shard_sha256": read_json(sidecar(directory / name))["sha256"],
        })
        del results, search
        gc.collect()
    after = model_parameter_fingerprint(model)
    if before != after:
        raise Lot44FatalError("MODEL_WEIGHTS_CHANGED", "model parameters changed during Lot44 search")
    status = "COMPLETE" if completed == len(records) else "PARTIAL"
    return {**summary, "status": status, "model_weights_changed": False}


def load_search_rows(out: Path, partition: str, budget: int, identity_fp: str, expected: set[str]) -> dict[str, dict]:
    """Charge une recherche et exige qu'elle soit complete."""

    rows, _ = load_completed(shard_dir(out, partition, budget), identity_fp, expected, remove_incomplete=False)
    if set(rows) != expected:
        raise Lot44FatalError("INCOMPLETE_SEARCH", f"{partition}/{budget}: {len(rows)}/{len(expected)} positions complete")
    return rows
