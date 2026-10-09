"""Execution distribuee Lot45 : tests REELLEMENT concurrents (processus spawn)
contre l'emulateur Firestore (transactions reelles), et MCTS reel a petit budget."""
import json
import multiprocessing
import os
import signal
import threading
import time

import pytest

from lot44.artifacts import Lot44FatalError, read_json, sidecar
from lot45.distributed.config import parse_coordinator_config
from lot45.distributed.coordinator import COMPLETED, FAILED_RETRYABLE, RUNNING, CoordinatorUnavailable, Lease, check_transition
from lot45.distributed.heartbeat import LeaseHeartbeat
from lot45.distributed.storage import attempt_root, validate_attempt
from lot45.finalize import DATASET_FILE
from lot45.generation import generate, prepare
from lot45.selection import coverage_report, write_selection
from lot45.smoke import smoke_selection

from lot45_distributed_support import Emulator, ToggleProxy, coordinator, new_namespace, pending, proc_claim, proc_complete, proc_drain, proc_exclusive, proc_hold, proc_worker
from lot45_test_support import PROBE, make_context, require_inputs

SPAWN = multiprocessing.get_context("spawn")
MCTS = dict(budget=64, shard_size=2, concurrency=2, target_positions=24)


@pytest.fixture(scope="module")
def emulator():
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    emu = Emulator()
    yield emu
    emu.stop()


def run_processes(target, arglists, *, timeout=180, exit_codes=None):
    queue = SPAWN.Queue()
    procs = [SPAWN.Process(target=target, args=(*args, queue)) for args in arglists]
    for p in procs:
        p.start()
    results = [queue.get(timeout=timeout) for _ in procs]
    for p in procs:
        p.join(timeout=timeout)
    codes = sorted(p.exitcode for p in procs)
    assert codes == sorted(exit_codes or [0] * len(procs)), f"{target.__name__} exit codes {codes}"
    return results


def scratch(emulator, n, **kwargs):
    ns = new_namespace()
    coord = coordinator(emulator.host, ns, "run", **kwargs)
    coord.import_shards({"scratch": True}, pending(n))
    return ns, coord


def prepared(tmp_path):
    """Experience miniature : 24 positions de sonde, shards de 2 -> 12 shards."""

    require_inputs()
    out = tmp_path / "exp"
    out.mkdir()
    rows = smoke_selection(PROBE, MCTS["target_positions"])
    write_selection(out / "selection.jsonl.gz", rows)
    for name in ("candidate_source_audit.json", "source_selection.json", "source_manifest.json", "deduplication_report.json"):
        (out / name).write_text("{}")
    (out / "coverage_report.json").write_text(json.dumps(coverage_report(rows)))
    ctx = make_context(out, **MCTS)
    prepare(ctx)
    return ctx


def real_coordinator(emulator, ctx, namespace, **kwargs):
    from lot45.distributed.migration import run_key

    return coordinator(emulator.host, namespace, run_key(read_json(ctx.out / "shard_manifest.json")), **kwargs)


# -- 1-3 : reservation atomique, vraie concurrence ---------------------------------------
def test_two_and_eight_workers_race_for_one_shard_single_lease(emulator):
    ns, coord = scratch(emulator, 1)
    barrier = SPAWN.Barrier(8)
    results = run_processes(proc_claim, [(emulator.host, ns, "run", f"w{i}", barrier) for i in range(8)])
    winners = [w for w, lease in results if lease is not None]
    assert len(winners) == 1
    record = coord.shard_records()[0]
    assert record["status"] == RUNNING and record["worker_id"] == winners[0] and record["fencing_token"] == 1


def test_three_workers_take_three_distinct_shards(emulator):
    ns, coord = scratch(emulator, 3)
    barrier = SPAWN.Barrier(3)
    results = run_processes(proc_claim, [(emulator.host, ns, "run", f"w{i}", barrier) for i in range(3)])
    shards = [lease["shard_id"] for _, lease in results]
    assert sorted(shards) == ["shard_00000", "shard_00001", "shard_00002"]


def test_six_workers_drain_twelve_shards_exactly_once(emulator):
    ns, coord = scratch(emulator, 12)
    barrier = SPAWN.Barrier(6)
    results = run_processes(proc_drain, [(emulator.host, ns, "run", f"w{i}", barrier) for i in range(6)])
    taken = [s for _, shards in results for s in shards]
    assert sorted(taken) == [f"shard_{i:05d}" for i in range(12)]
    assert coord.progress() == {"total": 12, "completed": 12, "failed_fatal": 0, "remaining": 0, "all_settled": True}


# -- 4-9 : panne, expiration, reprise, fencing, heartbeat --------------------------------
def test_killed_worker_lease_expires_and_is_recovered(emulator):
    ns, coord = scratch(emulator, 1, lease_ttl_s=3.0)
    queue = SPAWN.Queue()
    proc = SPAWN.Process(target=proc_hold, args=(emulator.host, ns, "run", "victim", queue, 3.0))
    proc.start()
    held = Lease(**queue.get(timeout=60))
    os.kill(proc.pid, signal.SIGKILL)
    proc.join()
    assert coord.claim_next_shard("rescuer", "r") is None  # bail encore valide
    time.sleep(3.5)
    rescued = coord.claim_next_shard("rescuer", "r")
    assert rescued.shard_id == held.shard_id and rescued.fencing_token == held.fencing_token + 1 and rescued.attempt_number == 2
    assert [e["event"] for e in coord.event_records()][-1] == "TAKEOVER"


def test_returning_worker_is_fenced_out(emulator):
    ns, coord = scratch(emulator, 1, lease_ttl_s=1.0)
    old = coord.claim_next_shard("A", "a")
    beat = LeaseHeartbeat(coord, old, interval_s=60)
    time.sleep(1.5)
    new = coord.claim_next_shard("B", "b")
    assert new.fencing_token == old.fencing_token + 1
    beat.beat()  # A revient : son renouvellement echoue
    assert beat.lost
    with pytest.raises(Lot44FatalError) as err:
        beat.assert_valid()
    assert err.value.code == "LEASE_LOST"
    with pytest.raises(Lot44FatalError) as err:
        coord.complete(old, {"files": [], "search_identity_fingerprint": "x"})
    assert err.value.code == "FENCING_REJECTED"
    assert coord.shard_records()[0]["status"] == RUNNING
    coord.complete(new, {"files": [], "search_identity_fingerprint": "x"})
    record = coord.shard_records()[0]
    assert record["status"] == COMPLETED and record["artifact"]["fencing_token"] == new.fencing_token and record["artifact"]["worker_id"] == "B"
    assert "COMPLETE_REJECTED" in [e["event"] for e in coord.event_records()]


def test_heartbeat_renews_and_keeps_the_lease(emulator):
    ns, coord = scratch(emulator, 1, lease_ttl_s=2.0)
    lease = coord.claim_next_shard("A", "a")
    beat = LeaseHeartbeat(coord, lease, interval_s=0.3).start()
    time.sleep(3.0)
    assert coord.claim_next_shard("B", "b") is None
    beat.stop()
    beat.assert_valid()
    assert beat.renewals >= 5 and beat.lease.lease_expires_at > lease.lease_expires_at + 2


def test_heartbeat_marks_lease_lost_after_expiry_without_renewal(emulator):
    ns, coord = scratch(emulator, 1, lease_ttl_s=1.0)
    lease = coord.claim_next_shard("A", "a")
    beat = LeaseHeartbeat(coord, lease, interval_s=60)
    time.sleep(1.2)
    with pytest.raises(Lot44FatalError) as err:
        beat.assert_valid()
    assert err.value.code == "LEASE_LOST"


def test_transitions_are_validated():
    check_transition("PENDING", "RUNNING")
    check_transition("RUNNING", "COMPLETED")
    for old, new in (("COMPLETED", "RUNNING"), ("COMPLETED", "PENDING"), ("FAILED_FATAL", "RUNNING"), ("PENDING", "COMPLETED")):
        with pytest.raises(Lot44FatalError) as err:
            check_transition(old, new)
        assert err.value.code == "INVALID_TRANSITION"


def test_attempts_are_bounded_then_failed_fatal(emulator):
    ns, coord = scratch(emulator, 1, lease_ttl_s=0.2, max_attempts_per_shard=2)
    first = coord.claim_next_shard("A", "a")
    time.sleep(0.3)
    second = coord.claim_next_shard("B", "b")
    assert second.attempt_number == 2
    time.sleep(0.3)
    assert coord.claim_next_shard("C", "c") is None
    assert coord.shard_records()[0]["status"] == "FAILED_FATAL" and coord.progress()["all_settled"]
    assert first.attempt_number == 1


# -- 10, 14, 15 : idempotence et fins simultanees -------------------------------------------
def test_simultaneous_completions_are_all_counted(emulator):
    ns, coord = scratch(emulator, 6)
    leases = [coord.claim_next_shard(f"w{i}", "r") for i in range(6)]
    barrier = SPAWN.Barrier(6)
    results = run_processes(proc_complete, [(emulator.host, ns, "run", lease.as_dict(), barrier) for lease in leases])
    assert sorted(status for _, status in results) == ["COMPLETED"] * 6
    assert coord.progress()["completed"] == 6
    again = coord.complete(leases[0], {"files": [], "search_identity_fingerprint": "test"})
    assert again["status"] == "ALREADY_COMPLETED" and coord.progress()["completed"] == 6
    assert coord.claim_next_shard("late", "r") is None


# -- 16-17 : finalisation -------------------------------------------------------------------
def test_concurrent_finalizers_only_one_granted(emulator):
    ns, coord = scratch(emulator, 1)
    barrier = SPAWN.Barrier(4)
    results = run_processes(proc_exclusive, [(emulator.host, ns, "run", f"f{i}", barrier) for i in range(4)])
    assert sorted(code for _, code in results) == ["FINALIZER_ACTIVE"] * 3 + ["GRANTED"]


# -- 18 : panne temporaire du coordinateur ---------------------------------------------------
def test_temporary_coordinator_outage_is_retried_then_bounded(emulator):
    ns, setup = scratch(emulator, 2)
    proxy = ToggleProxy(emulator.port)
    coord = coordinator(proxy.host, ns, "run", max_retries=6, backoff_s=0.5, op_timeout_s=5.0)
    assert coord.claim_next_shard("A", "a") is not None
    proxy.down()
    threading.Timer(2.0, proxy.up).start()
    lease = coord.claim_next_shard("A", "a")  # traverse la panne grace aux retries bornes
    assert lease is not None and lease.shard_id in {"shard_00000", "shard_00001"}
    proxy.down()
    impatient = coordinator(proxy.host, ns, "run", max_retries=1, backoff_s=0.2, op_timeout_s=2.0)
    started = time.time()
    with pytest.raises(CoordinatorUnavailable):
        impatient.progress()
    assert time.time() - started < 60
    proxy.up()


def test_coordinator_config_is_explicit():
    with pytest.raises(Lot44FatalError):
        parse_coordinator_config(None)
    with pytest.raises(Lot44FatalError):
        parse_coordinator_config({"namespace": "x"})
    with pytest.raises(Lot44FatalError):
        parse_coordinator_config({"project": "p", "backend": "drive-json"})
    with pytest.raises(Lot44FatalError):
        parse_coordinator_config({"project": "p", "secret": "x"})
    config = parse_coordinator_config('{"project": "songo-model-ai", "credentials": "/content/drive/key.json"}')
    assert config.public()["credentials"] == "service_account_file" and "key.json" not in json.dumps(config.public())


# -- 17 (E2E) : 3 workers MCTS reels simultanes, 12 shards, un worker interrompu --------------
def test_end_to_end_three_workers_twelve_shards_with_a_crash(emulator, tmp_path):
    from lot45.distributed.finalizer import finalize_distributed
    from lot45.distributed.migration import migrate

    ctx = prepared(tmp_path)
    ns = new_namespace()
    coord = real_coordinator(emulator, ctx, ns, lease_ttl_s=6.0)
    with pytest.raises(Lot44FatalError) as err:
        finalize_distributed(ctx, coord, holder="early")
    assert err.value.code == "COORDINATOR_RUN_MISSING"
    migrated = migrate(ctx, coord, holder="migrator")
    assert migrated["status"] == "MIGRATED" and migrated["counts"]["PENDING"] == 12
    assert migrate(ctx, coord, holder="again")["status"] == "ALREADY_MIGRATED"
    with pytest.raises(Lot44FatalError) as err:
        finalize_distributed(ctx, coord, holder="too-early")
    assert err.value.code == "FINALIZE_TOO_EARLY"
    with pytest.raises(Lot44FatalError) as err:
        generate(ctx, owner="legacy")
    assert err.value.code == "MIGRATED_TO_DISTRIBUTED"
    overrides = {k: v for k, v in MCTS.items()}
    results = run_processes(proc_worker, [(emulator.host, ns, str(ctx.out), overrides, "w-crash", True), (emulator.host, ns, str(ctx.out), overrides, "w-a", False), (emulator.host, ns, str(ctx.out), overrides, "w-b", False)], timeout=600, exit_codes=[0, 0, 137])
    assert sum(r[1] == "CRASHED" for r in results) == 1
    assert all(r[1] == "ALL_SHARDS_SETTLED" for r in results if r[1] != "CRASHED")
    crashed_shard = next(r[2] for r in results if r[1] == "CRASHED")
    decision = finalize_distributed(ctx, coord, holder="finalizer")
    dist = decision["DISTRIBUTED"]
    assert (dist["COMPLETED"], dist["DUPLICATE_ACCEPTED"], dist["MISSING"], dist["CHECKSUM_ERRORS"]) == (12, 0, 0, 0)
    assert dist["takeovers"] >= 1 and set(dist["workers"]) <= {"w-a", "w-b"}
    assert decision["LOT45_VALID"] == "YES" and decision["UNIQUE_PHYSICAL_STATES"] == 24 and decision["ALL_BUDGETS_EXACT_64"]
    record = next(r for r in coord.shard_records() if r["shard_id"] == crashed_shard)
    assert record["attempt_number"] >= 2 and record["artifact"]["worker_id"] in {"w-a", "w-b"}
    manifest = read_json(ctx.out / record["artifact"]["attempt_manifest"]["path"])
    assert {"worker_id", "device", "checkpoint_fingerprints", "mcts_identity_fingerprint", "engine_fingerprint", "dataset_fingerprint", "seed", "attempt_number"} <= set(manifest)
    assert (ctx.out / DATASET_FILE).is_file()


# -- 11, 12, 13, 19, 20 : ecriture interrompue, checksum, reprise, anciens shards -----------
def test_legacy_shards_partial_write_checksum_and_restart(emulator, tmp_path, monkeypatch):
    import lot44.search as search_module
    from lot45.distributed import worker as worker_module
    from lot45.distributed.finalizer import finalize_distributed, preconditions
    from lot45.distributed.migration import audit, migrate
    from lot45.distributed.worker import WorkerSettings, run_worker
    from colab_drive import sha256

    ctx = prepared(tmp_path)
    generate(ctx, owner="legacy-runner", max_shards=3)  # 3 shards par l'ancien runner mono-ecrivain
    legacy_dir = ctx.out / "search/shard_00002/64"
    corrupt = next(p for p in legacy_dir.glob("shard_*.json") if not p.name.endswith(".sha256.json"))
    corrupt.write_text(corrupt.read_text() + " ")
    legacy_hashes = {str(p): sha256(p) for p in (ctx.out / "search").rglob("*.json")}
    report = audit(ctx)
    assert report["counts"]["COMPLETED_VALID"] == 2 and report["counts"]["INVALID"] == 1 and report["counts"]["PENDING"] == 9
    ns = new_namespace()
    coord = real_coordinator(emulator, ctx, ns, lease_ttl_s=30.0)
    migrated = migrate(ctx, coord, holder="m")
    assert migrated["counts"] == report["counts"] and coord.progress()["completed"] == 2
    assert (ctx.out / "distributed/migration").is_dir() and any((ctx.out / "distributed/migration").glob("backup_*/shard_manifest.json"))

    # 11/19 : interruption pendant l'ecriture du premier shard (fichier ecrit, sidecar absent)
    real_write = search_module.write_checked_json

    def interrupted(path, payload):
        from lot44.artifacts import write_json

        write_json(path, payload)
        raise KeyboardInterrupt("Colab disconnected while writing")

    monkeypatch.setattr(search_module, "write_checked_json", interrupted)
    settings = WorkerSettings(worker_id="w-interrupted", concurrency=2, heartbeat_s=5.0, idle_poll_s=0.2, max_idle_polls=5)
    with pytest.raises(KeyboardInterrupt):
        run_worker(ctx, coord, settings, log=lambda _: None)
    monkeypatch.setattr(search_module, "write_checked_json", real_write)
    interrupted_record = next(r for r in coord.shard_records() if r["status"] == FAILED_RETRYABLE)
    partial_root = attempt_root(ctx.out, interrupted_record["shard_id"], interrupted_record["lease_id"])
    partial = next(partial_root.rglob("shard_*.json"))
    assert not sidecar(partial).exists()
    records = [{"fingerprint": r["fingerprint"]} for r in ctx.selection()]
    with pytest.raises(Lot44FatalError) as err:
        validate_attempt(ctx.out, partial_root, interrupted_record["shard_id"], ctx.budget, ctx.identity().fingerprint(), records, {})
    assert err.value.code == "ARTIFACT_INCOMPLETE"

    # 19/14 : un nouveau worker (meme notebook, nouvel id) reprend tout ; un redemarrage ne recalcule rien
    summary = run_worker(ctx, coord, WorkerSettings(worker_id="w-restart", concurrency=2, heartbeat_s=5.0, idle_poll_s=0.2, max_idle_polls=5), log=lambda _: None)
    assert summary["stop_reason"] == "ALL_SHARDS_SETTLED" and len(summary["completed"]) == 10
    attempts_before = sorted(str(p) for p in (ctx.out / "attempts").rglob("*"))
    again = run_worker(ctx, coord, WorkerSettings(worker_id="w-again", concurrency=2, idle_poll_s=0.2, max_idle_polls=1), log=lambda _: None)
    assert again["completed"] == [] and again["stop_reason"] == "ALL_SHARDS_SETTLED"
    assert sorted(str(p) for p in (ctx.out / "attempts").rglob("*")) == attempts_before

    # 20 : anciens shards jamais reecrits ; le shard corrompu reste tel quel et n'est pas reference
    assert {str(p): sha256(p) for p in (ctx.out / "search").rglob("*.json")} == legacy_hashes
    by_shard = {r["shard_id"]: r for r in coord.shard_records()}
    assert by_shard["shard_00000"]["artifact"]["legacy"] and by_shard["shard_00002"]["artifact"].get("legacy") is None
    assert by_shard["shard_00002"]["legacy_classification"] == "INVALID"

    # 13 : un artefact reference modifie est rejete
    target = ctx.out / by_shard["shard_00005"]["artifact"]["files"][0]["path"]
    original = target.read_bytes()
    target.write_bytes(original + b" ")
    with pytest.raises(Lot44FatalError) as err:
        preconditions(ctx, coord)
    assert err.value.code == "CHECKSUM_MISMATCH"
    target.write_bytes(original)

    decision = finalize_distributed(ctx, coord, holder="final")
    assert decision["LOT45_VALID"] == "YES" and decision["DISTRIBUTED"]["legacy_shards"] == 2 and decision["DISTRIBUTED"]["COMPLETED"] == 12
    report = read_json(ctx.out / "distributed/distributed_report.json")
    assert str(partial_root.relative_to(ctx.out)) in report["unreferenced_attempts"]
    assert worker_module.FATAL_CODES and "LEASE_LOST" in worker_module.LOST_CODES


def test_distributed_preflight_against_the_emulator(emulator, tmp_path):
    from lot45.distributed.config import CoordinatorConfig
    from lot45.distributed.preflight import DistributedPreflight

    ctx = prepared(tmp_path)
    config = CoordinatorConfig(project="demo-lot45", namespace=new_namespace(), emulator_host=emulator.host)
    report = DistributedPreflight(ctx, config, "w-preflight").run()
    assert report["PREFLIGHT_STATUS"] == "PASS", report["critical_failures"]
    assert list(report["summary"]) == ["PROJECT_IMPORTS", "CHECKPOINT", "CUDA", "DRIVE", "COORDINATOR_CONNECTION", "COORDINATOR_ATOMICITY_TEST", "WORKER_REGISTRATION", "SHARD_CLAIM", "HEARTBEAT", "ARTIFACT_WRITE", "FENCING", "RESUME"]
    assert (ctx.out / "distributed/workers/w-preflight/preflight_report.json").is_file()
    assert report["scratch_docs_deleted"] >= 1
