import json
import math
import shutil

import pytest

from lot44.artifacts import (
    ErrorLog,
    Lot44FatalError,
    append_execution,
    build_checksums,
    canonical_hash,
    checked_status,
    make_bundle,
    read_checked_json,
    read_csv,
    read_json,
    read_jsonl,
    sidecar,
    update_stage_state,
    verify_bundle,
    verify_checksums,
    with_retries,
    write_checked_json,
    write_csv,
    write_json,
    write_jsonl,
)


def test_json_jsonl_csv_roundtrip(tmp_path):
    obj = {"a": [1, 2.5, None], "b": {"c": "é"}}
    write_json(tmp_path / "x.json", obj)
    assert read_json(tmp_path / "x.json") == obj
    write_jsonl(tmp_path / "x.jsonl", [{"i": 1}, {"i": 2}])
    assert read_jsonl(tmp_path / "x.jsonl") == [{"i": 1}, {"i": 2}]
    rows = [{"k": "1", "v": "a,b"}, {"k": "2", "v": ""}]
    write_csv(tmp_path / "x.csv", rows)
    assert read_csv(tmp_path / "x.csv") == rows


def test_nan_is_rejected_and_no_partial_file(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "nan.json", {"x": math.nan})
    assert not (tmp_path / "nan.json").exists()
    assert not list(tmp_path.glob(".nan.json.tmp-*"))


def test_atomic_write_leaves_no_temporary(tmp_path):
    write_json(tmp_path / "a.json", {"x": 1})
    write_json(tmp_path / "a.json", {"x": 2})
    assert read_json(tmp_path / "a.json") == {"x": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_checked_status_lifecycle(tmp_path):
    path = tmp_path / "s.json"
    assert checked_status(path) == "MISSING"
    write_json(path, {"x": 1})
    assert checked_status(path) == "INCOMPLETE"
    write_checked_json(path, {"x": 1})
    assert checked_status(path) == "VALID"
    assert read_checked_json(path) == {"x": 1}
    path.write_text(json.dumps({"x": 2}))
    assert checked_status(path) == "CORRUPT"
    with pytest.raises(Lot44FatalError) as err:
        read_checked_json(path)
    assert err.value.code == "CHECKSUM_MISMATCH"
    assert sidecar(path).name == "s.json.sha256.json"


def test_checksums_and_bundle_roundtrip_and_tamper(tmp_path):
    root = tmp_path / "exp"
    write_json(root / "decision.json", {"ok": True})
    write_csv(root / "sub" / "t.csv", [{"a": "1"}])
    write_json(root / "heartbeat.json", {"volatile": True})
    checksums = build_checksums(root)
    assert "heartbeat.json" not in checksums and "sub/t.csv" in checksums
    write_json(root / "checksums.json", checksums)
    assert verify_checksums(root, checksums) == []
    bundle = tmp_path / "out" / "r.tar.gz"
    report = make_bundle(root, bundle, "lot44_x")
    assert report["bundle_valid"] and report["checksummed_members"] == len(checksums)
    assert verify_bundle(bundle, "lot44_x")["bundle_valid"]
    write_json(root / "decision.json", {"ok": False})
    assert verify_checksums(root, checksums) == ["decision.json"]
    tampered = tmp_path / "out" / "t.tar.gz"
    shutil.copy(bundle, tampered)
    shutil.copy(bundle.with_name(bundle.name + ".sha256"), tampered.with_name(tampered.name + ".sha256"))
    with tampered.open("ab") as f:
        f.write(b"0")
    with pytest.raises(Lot44FatalError):
        verify_bundle(tampered, "lot44_x")


def test_retries_recover_then_give_up(monkeypatch):
    monkeypatch.setattr("lot44.artifacts.time.sleep", lambda _: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("drive busy")
        return "ok"

    assert with_retries(flaky, what="t", max_retries=3) == "ok"
    with pytest.raises(OSError):
        with_retries(lambda: (_ for _ in ()).throw(OSError("down")), what="t", max_retries=2)
    with pytest.raises(ValueError):
        with_retries(lambda: (_ for _ in ()).throw(ValueError("logic")), what="t", max_retries=5)


def test_error_log_records_required_fields(tmp_path):
    log = ErrorLog(tmp_path / "errors.jsonl")
    try:
        raise Lot44FatalError("ILLEGAL_ACTION", "boom")
    except Lot44FatalError as exc:
        log.record(stage="search/test/65536", exc=exc, shard="shard_a", position_fingerprint="abc", retry_count=1, last_valid_checkpoint="search/x.json")
    entry = read_jsonl(tmp_path / "errors.jsonl")[0]
    for key in ("stage", "shard", "position_fingerprint", "exception_class", "message", "traceback", "retry_count", "timestamp", "last_valid_checkpoint", "error_code"):
        assert key in entry
    assert entry["error_code"] == "ILLEGAL_ACTION" and "Traceback" in entry["traceback"]


def test_stage_state_and_execution_manifest(tmp_path):
    update_stage_state(tmp_path, "a", {"status": "RUNNING"})
    update_stage_state(tmp_path, "a", {"status": "COMPLETE"})
    update_stage_state(tmp_path, "b", {"status": "PENDING"})
    state = read_json(tmp_path / "stage_state.json")["stages"]
    assert state["a"]["status"] == "COMPLETE" and state["b"]["status"] == "PENDING"
    append_execution(tmp_path, {"stage": "x"})
    append_execution(tmp_path, {"stage": "y"})
    assert [e["stage"] for e in read_json(tmp_path / "execution_manifest.json")["invocations"]] == ["x", "y"]


def test_canonical_hash_is_order_independent_for_keys():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
    assert canonical_hash([1, 2]) != canonical_hash([2, 1])
