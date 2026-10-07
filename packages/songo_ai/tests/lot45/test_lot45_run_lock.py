import time

import pytest

from lot44.artifacts import Lot44FatalError, read_json
from lot45.run_lock import RunLock


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t


def lock(tmp_path, owner, clock, **kw):
    return RunLock(tmp_path, owner=owner, settle_s=0, clock=clock, **kw)


def test_single_writer_and_release(tmp_path):
    clock = Clock()
    a = lock(tmp_path, "colab", clock)
    a.acquire()
    payload = read_json(tmp_path / "run_lock.json")
    for key in ("run_id", "machine_id", "hostname", "pid", "start_utc", "heartbeat_utc", "owner"):
        assert key in payload
    with pytest.raises(Lot44FatalError) as err:
        lock(tmp_path, "mac", clock).acquire()
    assert err.value.code == "RUN_LOCKED"
    a.assert_owner()
    a.release()
    b = lock(tmp_path, "mac", clock)
    b.acquire()
    with pytest.raises(Lot44FatalError) as err:
        a.assert_owner()
    assert err.value.code == "LOCK_LOST"
    b.release()


def test_stale_lock_requires_explicit_takeover_and_is_logged(tmp_path):
    clock = Clock()
    old = lock(tmp_path, "colab", clock, stale_after_s=600)
    old.acquire()
    clock.t += 300
    with pytest.raises(Lot44FatalError) as err:
        lock(tmp_path, "mac", clock, stale_after_s=600).acquire(takeover_stale=True)
    assert err.value.code == "RUN_LOCKED"  # heartbeat encore frais : jamais de reprise
    clock.t += 400
    new = lock(tmp_path, "mac", clock, stale_after_s=600)
    with pytest.raises(Lot44FatalError) as err:
        new.acquire()
    assert err.value.code == "STALE_LOCK"
    new.acquire(takeover_stale=True, reason="colab runtime died")
    with pytest.raises(Lot44FatalError):
        old.assert_owner()
    events = read_json(tmp_path / "run_lock_history.json")["events"]
    takeover = [e for e in events if e["event"] == "TAKEOVER"][0]
    assert takeover["previous_run_id"] == old.run_id and takeover["reason"] == "colab runtime died" and takeover["previous_heartbeat_utc"]
    assert takeover["new_owner"] and takeover["timestamp_utc"]


def test_heartbeat_refreshes_lock(tmp_path):
    a = RunLock(tmp_path, owner="x", settle_s=0, heartbeat_s=0.05)
    a.acquire()
    first = read_json(tmp_path / "run_lock.json")["heartbeat_utc"]
    a.clock = lambda: time.time() + 5
    a.start_heartbeat()
    time.sleep(0.3)
    a.release()
    payload = read_json(tmp_path / "run_lock.json")
    assert payload["heartbeat_utc"] != first and payload["released"]
