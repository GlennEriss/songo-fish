"""Support des tests distribues : emulateur Firestore reel, processus concurrents
(spawn) et proxy TCP pour simuler une panne du coordinateur."""
import glob
import os
import shutil
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

import pytest

import lot45_test_support  # noqa: F401  (chemins scripts/packages)

PROJECT = "demo-lot45"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def emulator_jar() -> str | None:
    explicit = os.environ.get("LOT45_FIRESTORE_EMULATOR_JAR")
    if explicit:
        return explicit if Path(explicit).is_file() else None
    jars = sorted(glob.glob(str(Path.home() / ".cache/firebase/emulators/cloud-firestore-emulator-*.jar")))
    return jars[-1] if jars else None


class Emulator:
    def __init__(self) -> None:
        jar, java = emulator_jar(), shutil.which("java")
        if jar is None or java is None:
            pytest.skip("Firestore emulator unavailable (java + cloud-firestore-emulator jar required)")
        self.port = free_port()
        self.host = f"127.0.0.1:{self.port}"
        self.proc = subprocess.Popen([java, "-jar", jar, "--host=127.0.0.1", f"--port={self.port}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 60
        while time.time() < deadline:
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", self.port)) == 0:
                    return
            time.sleep(0.2)
        self.stop()
        raise RuntimeError("Firestore emulator did not start within 60s")

    def stop(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=30)


def coordinator(host: str, namespace: str, run_key: str, **kwargs):
    from lot45.distributed.config import CoordinatorConfig
    from lot45.distributed.coordinator import ShardCoordinator, make_client

    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    config = CoordinatorConfig(project=PROJECT, namespace=namespace, emulator_host=host)
    kwargs.setdefault("log", lambda _: None)
    return ShardCoordinator(make_client(config), config, run_key, **kwargs)


def new_namespace() -> str:
    return f"t{uuid.uuid4().hex[:10]}"


def pending(n: int) -> list[dict]:
    return [{"shard_id": f"shard_{i:05d}", "index": i, "input_sha256": f"sha-{i}", "count": 1, "status": "PENDING"} for i in range(n)]


# -- cibles de processus (spawn : fonctions de module) -----------------------------
def proc_claim(host, namespace, run_key, worker_id, barrier, queue, lease_ttl_s=900.0):
    coord = coordinator(host, namespace, run_key, lease_ttl_s=lease_ttl_s)
    barrier.wait()
    lease = coord.claim_next_shard(worker_id, f"run-{worker_id}")
    queue.put((worker_id, None if lease is None else lease.as_dict()))


def proc_drain(host, namespace, run_key, worker_id, barrier, queue):
    """Reserve et complete des shards jusqu'a epuisement."""
    from lot45.distributed.coordinator import Lease

    coord = coordinator(host, namespace, run_key)
    barrier.wait()
    taken = []
    while True:
        lease = coord.claim_next_shard(worker_id, f"run-{worker_id}")
        if lease is None:
            break
        assert isinstance(lease, Lease)
        coord.complete(lease, {"files": [], "search_identity_fingerprint": "test"})
        taken.append(lease.shard_id)
    queue.put((worker_id, taken))


def proc_complete(host, namespace, run_key, lease_dict, barrier, queue):
    from lot45.distributed.coordinator import Lease

    coord = coordinator(host, namespace, run_key)
    barrier.wait()
    outcome = coord.complete(Lease(**lease_dict), {"files": [], "search_identity_fingerprint": "test"})
    queue.put((lease_dict["shard_id"], outcome["status"]))


def proc_hold(host, namespace, run_key, worker_id, queue, lease_ttl_s):
    """Reserve un shard puis 'calcule' indefiniment (tue par le test)."""

    coord = coordinator(host, namespace, run_key, lease_ttl_s=lease_ttl_s)
    lease = coord.claim_next_shard(worker_id, "run")
    queue.put(lease.as_dict())
    time.sleep(3600)


def proc_exclusive(host, namespace, run_key, holder, barrier, queue):
    from lot44.artifacts import Lot44FatalError

    coord = coordinator(host, namespace, run_key)
    barrier.wait()
    try:
        coord.acquire_exclusive("finalizer", holder, 600.0)
        queue.put((holder, "GRANTED"))
    except Lot44FatalError as exc:
        queue.put((holder, exc.code))


def proc_worker(host, namespace, out, ctx_overrides, worker_id, crash_on_claim, queue, lease_ttl_s=6.0):
    """Vrai worker MCTS (petit budget). ``crash_on_claim`` : meurt (os._exit) juste
    apres sa premiere reservation, comme un Colab coupe en plein calcul."""
    from lot45.distributed.migration import run_key
    from lot45.distributed.worker import WorkerSettings, run_worker
    from lot44.artifacts import read_json

    from lot45_test_support import make_context

    ctx = make_context(Path(out), **ctx_overrides)
    coord = coordinator(host, namespace, run_key(read_json(ctx.out / "shard_manifest.json")), lease_ttl_s=lease_ttl_s)

    def die(lease):
        queue.put((worker_id, "CRASHED", lease.shard_id))
        queue.close()
        queue.join_thread()  # vide le tampon de la file avant la mort brutale
        os._exit(137)

    settings = WorkerSettings(worker_id=worker_id, concurrency=ctx.concurrency, heartbeat_s=1.0, idle_poll_s=0.5, max_idle_polls=60, on_claimed=die if crash_on_claim else None)
    summary = run_worker(ctx, coord, settings, log=lambda _: None)
    queue.put((worker_id, summary["stop_reason"], summary["completed"]))


class ToggleProxy:
    """Proxy TCP vers l'emulateur ; ``down()`` coupe toutes les connexions et
    refuse les nouvelles (panne reseau reelle vue par gRPC)."""

    def __init__(self, target_port: int) -> None:
        self.target_port = target_port
        self.port = free_port()
        self.host = f"127.0.0.1:{self.port}"
        self.available = True
        self.sockets: list[socket.socket] = []
        self.lock = threading.Lock()
        self.server = socket.create_server(("127.0.0.1", self.port))
        threading.Thread(target=self._accept, daemon=True).start()

    def _pipe(self, src: socket.socket, dst: socket.socket) -> None:
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            return  # connexion coupee par down() : fin normale du relais
        finally:
            for s in (src, dst):
                s.close()

    def _accept(self) -> None:
        while True:
            client, _ = self.server.accept()
            if not self.available:
                client.close()
                continue
            upstream = socket.create_connection(("127.0.0.1", self.target_port))
            with self.lock:
                self.sockets += [client, upstream]
            threading.Thread(target=self._pipe, args=(client, upstream), daemon=True).start()
            threading.Thread(target=self._pipe, args=(upstream, client), daemon=True).start()

    def down(self) -> None:
        self.available = False
        with self.lock:
            for s in self.sockets:
                s.close()
            self.sockets.clear()

    def up(self) -> None:
        self.available = True
