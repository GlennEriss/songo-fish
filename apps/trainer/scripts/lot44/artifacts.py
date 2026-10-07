"""Ecritures atomiques, checksums, journal d'erreurs, heartbeat et bundle."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import resource
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import torch

from colab_drive import sha256

T = TypeVar("T")


class Lot44FatalError(RuntimeError):
    """Erreur qui doit stopper l'experience (section 29)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def with_retries(
    action: Callable[[], T],
    *,
    what: str,
    max_retries: int = 3,
    backoff_s: float = 2.0,
    recoverable: tuple[type[BaseException], ...] = (OSError,),
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Retry borne pour les erreurs d'E/S transitoires (Drive, verrou)."""

    attempt = 0
    while True:
        try:
            return action()
        except recoverable as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            if on_retry is not None:
                on_retry(attempt, exc)
            print(f"[Lot44][retry] {what}: {type(exc).__name__}: {exc} (tentative {attempt}/{max_retries})", flush=True)
            time.sleep(backoff_s * (2 ** (attempt - 1)))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    def write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    with_retries(write, what=f"write {path.name}")


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: Path, payload: Any) -> None:
    # allow_nan=False : un NaN dans un artefact est une erreur critique.
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    return with_retries(lambda: json.loads(path.read_text(encoding="utf-8")), what=f"read {path.name}")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict]:
    text = with_retries(lambda: path.read_text(encoding="utf-8"), what=f"read {path.name}")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    fields = fieldnames if fieldnames is not None else (list(rows[0]) if rows else [])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def read_csv(path: Path) -> list[dict]:
    text = with_retries(lambda: path.read_text(encoding="utf-8"), what=f"read {path.name}")
    return list(csv.DictReader(io.StringIO(text)))


def sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256.json")


def write_checked_json(path: Path, payload: Any) -> str:
    """JSON atomique + sidecar checksum ecrit apres le fichier principal."""

    write_json(path, payload)
    digest = sha256(path)
    write_json(sidecar(path), {"file": path.name, "sha256": digest, "size": path.stat().st_size})
    return digest


def checked_status(path: Path) -> str:
    """``MISSING`` / ``INCOMPLETE`` (pas de sidecar) / ``VALID`` / ``CORRUPT``."""

    if not path.is_file():
        return "MISSING"
    side = sidecar(path)
    if not side.is_file():
        return "INCOMPLETE"
    expected = read_json(side).get("sha256")
    return "VALID" if expected == sha256(path) else "CORRUPT"


def read_checked_json(path: Path) -> Any:
    status = checked_status(path)
    if status != "VALID":
        raise Lot44FatalError("CHECKSUM_MISMATCH" if status == "CORRUPT" else "ARTIFACT_MISSING", f"{path} is {status}")
    return read_json(path)


class ErrorLog:
    """Journal ``errors.jsonl`` : toute exception inattendue y est consignee."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(
        self,
        *,
        stage: str,
        exc: BaseException,
        shard: str | None = None,
        position_fingerprint: str | None = None,
        retry_count: int = 0,
        last_valid_checkpoint: str | None = None,
        fatal: bool = True,
    ) -> None:
        entry = {
            "timestamp": utc_now(),
            "stage": stage,
            "shard": shard,
            "position_fingerprint": position_fingerprint,
            "exception_class": type(exc).__name__,
            "error_code": getattr(exc, "code", None),
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "retry_count": retry_count,
            "last_valid_checkpoint": last_valid_checkpoint,
            "fatal": fatal,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def memory_snapshot() -> dict:
    snapshot: dict[str, Any] = {"peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)}
    if torch.cuda.is_available():
        snapshot["gpu_allocated_bytes"] = int(torch.cuda.memory_allocated())
        snapshot["gpu_peak_bytes"] = int(torch.cuda.max_memory_allocated())
    return snapshot


def total_ram_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def write_heartbeat(out: Path, *, stage: str, completed: int, total: int, current_shard: str | None, started: float, last_artifact: str | None) -> None:
    write_json(out / "heartbeat.json", {
        "timestamp": utc_now(),
        "stage": stage,
        "completed": completed,
        "total": total,
        "current_shard": current_shard,
        "elapsed_s": time.time() - started,
        "last_successful_artifact": last_artifact,
        **memory_snapshot(),
    })


def update_stage_state(out: Path, key: str, payload: dict) -> None:
    path = out / "stage_state.json"
    state = read_json(path) if path.is_file() else {"stages": {}}
    state["stages"][key] = {**payload, "updated_utc": utc_now()}
    write_json(path, state)


def append_execution(out: Path, entry: dict) -> None:
    path = out / "execution_manifest.json"
    manifest = read_json(path) if path.is_file() else {"lot": 44, "invocations": []}
    manifest["invocations"].append(entry)
    write_json(path, manifest)


CHECKSUM_EXCLUDE = ("checksums.json", "heartbeat.json", "run_lock.json", "execution_manifest.json", "experiment_manifest.json", "errors.jsonl")


def artifact_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and not p.name.startswith(".") and p.name not in CHECKSUM_EXCLUDE
    )


def build_checksums(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256(p) for p in artifact_files(root)}


def verify_checksums(root: Path, checksums: dict[str, str]) -> list[str]:
    """Retourne la liste des fichiers manquants ou modifies."""

    bad = []
    for name, digest in checksums.items():
        path = root / name
        if not path.is_file() or sha256(path) != digest:
            bad.append(name)
    return bad


def make_bundle(root: Path, bundle: Path, arc_prefix: str) -> dict:
    """Archive tar.gz de ``root`` puis verification complete par relecture."""

    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle.with_name(f".{bundle.name}.tmp-{os.getpid()}")
    with tarfile.open(temporary, "w:gz") as archive:
        for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")):
            archive.add(path, arcname=f"{arc_prefix}/{path.relative_to(root)}")
    os.replace(temporary, bundle)
    digest = sha256(bundle)
    atomic_write_text(bundle.with_name(bundle.name + ".sha256"), f"{digest}  {bundle.name}\n")
    return {"bundle": str(bundle), "sha256": digest, **verify_bundle(bundle, arc_prefix)}


def verify_bundle(bundle: Path, arc_prefix: str) -> dict:
    expected = bundle.with_name(bundle.name + ".sha256").read_text(encoding="utf-8").split()[0]
    if expected != sha256(bundle):
        raise Lot44FatalError("CHECKSUM_MISMATCH", f"bundle checksum mismatch: {bundle}")
    with tarfile.open(bundle, "r:gz") as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        manifest_name = f"{arc_prefix}/checksums.json"
        if manifest_name not in members:
            raise Lot44FatalError("ARTIFACT_MISSING", f"checksums.json absent from {bundle}")
        checksums = json.loads(archive.extractfile(members[manifest_name]).read().decode("utf-8"))
        bad = []
        for name, digest in checksums.items():
            member = members.get(f"{arc_prefix}/{name}")
            if member is None or hashlib.sha256(archive.extractfile(member).read()).hexdigest() != digest:
                bad.append(name)
    if bad:
        raise Lot44FatalError("CHECKSUM_MISMATCH", f"{len(bad)} bundle members fail checksum: {bad[:5]}")
    return {"members": len(members), "checksummed_members": len(checksums), "bundle_valid": True}
