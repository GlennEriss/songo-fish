"""ARTIFACT_STORAGE : tentatives isolees sur Drive et references immuables.

Chaque tentative ecrit uniquement sous ``attempts/{shard_id}/{lease_id}/`` :
deux workers n'ecrivent jamais le meme fichier.  Une reference d'artefact
(enregistree par le coordinateur a la transition COMPLETED) liste les fichiers
par chemin relatif au dossier d'experience + sha256 ; le finalizer ne lit que
ces fichiers, apres reverification.
"""
from __future__ import annotations

from pathlib import Path

from colab_drive import sha256
from lot44.artifacts import Lot44FatalError, checked_status, read_json, sidecar
from lot44.search import load_completed, shard_dir

from ..dataset import validate_row
from .config import ATTEMPTS_DIR


def attempt_root(out: Path, shard_id: str, lease_id: str) -> Path:
    return out / ATTEMPTS_DIR / shard_id / lease_id


def shard_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("shard_*.json") if not p.name.endswith(".sha256.json")) if directory.is_dir() else []


def file_ref(out: Path, path: Path) -> dict:
    status = checked_status(path)
    if status != "VALID":
        raise Lot44FatalError("CHECKSUM_MISMATCH" if status == "CORRUPT" else "ARTIFACT_INCOMPLETE", f"{path.relative_to(out)} is {status}")
    return {"path": str(path.relative_to(out)), "sha256": read_json(sidecar(path))["sha256"], "size": path.stat().st_size}


def validate_attempt(out: Path, root: Path, shard_id: str, budget: int, identity_fp: str, records: list[dict], selection: dict[str, dict]) -> dict:
    """Verifie une tentative terminee et retourne sa reference d'artefact.

    Rejette : fichier sans sidecar (ecriture interrompue), checksum faux,
    identite MCTS differente, positions manquantes/en trop, cible invalide.
    """

    directory = shard_dir(root, shard_id, budget)
    files = shard_files(directory)
    refs = [file_ref(out, p) for p in files]
    expected = {r["fingerprint"] for r in records}
    rows, _ = load_completed(directory, identity_fp, expected, remove_incomplete=False)
    if set(rows) != expected:
        raise Lot44FatalError("ARTIFACT_INCOMPLETE", f"{shard_id}: attempt holds {len(rows)}/{len(expected)} positions")
    problems = [(fp[:12], p) for fp, row in rows.items() for p in validate_row(row, selection[fp], budget)]
    if problems:
        raise Lot44FatalError("TARGET_INVALID", f"{shard_id}: {len(problems)} invalid targets, first {problems[:3]}")
    payloads = [read_json(p) for p in files]
    return {
        "files": refs, "rows": len(rows), "devices": sorted({p["device"] for p in payloads}),
        "wall_time_s": sum(p["wall_time_s"] for p in payloads), "search_identity_fingerprint": identity_fp,
        "attempt_dir": str(root.relative_to(out)),
    }


def load_referenced(out: Path, refs: list[dict], identity_fp: str, expected: set[str], *, partition: str) -> tuple[dict[str, dict], list[Path]]:
    """Charge uniquement les fichiers references, apres reverification sha256."""

    rows: dict[str, dict] = {}
    paths = []
    for ref in refs:
        path = (out / ref["path"]).resolve()
        if out.resolve() not in path.parents:
            raise Lot44FatalError("ARTIFACT_INCONSISTENT", f"artifact path escapes the experiment: {ref['path']}")
        if checked_status(path) != "VALID" or sha256(path) != ref["sha256"]:
            raise Lot44FatalError("CHECKSUM_MISMATCH", f"{ref['path']}: content differs from the coordinator record")
        payload = read_json(path)
        if payload.get("search_identity_fingerprint") != identity_fp:
            raise Lot44FatalError("MCTS_CONFIG_INCOMPATIBLE", f"{ref['path']} was produced with another search identity")
        if payload.get("status") != "COMPLETE" or payload.get("partition") != partition:
            raise Lot44FatalError("ARTIFACT_INCONSISTENT", f"{ref['path']}: status {payload.get('status')} partition {payload.get('partition')} (expected {partition})")
        for row in payload["rows"]:
            fp = row["state_fingerprint"]
            if fp not in expected:
                raise Lot44FatalError("SPLIT_MODIFIED", f"{ref['path']} contains {fp[:12]} outside {partition}")
            if fp in rows:
                raise Lot44FatalError("ARTIFACT_INCONSISTENT", f"{fp[:12]} present twice in {partition}")
            rows[fp] = row
        paths.append(path)
    return rows, paths


def unreferenced_attempts(out: Path, accepted: dict[str, str | None]) -> list[str]:
    """Tentatives presentes sur Drive mais non retenues (conservees pour audit)."""

    base = out / ATTEMPTS_DIR
    if not base.is_dir():
        return []
    return sorted(str(d.relative_to(out)) for shard in base.iterdir() if shard.is_dir() for d in shard.iterdir() if d.is_dir() and accepted.get(shard.name) != d.name)
