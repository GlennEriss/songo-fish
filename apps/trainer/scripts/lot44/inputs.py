"""Preparation locale des inputs Colab (candidats OOS + checkpoints)."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

from colab_drive import atomic_copy_verified, detect_google_drive_root, ensure_layout, sha256
from run_srn_colab_benchmark import IDENTITY, engine_fingerprint, git_commit, pool_fingerprints, pool_identity

from .artifacts import atomic_write_text, canonical_hash, utc_now, write_json
from .config import CANDIDATES_FILE, INPUT_BUNDLE_NAME, MIN_LEGAL_ACTIONS, ORIGINAL_POSITIONS_FILE
from .corpus import IDENTITY_KEY, iter_source_records, load_original_reference, select_candidates


def build_candidates(source_dir: Path, original_positions: Path, output: Path, *, seed: int, lot41: Path | None, lot42: Path | None) -> dict:
    reference = load_original_reference(original_positions, lot41, lot42)
    files = sorted(source_dir.glob("part-*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no part-*.jsonl shards in {source_dir}")
    selection = select_candidates(iter_source_records(source_dir), excluded_fps=reference["fingerprints"], excluded_games=reference["games"], seed=seed)
    rows = selection["rows"]
    payload = {
        "lot": 44,
        "created_utc": utc_now(),
        "code_commit": git_commit(),
        "provenance": {"dataset": "LOT34_POOL_SELFPLAY (autonomous self-play, same generator family as the original 256)", "source_dir": str(source_dir), "teacher_labels_used": False, "policy_targets_kept": False, "value_targets_kept": False, "minimax_labels_used": False},
        "source_files": [{"name": p.name, "size": p.stat().st_size} for p in files],
        "source_files_sha256": canonical_hash([[p.name, p.stat().st_size, sha256(p)] for p in files]),
        "selection": {"seed": seed, "positions_per_game": 1, "min_legal_actions": MIN_LEGAL_ACTIONS, "excluded_original_games": len(reference["games"]), "excluded_original_states": len(reference["fingerprints"]), "lot41_consistent": reference["lot41_consistent"], "identity_key": IDENTITY_KEY},
        "stats": selection["stats"],
        "fingerprints_sha256": canonical_hash([r["fingerprint"] for r in rows]),
        "rows": rows,
    }
    write_json(output, payload)
    return {"candidates": len(rows), "stats": selection["stats"], "output": str(output), "sha256": sha256(output)}


def build_input_bundle(repo: Path, bundle_dir: Path, *, sync_to_drive: bool, drive_root: str | None) -> dict:
    identity = pool_identity()
    files = [IDENTITY, Path(identity["policy_checkpoint"]), Path(identity["value_checkpoint"]), Path(ORIGINAL_POSITIONS_FILE), Path(CANDIDATES_FILE)]
    for f in files:
        if not (repo / f).is_file():
            raise FileNotFoundError(repo / f)
    manifest = {"experiment_id": "lot44", "git_commit": git_commit(), "created_utc": utc_now(), "model_fingerprints": pool_fingerprints(), "engine_fingerprint": engine_fingerprint(), "files": {str(f): sha256(repo / f) for f in files}}
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = bundle_dir / "lot44_inputs_manifest.json"
    write_json(manifest_path, manifest)
    bundle = bundle_dir / INPUT_BUNDLE_NAME
    with tarfile.open(bundle, "w:gz") as archive:
        for f in files:
            archive.add(repo / f, arcname=str(f))
        archive.add(manifest_path, arcname="lot44_input_manifest.json")
    digest = sha256(bundle)
    checksum = bundle.with_name(bundle.name + ".sha256")
    atomic_write_text(checksum, f"{digest}  {bundle.name}\n")
    with tarfile.open(bundle, "r:gz") as archive:
        names = set(archive.getnames())
    if not {str(f) for f in files} <= names:
        raise RuntimeError("input bundle is missing files")
    result = {"bundle": str(bundle), "sha256": digest, "files": manifest["files"], "DRIVE_SYNC": "NOT_REQUESTED"}
    if sync_to_drive:
        layout = ensure_layout(detect_google_drive_root(drive_root))
        copy = atomic_copy_verified(bundle, layout["inputs"] / bundle.name)
        atomic_copy_verified(checksum, layout["inputs"] / checksum.name)
        atomic_copy_verified(manifest_path, layout["manifests"] / manifest_path.name)
        result.update({"DRIVE_SYNC": "DONE" if copy["valid"] else "CHECKSUM_MISMATCH", "drive_bundle": str(layout["inputs"] / bundle.name)})
    print(json.dumps(result, indent=2))
    return result
