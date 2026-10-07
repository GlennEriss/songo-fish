"""Preparation LOCALE (les sources font ~4 Go et ne sont pas sur Drive) :
audit des sources, deduplication, selection ordonnee, rapports et bundle."""
from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

from colab_drive import atomic_copy_verified, detect_google_drive_root, ensure_layout, sha256
from run_srn_colab_benchmark import IDENTITY, engine_fingerprint, git_commit, pool_fingerprints, pool_identity
from run_srn_lot39 import fingerprint_state

from lot44.artifacts import Lot44FatalError, atomic_write_text, canonical_hash, read_json, utc_now, write_json
from lot44.corpus import state_from_dict

from .config import FAMILY_WEIGHTS, INPUT_BUNDLE_NAME, MAX_SELECTION, PROBE_POSITIONS_FILE, SELECTION_FILE, SELECTION_MANIFEST_FILE
from .selection import composition, coverage_report, deduplication_report, detail_selected, eligibility, ordered_selection, scan, selection_rows, write_selection
from .sources import NOT_USED, SOURCES, source_files

REPORTS_DIR = "data/colab_bridge/lot45_reports"
REPORT_NAMES = ("candidate_source_audit.json", "source_selection.json", "source_manifest.json", "deduplication_report.json", "coverage_report.json")


def diagnostic_exclusions(repo: Path, lot44: Path) -> dict:
    """Etats et parties des Lots41-44 (benchmarks) : jamais en entrainement."""

    originals = read_json(repo / PROBE_POSITIONS_FILE)["states"]
    corpus = read_json(lot44 / "independent_corpus_manifest.json")["rows"]
    fps = {fingerprint_state(state_from_dict(s)) for s in originals} | {r["fingerprint"] for r in corpus}
    games = {str(s["source_game_id"]) for s in originals if s.get("source_game_id")} | {r["game_id"] for r in corpus}
    if len(fps) != 256 + len(corpus):
        raise Lot44FatalError("CORPUS_CORRUPT", "Lot41-44 diagnostic sets overlap unexpectedly")
    return {"fingerprints": fps, "games": games, "lot41_43_states": 256, "lot44_states": len(corpus), "games_excluded": len(games)}


def build_selection(repo: Path, lot44: Path, *, seed: int, total: int = MAX_SELECTION, log=print) -> dict:
    exclusions = diagnostic_exclusions(repo, lot44)
    invalid: list[dict] = []
    states, audit = scan(repo, excluded_games=exclusions["games"], on_error=invalid.append, log=log)
    eligible, reasons = eligibility(states, exclusions["fingerprints"])
    log(f"[Lot45] eligible unique states: {len(eligible)}")
    order, allocation = ordered_selection(states, eligible, total=total, seed=seed)
    details = detail_selected(repo, set(order), on_error=invalid.append)
    rows = selection_rows(order, states, eligible, details, seed=seed)
    selection_sha = write_selection(repo / SELECTION_FILE, rows)
    reports = Path(repo / REPORTS_DIR)
    reports.mkdir(parents=True, exist_ok=True)
    available_by_family = {}
    for fp in eligible:
        family = SOURCES[states[fp].primary].family
        available_by_family[family] = available_by_family.get(family, 0) + 1
    write_json(reports / "candidate_source_audit.json", {"sources": audit, "not_used": NOT_USED, "invalid_records": len(invalid), "invalid_examples": invalid[:20], "d_teacher_note": "raw D_TEACHER files absent locally; ~100k of its physical states are reachable via D_SCALE_REANALYSIS_LARGE (historical teacher labels never read)"})
    write_json(reports / "source_selection.json", {
        "seed": seed, "max_selection": total, "family_weights": FAMILY_WEIGHTS, "eligible_unique_states": len(eligible),
        "eligible_by_primary_family": available_by_family, "ineligible": dict(reasons), "exclusions": {k: v for k, v in exclusions.items() if k not in ("fingerprints", "games")},
        **allocation, "selected": len(rows), "ordering": "family-interleaved so that every prefix (10k/25k/50k/100k) keeps the family shares",
        "composition_full": composition(rows), "composition_prefix": {str(n): composition(rows[:n]) for n in (10_000, 25_000, 50_000) if n <= len(rows)},
        "primary_family_rule": "a state shared by several sources belongs to its highest-priority source family",
    })
    write_json(reports / "source_manifest.json", {"files": {str(p.relative_to(repo)): {"size": p.stat().st_size, "sha256": sha256(p)} for s in SOURCES for p in source_files(repo, s)}, "created_utc": utc_now()})
    write_json(reports / "deduplication_report.json", deduplication_report(states))
    write_json(reports / "coverage_report.json", coverage_report(rows))
    manifest = {"created_utc": utc_now(), "code_commit": git_commit(), "seed": seed, "rows": len(rows), "fingerprints_sha256": selection_sha, "file": SELECTION_FILE, "file_sha256": sha256(repo / SELECTION_FILE), "reports": {n: sha256(reports / n) for n in REPORT_NAMES}}
    write_json(repo / SELECTION_MANIFEST_FILE, manifest)
    return {"selected": len(rows), "eligible": len(eligible), "unique_states": len(states), "invalid_records": len(invalid), "allocation": allocation["allocation"]}


def install_reports(repo: Path, out: Path) -> dict:
    """Copie les rapports d'inputs dans le dossier d'experience apres verification."""

    manifest = read_json(repo / SELECTION_MANIFEST_FILE)
    if sha256(repo / SELECTION_FILE) != manifest["file_sha256"]:
        raise Lot44FatalError("CHECKSUM_MISMATCH", "selection file differs from its manifest")
    out.mkdir(parents=True, exist_ok=True)
    for name, digest in manifest["reports"].items():
        source = repo / REPORTS_DIR / name
        if sha256(source) != digest:
            raise Lot44FatalError("CHECKSUM_MISMATCH", f"input report {name} corrupted")
        target = out / name
        if not target.is_file() or sha256(target) != digest:
            shutil.copyfile(source, target)
    write_json(out / "selection_manifest.json", manifest)
    return {"reports": len(manifest["reports"]), "selection_rows": manifest["rows"]}


def build_input_bundle(repo: Path, bundle_dir: Path, *, sync_to_drive: bool, drive_root: str | None) -> dict:
    identity = pool_identity()
    manifest_data = read_json(repo / SELECTION_MANIFEST_FILE)
    files = [IDENTITY, Path(identity["policy_checkpoint"]), Path(identity["value_checkpoint"]), Path(PROBE_POSITIONS_FILE), Path(SELECTION_FILE), Path(SELECTION_MANIFEST_FILE)] + [Path(REPORTS_DIR) / n for n in manifest_data["reports"]]
    manifest = {"experiment_id": "lot45", "git_commit": git_commit(), "created_utc": utc_now(), "model_fingerprints": pool_fingerprints(), "engine_fingerprint": engine_fingerprint(), "files": {str(f): sha256(repo / f) for f in files}}
    manifest_path = bundle_dir / "lot45_inputs_manifest.json"
    write_json(manifest_path, manifest)
    bundle = bundle_dir / INPUT_BUNDLE_NAME
    with tarfile.open(bundle, "w:gz") as archive:
        for f in files:
            archive.add(repo / f, arcname=str(f))
        archive.add(manifest_path, arcname="lot45_input_manifest.json")
    with tarfile.open(bundle, "r:gz") as archive:
        if not {str(f) for f in files} <= set(archive.getnames()):
            raise Lot44FatalError("ARTIFACT_MISSING", "input bundle incomplete")
    digest = sha256(bundle)
    checksum = bundle.with_name(bundle.name + ".sha256")
    atomic_write_text(checksum, f"{digest}  {bundle.name}\n")
    result = {"bundle": str(bundle), "sha256": digest, "size_mb": bundle.stat().st_size / 1e6, "files": len(files), "DRIVE_SYNC": "NOT_REQUESTED", "manifest_sha256": canonical_hash(manifest["files"])}
    if sync_to_drive:
        layout = ensure_layout(detect_google_drive_root(drive_root))
        copy = atomic_copy_verified(bundle, layout["inputs"] / bundle.name)
        atomic_copy_verified(checksum, layout["inputs"] / checksum.name)
        atomic_copy_verified(manifest_path, layout["manifests"] / manifest_path.name)
        result.update({"DRIVE_SYNC": "DONE" if copy["valid"] else "CHECKSUM_MISMATCH", "drive_bundle": str(layout["inputs"] / bundle.name)})
    return result
