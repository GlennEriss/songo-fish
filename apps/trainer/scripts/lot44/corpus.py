"""Corpus OOS independant des 256 positions des Lots 41-43.

Identite physique d'un etat (section 42) : ``board`` complet (14 cases + 2
magasins) et ``player_to_move``. Le moteur ``SongoLegacyGame`` n'a aucun autre
etat (pas d'historique ni de compteur de coups) : cette cle est exactement
celle de ``run_srn_lot39.fingerprint_state``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from songo_ai.dataset import RawSongoState
from songo_ai.songo.rules import SongoLegacyGame
from run_srn_lot39 import fingerprint_state

from .artifacts import Lot44FatalError, canonical_hash, read_json
from .config import MIN_LEGAL_ACTIONS

IDENTITY_KEY = {"fields": ["board[0..15] (14 pits + 2 stores)", "player_to_move"], "function": "run_srn_lot39.fingerprint_state", "canonicalization": False}


def state_from_dict(state: dict) -> RawSongoState:
    return RawSongoState(tuple(int(x) for x in state["board"]), int(state["player_to_move"]))


def legal_action_count(state: RawSongoState) -> int | None:
    """Nombre d'actions legales, ``None`` si l'etat est terminal."""

    game = SongoLegacyGame.from_state(state.to_engine_state())
    game.normalize_terminal()
    if game.finished:
        return None
    return len(game.legal_local_actions())


def eligible(state: RawSongoState) -> bool:
    count = legal_action_count(state)
    return count is not None and count >= MIN_LEGAL_ACTIONS


def load_original_reference(positions_file: Path, lot41: Path | None = None, lot42: Path | None = None) -> dict:
    """Empreintes et parties des 256 positions des Lots 41-43."""

    payload = read_json(positions_file)
    states = payload["states"]
    fps = {fingerprint_state(state_from_dict(s)) for s in states}
    games = {str(s["source_game_id"]) for s in states if s.get("source_game_id")}
    result = {"positions_file": str(positions_file), "fingerprints": fps, "games": games, "lot41_consistent": None, "lot42_fingerprints": 0}
    if len(fps) != 256:
        raise Lot44FatalError("CORPUS_CORRUPT", f"original reference must contain 256 unique states, got {len(fps)}")
    if lot41 is not None:
        lot41_rows = read_json(lot41 / "search" / "budget_32768.json")["rows"]
        lot41_fps = {r["state_fingerprint"] for r in lot41_rows}
        if lot41_fps != fps:
            raise Lot44FatalError("CORPUS_CORRUPT", "Lot39 positions file does not match the Lot41 MCTS32768 state set")
        result["lot41_consistent"] = True
    if lot42 is not None:
        lot42_fps = {r["state_fingerprint"] for r in read_json(lot42 / "search_65536_results.json")["rows"]}
        if not lot42_fps <= fps:
            raise Lot44FatalError("CORPUS_CORRUPT", "Lot42 65536 positions are not a subset of the original 256")
        result["lot42_fingerprints"] = len(lot42_fps)
    return result


def iter_source_records(source_dir: Path) -> Iterable[dict]:
    for path in sorted(source_dir.glob("part-*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("record_type") == "manifest":
                    continue
                meta = row.get("metadata") or {}
                yield {
                    "state": row["state"],
                    "game_id": meta.get("game_id"),
                    "ply": meta.get("ply"),
                    "source_shard": path.name,
                    "source_line": line_number,
                    "generator": {k: meta.get(k) for k in ("p1_model_id", "p2_model_id", "checkpoint_id", "generation_id", "source_type")},
                }


def select_candidates(records: Iterable[dict], *, excluded_fps: set[str], excluded_games: set[str], seed: int) -> dict:
    """Une position eligible par partie, choisie de facon deterministe.

    Une seule position par partie supprime par construction la fuite par
    trajectoire ; les doublons physiques entre parties sont ensuite retires.
    Aucune cible (policy/value/visites) n'est conservee.
    """

    stats = {"records": 0, "missing_game_id": 0, "excluded_original_game_records": 0, "excluded_original_state_records": 0, "ineligible_records": 0}
    per_game: dict[str, list[dict]] = {}
    for record in records:
        stats["records"] += 1
        game = record["game_id"]
        if not game:
            stats["missing_game_id"] += 1
            continue
        if game in excluded_games:
            stats["excluded_original_game_records"] += 1
            continue
        state = state_from_dict(record["state"])
        fp = fingerprint_state(state)
        if fp in excluded_fps:
            stats["excluded_original_state_records"] += 1
            continue
        if not eligible(state):
            stats["ineligible_records"] += 1
            continue
        per_game.setdefault(game, []).append({**record, "state": {"board": list(state.board), "player_to_move": state.player_to_move}, "fingerprint": fp})
    chosen = []
    for game in sorted(per_game):
        options = sorted(per_game[game], key=lambda r: (r["ply"] if r["ply"] is not None else -1, r["source_line"]))
        chosen.append(options[int(canonical_hash({"seed": seed, "game": game}), 16) % len(options)])
    by_fp: dict[str, list[dict]] = {}
    for row in chosen:
        by_fp.setdefault(row["fingerprint"], []).append(row)
    rows = []
    for fp, group in by_fp.items():
        rows.append(min(group, key=lambda r: canonical_hash({"seed": seed, "game": r["game_id"]})))
    rows.sort(key=lambda r: r["fingerprint"])
    stats.update({"games_with_eligible_positions": len(per_game), "cross_game_physical_duplicates_dropped": len(chosen) - len(rows), "candidates": len(rows)})
    return {"rows": rows, "stats": stats}


def build_corpus(candidates: list[dict], reference: dict, *, max_positions: int, seed: int) -> dict:
    """Recontrole chaque candidat puis applique le plafond de calcul deterministe."""

    seen_fp: set[str] = set()
    seen_game: set[str] = set()
    removed_overlap_fp, removed_overlap_game = [], []
    clean = []
    for row in candidates:
        state = state_from_dict(row["state"])
        fp = fingerprint_state(state)
        if fp != row["fingerprint"]:
            raise Lot44FatalError("CORPUS_CORRUPT", f"fingerprint mismatch for candidate {row['fingerprint'][:12]}")
        if not eligible(state):
            raise Lot44FatalError("CORPUS_CORRUPT", f"ineligible candidate {fp[:12]}")
        if fp in seen_fp or row["game_id"] in seen_game:
            raise Lot44FatalError("CORPUS_CORRUPT", f"duplicate state or game in candidates: {fp[:12]}")
        seen_fp.add(fp)
        seen_game.add(row["game_id"])
        if fp in reference["fingerprints"]:
            removed_overlap_fp.append(fp)
            continue
        if row["game_id"] in reference["games"]:
            removed_overlap_game.append(row["game_id"])
            continue
        clean.append(row)
    ordered = sorted(clean, key=lambda r: canonical_hash({"seed": seed, "fp": r["fingerprint"]}))
    selected = sorted(ordered[:max_positions], key=lambda r: r["fingerprint"])
    if not selected:
        raise Lot44FatalError("CORPUS_CORRUPT", "independent corpus is empty")
    final_fps = {r["fingerprint"] for r in selected}
    final_games = {r["game_id"] for r in selected}
    overlap = {
        "OOS_OVERLAP_WITH_ORIGINAL_256": len(final_fps & reference["fingerprints"]),
        "OOS_GAME_OVERLAP_WITH_ORIGINAL_256": len(final_games & reference["games"]),
        "candidate_state_overlap_removed": len(removed_overlap_fp),
        "candidate_game_overlap_removed": len(removed_overlap_game),
        "removed_fingerprints": sorted(removed_overlap_fp),
        "reference": {"positions_file": reference["positions_file"], "original_states": len(reference["fingerprints"]), "original_games": len(reference["games"]), "lot41_consistent": reference["lot41_consistent"], "lot42_fingerprints": reference["lot42_fingerprints"]},
        "identity_key": IDENTITY_KEY,
    }
    return {"rows": selected, "available": len(clean), "overlap": overlap}
