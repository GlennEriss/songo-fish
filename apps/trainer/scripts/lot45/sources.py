"""Registre des sources de positions et normalisation des occurrences.

Chaque parser produit des occurrences ne contenant QUE ``OCCURRENCE_FIELDS``
(liste blanche) : etat physique, partie, ply, z, generateur et ancienne cible
autonome MCTS (comparaison diagnostique). Les champs teacher historiques
(best_action, action_values, PV, wdl, regrets...) ne sont jamais lus.

z est exprime du point de vue du joueur au trait (verifie sur Lot34 :
``value_target == z`` pour 100 % des exemples D_RL).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from songo_ai.dataset import RawSongoState
from run_srn_lot39 import fingerprint_state

from .config import OCCURRENCE_FIELDS


@dataclass(frozen=True)
class Source:
    name: str
    family: str
    pattern: str
    fmt: str
    priority: int
    description: str


SOURCES = (
    Source("D_REAL_HUMAN_MATCHES", "D_REAL_HUMAN", "data/real_matches/match_moves_v1.jsonl", "real", 1, "human online matches (positions only; human moves are never labels)"),
    Source("LOT34_POOL_DATA", "G4_TRAINING_CROSSPLAY", "data/experiments/lot34_g4_training/pool_data/part-*.jsonl", "d_rl", 2, "Lot34 POOL self-/cross-play (G2/G3 generators), G4 training data"),
    Source("LOT34_CONTROL_DATA", "G4_TRAINING_CROSSPLAY", "data/experiments/lot34_g4_training/control_data/part-*.jsonl", "d_rl", 2, "Lot34 CONTROL self-play (G2), G4 training data"),
    Source("LOT32_CROSSPLAY_ENRICHED", "G3_CROSSPLAY_SCALING", "data/experiments/lot32_crossplay_scaling/crossplay_enriched/part-*.jsonl", "d_rl", 3, "Lot32 G2/G3 cross-play"),
    Source("LOT32_ORIGINAL_POOL", "G3_CROSSPLAY_SCALING", "data/experiments/lot32_crossplay_scaling/original_pool/part-*.jsonl", "d_rl", 3, "Lot32 original pool"),
    Source("LOT32_CONTROL", "G3_CROSSPLAY_SCALING", "data/experiments/lot32_crossplay_scaling/control/part-*.jsonl", "d_rl", 3, "Lot32 G2 control"),
    Source("D_SCALE_REANALYSIS_LARGE", "D_TEACHER_STATES_VIA_REANALYSIS", "data/d_scale_v1/d_reanalysis_large/part-*.jsonl", "reanalysis", 4, "D_TEACHER physical states (position-only reservoir), old G2 MCTS128 targets"),
    Source("LOT31_POOL", "G3_GENERATOR_POOL", "data/experiments/lot31_generator_pool/pool/part-*.jsonl", "d_rl", 4, "Lot31 generator pool"),
    Source("LOT31_CONTROL", "G3_GENERATOR_POOL", "data/experiments/lot31_generator_pool/control/part-*.jsonl", "d_rl", 4, "Lot31 control"),
    Source("D_SCALE_SELFPLAY_LARGE", "G2_SELFPLAY", "data/d_scale_v1/d_selfplay_large/part-*.jsonl", "d_rl", 5, "G2 self-play (d_scale_v1)"),
    Source("LOT34_COMMON_REANALYSIS", "AUTONOMOUS_REANALYSIS_STATES", "data/experiments/lot34_g4_training/common_reanalysis/*.jsonl", "reanalysis", 5, "Lot34 common autonomous reanalysis states"),
    Source("D_RL_G1", "G1_SELFPLAY", "data/d_rl/*.jsonl", "d_rl", 6, "early D_RL (G0/G1 self-play)"),
)

NOT_USED = {
    "D_TEACHER_RAW_FILES": "dataset_v00x / full_matrix position files are not present locally (only manifests; see lot18 inventory). Their physical states are reached through D_SCALE_REANALYSIS_LARGE (position-only reservoir).",
    "ARENA_GAMES_LOT12_LOT14": "action sequences only (no stored states); positions would require replay and duplicate the G2/G3 self-play families",
    "QDIAG_AND_SEARCH_CACHES": "diagnostic caches (qdiag, regrets, search traces) keyed by hash: not position sources and contain value diagnostics",
    "PILOT_DATASETS": "tiny historical pilots with teacher annotations",
}


def _z(winner, player_to_move: int) -> float | None:
    if winner is None:
        return None
    winner = int(winner)
    return 0.0 if winner == 0 else (1.0 if winner == player_to_move else -1.0)


def _occurrence(**values) -> dict:
    occurrence = {k: values.get(k) for k in OCCURRENCE_FIELDS}
    state = occurrence["state"]
    raw = RawSongoState(tuple(int(x) for x in state["board"]), int(state["player_to_move"]))
    occurrence["state"] = {"board": list(raw.board), "player_to_move": raw.player_to_move}
    occurrence["fingerprint"] = fingerprint_state(raw)
    return occurrence


def from_d_rl(source: Source, row: dict, context: dict) -> dict | None:
    if row.get("record_type") == "manifest":
        return None
    meta = row.get("metadata") or {}
    state = row["state"]
    terminal = str(meta.get("status", "")).startswith("TERMINAL")
    return _occurrence(
        state=state, source=source.name, family=source.family, game_id=meta.get("game_id"), ply=meta.get("ply"),
        z=_z(meta.get("winner"), int(state["player_to_move"])) if terminal else None,
        generator={k: meta.get(k) for k in ("checkpoint_id", "generation", "acting_model_id", "source_type")},
        old_target={"visit_counts": row.get("visit_counts"), "budget": meta.get("mcts_simulations"), "root_noise": meta.get("root_noise"), "kind": "SELFPLAY_SEARCH"},
    )


def from_reanalysis(source: Source, row: dict, context: dict) -> dict:
    return _occurrence(
        state=row["state"], source=source.name, family=source.family, game_id=None, ply=None, z=None,
        generator={"generation_model": row.get("generation_model"), "source_position_reservoir": row.get("source_position_reservoir")},
        old_target={"visit_counts": row.get("visit_counts"), "budget": row.get("mcts_budget"), "root_noise": row.get("dirichlet"), "kind": "REANALYSIS"},
    )


def real_context(rows: list[dict]) -> dict:
    final: dict[str, dict] = {}
    for row in rows:
        if row["match_id"] not in final or row["ply"] >= final[row["match_id"]]["ply"]:
            final[row["match_id"]] = row
    return {"final": final}


def from_real(source: Source, row: dict, context: dict) -> dict:
    player = int(row["player_position"])
    last = context["final"][row["match_id"]]
    winner = last.get("winner_after") if last.get("finished_after") else None
    return _occurrence(
        state={"board": row["state"], "player_to_move": player}, source=source.name, family=source.family, game_id=f"real:{row['match_id']}",
        ply=row.get("ply"), z=_z(winner, player), generator={"origin": "human"}, old_target=None,
    )


PARSERS: dict[str, Callable[[Source, dict, dict], dict | None]] = {"d_rl": from_d_rl, "reanalysis": from_reanalysis, "real": from_real}


def source_files(repo: Path, source: Source) -> list[Path]:
    return sorted(p for p in repo.glob(source.pattern) if p.is_file() and p.stat().st_size > 0)


def iter_source(repo: Path, source: Source, on_error: Callable[[dict], None]) -> Iterator[dict]:
    """Occurrences valides ; chaque ligne invalide est signalee (jamais ignoree)."""

    parser = PARSERS[source.fmt]
    for path in source_files(repo, source):
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = []
        for number, line in enumerate(lines, 1):
            if line.strip():
                try:
                    rows.append((number, json.loads(line)))
                except ValueError as exc:
                    on_error({"source": source.name, "file": str(path.relative_to(repo)), "line": number, "reason": f"JSON: {exc}"})
        context = real_context([r for _, r in rows]) if source.fmt == "real" else {}
        for number, row in rows:
            try:
                occurrence = parser(source, row, context)
            except (ValueError, KeyError, TypeError) as exc:
                on_error({"source": source.name, "file": str(path.relative_to(repo)), "line": number, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            if occurrence is not None:
                yield occurrence
