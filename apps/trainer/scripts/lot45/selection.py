"""Deduplication physique, comptage d'occurrences, exclusions et ordre stratifie.

Deux passes sur les sources :
1. agregat compact par etat physique (occurrences, sources, z, partie, ply) ;
2. pour les seuls etats selectionnes : detail des sources, parties et
   anciennes cibles autonomes (comparaison diagnostique).

L'ordre final est entrelace par famille : tout prefixe (10k, 25k, 50k...) est
lui-meme un echantillon stratifie, ce qui permet d'arreter la generation a un
palier sans desequilibrer le corpus.
"""
from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Callable

from lot44.artifacts import Lot44FatalError, canonical_hash
from lot44.corpus import legal_action_count

from .config import FAMILY_WEIGHTS, HOLDOUT_FRACTION, MAX_POSITIONS_PER_GAME, MIN_LEGAL_ACTIONS
from .sources import SOURCES, Source, iter_source

SOURCE_INDEX = {s.name: i for i, s in enumerate(SOURCES)}
TRAINING_CORPORA = {
    "G4_TRAINING": ("LOT34_POOL_DATA", "LOT34_CONTROL_DATA", "LOT34_COMMON_REANALYSIS"),
    "G3_TRAINING": ("D_SCALE_SELFPLAY_LARGE", "D_SCALE_REANALYSIS_LARGE"),
    "G2_TRAINING_PROXY": ("D_RL_G1",),
    "PREVIOUS_AUTONOMOUS_REANALYSIS": ("D_SCALE_REANALYSIS_LARGE", "LOT34_COMMON_REANALYSIS"),
}


def phase_bin(seeds_in_play: int) -> str:
    return "opening_56_70" if seeds_in_play >= 56 else "middle_36_55" if seeds_in_play >= 36 else "late_16_35" if seeds_in_play >= 16 else "endgame_0_15"


class Aggregate:
    """Agregat compact par etat physique (passe 1)."""

    __slots__ = ("board", "player", "occurrences", "source_mask", "primary", "z_win", "z_draw", "z_loss", "z_unknown", "group", "min_ply", "excluded_game")

    def __init__(self, board: tuple, player: int, primary: int) -> None:
        self.board, self.player, self.primary = board, player, primary
        self.occurrences = self.source_mask = self.z_win = self.z_draw = self.z_loss = self.z_unknown = 0
        self.group = None
        self.min_ply = None
        self.excluded_game = False


def scan(repo: Path, *, excluded_games: set[str], on_error: Callable[[dict], None], sources=SOURCES, log=print) -> tuple[dict[str, Aggregate], dict]:
    states: dict[str, Aggregate] = {}
    audit: dict[str, dict] = {}
    for source in sources:
        index = SOURCE_INDEX[source.name]
        stats = Counter()
        games, fps = set(), set()
        budgets, noise = Counter(), Counter()
        for occ in iter_source(repo, source, lambda e: (stats.update(["invalid_records"]), on_error(e))):
            stats["occurrences"] += 1
            fp = occ["fingerprint"]
            fps.add(fp)
            agg = states.get(fp)
            if agg is None:
                agg = states[fp] = Aggregate(tuple(occ["state"]["board"]), occ["state"]["player_to_move"], index)
            agg.occurrences += 1
            agg.source_mask |= 1 << index
            if SOURCES[index].priority < SOURCES[agg.primary].priority:
                agg.primary = index
            z = occ["z"]
            if z is None:
                agg.z_unknown += 1
            elif z > 0:
                agg.z_win += 1
            elif z < 0:
                agg.z_loss += 1
            else:
                agg.z_draw += 1
            stats["with_z"] += z is not None
            game = occ["game_id"]
            if game:
                games.add(game)
                if agg.group is None or index == agg.primary:
                    agg.group = game
                if game in excluded_games:
                    agg.excluded_game = True
            if occ["ply"] is not None:
                stats["with_ply"] += 1
                agg.min_ply = occ["ply"] if agg.min_ply is None else min(agg.min_ply, occ["ply"])
            if occ["old_target"]:
                budgets[str(occ["old_target"]["budget"])] += 1
                noise[str(occ["old_target"]["root_noise"])] += 1
        audit[source.name] = {
            "family": source.family, "priority": source.priority, "format": source.fmt, "description": source.description,
            "files": len(list(repo.glob(source.pattern))), "occurrences": stats["occurrences"], "invalid_records": stats["invalid_records"],
            "unique_states": len(fps), "games": len(games), "z_available_rate": stats["with_z"] / stats["occurrences"] if stats["occurrences"] else None,
            "ply_available_rate": stats["with_ply"] / stats["occurrences"] if stats["occurrences"] else None,
            "old_target_budgets": dict(budgets), "old_target_root_noise": dict(noise), "status": "AVAILABLE" if stats["occurrences"] else "EMPTY",
        }
        log(f"[Lot45][scan] {source.name}: {stats['occurrences']} occurrences, {len(fps)} unique states, {len(games)} games")
    return states, audit


def characterize(agg: Aggregate) -> dict:
    from songo_ai.dataset import RawSongoState

    legal = legal_action_count(RawSongoState(agg.board, agg.player))
    seeds = sum(agg.board[:14])
    return {"legal_count": legal, "seeds_in_play": seeds, "phase": phase_bin(seeds)}


def eligibility(states: dict[str, Aggregate], excluded_fps: set[str]) -> tuple[dict[str, dict], Counter]:
    eligible: dict[str, dict] = {}
    reasons = Counter()
    for fp, agg in states.items():
        info = characterize(agg)
        if info["legal_count"] is None:
            reasons["terminal"] += 1
        elif info["legal_count"] < MIN_LEGAL_ACTIONS:
            reasons["single_legal_action"] += 1
        elif fp in excluded_fps:
            reasons["lot41_44_diagnostic_state"] += 1
        elif agg.excluded_game:
            reasons["game_of_lot41_44_diagnostic_state"] += 1
        else:
            eligible[fp] = info
    return eligible, reasons


def allocate(available: dict[str, int], total: int) -> dict[str, int]:
    """Quotas par famille ; reliquat redistribue dans l'ordre de priorite."""

    alloc = {f: 0 for f in FAMILY_WEIGHTS}
    fixed = {f: available.get(f, 0) for f, w in FAMILY_WEIGHTS.items() if w is None}
    alloc.update(fixed)
    remaining = total - sum(fixed.values())
    weighted = {f: w for f, w in FAMILY_WEIGHTS.items() if w is not None}
    open_families = dict(weighted)
    while remaining > 0 and open_families:
        scale = sum(open_families.values())
        progress = 0
        for family, weight in list(open_families.items()):
            want = math.floor(remaining * weight / scale) or 1
            take = min(want, available.get(family, 0) - alloc[family], remaining - progress)
            alloc[family] += max(take, 0)
            progress += max(take, 0)
            if alloc[family] >= available.get(family, 0):
                del open_families[family]
            if progress >= remaining:
                break
        remaining -= progress
        if progress == 0:
            break
    return alloc


def ordered_selection(states: dict[str, Aggregate], eligible: dict[str, dict], *, total: int, seed: int) -> tuple[list[str], dict]:
    queues: dict[str, list[str]] = {f: [] for f in FAMILY_WEIGHTS}
    per_group: Counter = Counter()
    capped = Counter()
    ordered_fps = sorted(eligible, key=lambda fp: canonical_hash({"seed": seed, "select": fp}))
    for fp in ordered_fps:
        agg = states[fp]
        family = SOURCES[agg.primary].family
        group = agg.group or f"state:{fp}"
        if per_group[group] >= MAX_POSITIONS_PER_GAME:
            capped[family] += 1
            continue
        per_group[group] += 1
        queues[family].append(fp)
    available = {f: len(q) for f, q in queues.items()}
    alloc = allocate(available, total)
    target = sum(alloc.values())
    shares = {f: alloc[f] / target for f in alloc if alloc[f]}
    taken = Counter()
    order: list[str] = []
    for i in range(target):
        family = max(shares, key=lambda f: (shares[f] * (i + 1) - taken[f], -list(FAMILY_WEIGHTS).index(f)) if taken[f] < alloc[f] else (-math.inf, 0))
        order.append(queues[family][taken[family]])
        taken[family] += 1
    return order, {"available_after_game_cap": available, "allocation": alloc, "dropped_by_game_cap": dict(capped), "max_positions_per_game": MAX_POSITIONS_PER_GAME}


def holdout(group: str, seed: int) -> bool:
    return int(canonical_hash({"seed": seed, "holdout": group})[:8], 16) / 0xFFFFFFFF < HOLDOUT_FRACTION


def detail_selected(repo: Path, selected: set[str], *, on_error, sources=SOURCES) -> dict[str, dict]:
    """Passe 2 : sources, parties et anciennes cibles des seuls etats selectionnes."""

    details: dict[str, dict] = {fp: {"sources": Counter(), "games": [], "old_targets": {}} for fp in selected}
    for source in sources:
        for occ in iter_source(repo, source, on_error):
            item = details.get(occ["fingerprint"])
            if item is None:
                continue
            item["sources"][source.name] += 1
            if occ["game_id"] and len(item["games"]) < 8 and occ["game_id"] not in item["games"]:
                item["games"].append(occ["game_id"])
            old = occ["old_target"]
            if old and old.get("visit_counts") and old["kind"] not in item["old_targets"]:
                item["old_targets"][old["kind"]] = {**old, "source": source.name}
    return details


def selection_rows(order: list[str], states: dict[str, Aggregate], eligible: dict[str, dict], details: dict[str, dict], *, seed: int) -> list[dict]:
    rows = []
    for rank, fp in enumerate(order):
        agg, info, detail = states[fp], eligible[fp], details[fp]
        group = agg.group or f"state:{fp}"
        z_known = agg.z_win + agg.z_draw + agg.z_loss
        rows.append({
            "order": rank, "fingerprint": fp, "state": {"board": list(agg.board), "player_to_move": agg.player},
            "family": SOURCES[agg.primary].family, "primary_source": SOURCES[agg.primary].name,
            "source_occurrence_count": agg.occurrences, "sources": dict(detail["sources"]),
            "z_counts": {"win": agg.z_win, "draw": agg.z_draw, "loss": agg.z_loss, "unknown": agg.z_unknown},
            "z_perspective": "player_to_move", "value_target_available": z_known > 0,
            "z_mean": (agg.z_win - agg.z_loss) / z_known if z_known else None,
            "split_group": group, "games_sample": detail["games"], "min_ply": agg.min_ply,
            "legal_count": info["legal_count"], "seeds_in_play": info["seeds_in_play"], "phase": info["phase"],
            "stores": [agg.board[14], agg.board[15]], "holdout": holdout(group, seed),
            "old_targets": list(detail["old_targets"].values()),
        })
    return rows


def write_selection(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
    return canonical_hash([r["fingerprint"] for r in rows])


def read_selection(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if [r["order"] for r in rows] != list(range(len(rows))):
        raise Lot44FatalError("CORPUS_CORRUPT", f"selection order is not contiguous: {path}")
    if len({r["fingerprint"] for r in rows}) != len(rows):
        raise Lot44FatalError("CORPUS_CORRUPT", "duplicate physical state in selection")
    return rows


def composition(rows: list[dict]) -> dict:
    def dist(key) -> dict:
        return dict(sorted(Counter(key(r) for r in rows).items(), key=lambda kv: str(kv[0])))

    return {
        "positions": len(rows),
        "family": dist(lambda r: r["family"]), "player_to_move": dist(lambda r: r["state"]["player_to_move"]),
        "legal_count": dist(lambda r: r["legal_count"]), "phase": dist(lambda r: r["phase"]),
        "ply_bin": dist(lambda r: "unknown" if r["min_ply"] is None else f"{(r['min_ply'] // 20) * 20}-{(r['min_ply'] // 20) * 20 + 19}"),
        "store_leader": dist(lambda r: "p1" if r["stores"][0] > r["stores"][1] else "p2" if r["stores"][1] > r["stores"][0] else "tie"),
        "value_target_available": dist(lambda r: r["value_target_available"]), "holdout": dist(lambda r: r["holdout"]),
        "multi_occurrence_states": sum(r["source_occurrence_count"] > 1 for r in rows),
        "logical_occurrences": sum(r["source_occurrence_count"] for r in rows),
    }


def deduplication_report(states: dict[str, Aggregate]) -> dict:
    multiplicity = Counter(min(a.occurrences, 10) for a in states.values())
    overlaps = Counter()
    for a in states.values():
        names = tuple(s.name for i, s in enumerate(SOURCES) if a.source_mask >> i & 1)
        if len(names) > 1:
            overlaps[" + ".join(names)] += 1
    raw = sum(a.occurrences for a in states.values())
    return {
        "identity_key": "run_srn_lot39.fingerprint_state(board[16], player_to_move); no canonicalization",
        "raw_occurrences": raw, "unique_physical_states": len(states), "duplicate_occurrences": raw - len(states),
        "duplication_rate": (raw - len(states)) / raw if raw else None,
        "multiplicity_histogram_capped_10": {str(k): v for k, v in sorted(multiplicity.items())},
        "cross_source_overlap_top": dict(overlaps.most_common(25)),
        "frequency_preserved_as": "source_occurrence_count + sources{} + z_counts{} per unique state",
    }


def coverage_report(rows: list[dict]) -> dict:
    out = {"selected": len(rows)}
    for corpus, names in TRAINING_CORPORA.items():
        hits = sum(any(n in r["sources"] for n in names) for r in rows)
        out[corpus] = {"sources": list(names), "present": hits, "new": len(rows) - hits, "new_rate": (len(rows) - hits) / len(rows) if rows else None}
    trained = set(TRAINING_CORPORA["G4_TRAINING"]) | set(TRAINING_CORPORA["G3_TRAINING"]) | set(TRAINING_CORPORA["G2_TRAINING_PROXY"])
    new = sum(not any(n in trained for n in r["sources"]) for r in rows)
    out["NEW_STATE_RATE_VS_G2_G3_G4_TRAINING"] = new / len(rows) if rows else None
    out["lot41_44_diagnostic_states"] = 0
    out["note"] = "states are reused from autonomous corpora; the novelty of Lot45 is the MCTS32768 target. G2 training data are only partially available locally (D_RL proxy)."
    return out


def source_by_name(name: str) -> Source:
    return next(s for s in SOURCES if s.name == name)
