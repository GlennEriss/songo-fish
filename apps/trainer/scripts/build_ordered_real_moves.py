#!/usr/bin/env python3
"""Build a privacy-safe dataset of recoverable human Songo moves.

The Firebase exports contain the current/final board, match events and, for
newer matches, a server replay seed for the latest move.  They do *not*
contain every historical board.  This script therefore emits only moves that
can be proved by the legacy rules engine:

* the exact latest transition saved by ``replaySeedBoardCounts``;
* complete short trajectories that have exactly one legal path from the
  standard initial board to the exported board at the known/inferred depth.

Moves made by fake players or bots are excluded, while verified moves made by
the real player in a mixed match are preserved. Player identifiers and names
are never written to the output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "songo-real-move-v1"
RULES_VERSION = "songo_legacy_single.py"
INITIAL_BOARD = tuple([5] * 14 + [0, 0])
FAKE_ID_PREFIXES = ("fake_", "bot_")

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = REPOSITORY_ROOT / "apps" / "data"
DEFAULT_MATCHES = DATA_DIR / "brutes" / "gestion-songo-prod-default-rtdb-matches_live-export.json"
DEFAULT_EVENTS = DATA_DIR / "brutes" / "gestion-songo-prod-default-rtdb-matches_live_events-export.json"
DEFAULT_RULES = REPOSITORY_ROOT / "docs" / "songo_legacy_single.py"
DEFAULT_OUTPUT = DATA_DIR / "ordonnées" / "match_moves_v1.jsonl"
DEFAULT_JSON_OUTPUT = DATA_DIR / "ordonnées" / "match_moves_v1.json"


Board = Tuple[int, ...]
PathMoves = Tuple[int, ...]
StateKey = Tuple[Board, int, bool, Optional[int]]


@dataclass
class MatchSpec:
    match_id: str
    match: Mapping[str, Any]
    events: List[Tuple[str, Mapping[str, Any]]]
    final_board: Board
    candidate_depths: Tuple[int, ...]
    depth_source: str
    expected_last_case_id: Optional[int]
    event_sequence_valid: bool
    fake_player_positions: Tuple[int, ...]


@dataclass
class PathSolution:
    # Counts are deliberately capped at two: two means "ambiguous".
    count: int = 0
    path: Optional[PathMoves] = None
    depth: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover only verifiable human moves and their board-before state."
    )
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help="Also write the same records as a standard JSON array.",
    )
    parser.add_argument(
        "--max-reconstruction-depth",
        type=int,
        default=7,
        help="Maximum full-history depth. State-space growth is exponential (default: 7).",
    )
    return parser.parse_args()


def load_json_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object at its root".format(path))
    return value


def load_rules_module(path: Path) -> Any:
    module_name = "songo_legacy_rules_for_dataset"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import rules from {}".format(path))
    module = importlib.util.module_from_spec(spec)
    # dataclasses on Python 3.9 resolves the module through sys.modules.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def integer(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def position(value: Any) -> Optional[int]:
    parsed = integer(value)
    return parsed if parsed in (1, 2) else None


def case_id(value: Any) -> Optional[int]:
    parsed = integer(value)
    return parsed if parsed is not None and 0 <= parsed <= 13 else None


def normalize_board(value: Any) -> Optional[Board]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, dict):
        try:
            raw = [value[str(index)] if str(index) in value else value[index] for index in range(16)]
        except (KeyError, TypeError):
            return None
    else:
        return None

    if len(raw) != 16:
        return None
    parsed: List[int] = []
    for item in raw:
        number = integer(item)
        if number is None or number < 0:
            return None
        parsed.append(number)
    if sum(parsed) != 70:
        return None
    return tuple(parsed)


def event_timestamp(event: Mapping[str, Any]) -> Optional[int]:
    return unix_milliseconds(event.get("createdAtUnix"))


def unix_milliseconds(value: Any) -> Optional[int]:
    timestamp = integer(value)
    if timestamp is None:
        return None
    # Accept plausible Unix seconds or milliseconds, reject zero/sentinels.
    if 946_684_800 <= timestamp <= 7_258_118_400:
        return timestamp * 1000
    if 946_684_800_000 <= timestamp <= 7_258_118_400_000:
        return timestamp
    return None


def ordered_events(value: Any) -> List[Tuple[str, Mapping[str, Any]]]:
    if not isinstance(value, dict):
        return []
    events = [(str(event_id), event) for event_id, event in value.items() if isinstance(event, dict)]
    events.sort(
        key=lambda item: (
            event_timestamp(item[1]) if event_timestamp(item[1]) is not None else sys.maxsize,
            item[0],
        )
    )
    return events


def looks_like_fake_identifier(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower().startswith(FAKE_ID_PREFIXES)


def contains_fake_identifier(value: Any, parent_key: str = "") -> bool:
    """Inspect identifier-shaped fields without treating ordinary text as IDs."""
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if contains_fake_identifier(child, lowered):
                return True
        return False
    if isinstance(value, list):
        return any(contains_fake_identifier(child, parent_key) for child in value)
    identifier_field = (
        parent_key.endswith("id")
        or parent_key.startswith("id")
        or "playerid" in parent_key
        or parent_key in {"playeridscsv", "actorid", "winnerid", "abandonedplayerid"}
    )
    if identifier_field and isinstance(value, str):
        return any(looks_like_fake_identifier(part) for part in value.split(","))
    return False


def fake_player_positions(
    match: Mapping[str, Any], events: Sequence[Tuple[str, Mapping[str, Any]]]
) -> Tuple[int, ...]:
    positions = set()
    bot_position = position(match.get("botPlayerPosition"))
    if bot_position is not None:
        positions.add(bot_position)
    for player in (1, 2):
        if looks_like_fake_identifier(match.get("player{}Id".format(player))):
            positions.add(player)
    for _, event in events:
        for player in (1, 2):
            if looks_like_fake_identifier(event.get("player{}Id".format(player))):
                positions.add(player)
    return tuple(sorted(positions))


def is_marked_fake_or_bot(
    match: Mapping[str, Any], events: Sequence[Tuple[str, Mapping[str, Any]]]
) -> bool:
    if match.get("isBotMatch") is True or position(match.get("botPlayerPosition")) is not None:
        return True
    mmr = match.get("mmr")
    if isinstance(mmr, dict) and mmr.get("isFakeMatch") is True:
        return True
    if contains_fake_identifier(match):
        return True
    return any(contains_fake_identifier(event) for _, event in events)


def infer_candidate_depths(
    match: Mapping[str, Any],
    events: Sequence[Tuple[str, Mapping[str, Any]]],
    fake_positions: Sequence[int],
) -> Tuple[Tuple[int, ...], str]:
    if fake_positions:
        return infer_mixed_match_depths(match, events, fake_positions)

    revision = integer(match.get("revision"))
    if revision is not None and revision >= 0:
        return (revision,), "revision"

    your_turn_count = sum(1 for _, event in events if event.get("event") == "your_turn")
    has_finish_event = any(event.get("event") == "match_finished" for _, event in events)
    depths = {your_turn_count}
    # A normal terminal move may emit match_finished instead of your_turn.
    # An abandonment also emits match_finished but does not add a move. Both
    # hypotheses are tested against the final board and the rules engine.
    if has_finish_event:
        depths.add(your_turn_count + 1)
    return tuple(sorted(depths)), "events"


def has_abandonment_marker(match: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
    if position(event.get("abandonedPlayerPosition")) is not None:
        return True
    if event.get("idAbandonneur") or event.get("abandonedPlayerId"):
        return True
    text = " ".join(
        str(value).lower()
        for value in (
            event.get("abandonReason"),
            event.get("reason"),
            match.get("reason"),
        )
        if value
    )
    return any(marker in text for marker in ("abandon", "inactive", "timeout", "disconnect"))


def infer_mixed_match_depths(
    match: Mapping[str, Any],
    events: Sequence[Tuple[str, Mapping[str, Any]]],
    fake_positions: Sequence[int],
) -> Tuple[Tuple[int, ...], str]:
    # The historical bot stream in this export always uses a real P1 and a
    # fake P2. Refuse to generalize the omission formula to another layout.
    if tuple(fake_positions) != (2,):
        return tuple(), "mixed_actor_events_invalid"

    move_events = [event for _, event in events if event.get("event") == "your_turn"]
    actors = [actor_position(event, match) for event in move_events]
    if not actors or any(actor not in (1, 2) for actor in actors):
        return tuple(), "mixed_actor_events_invalid"

    complete_pattern = [1 if index % 2 == 0 else 2 for index in range(len(actors))]
    if actors == complete_pattern:
        stream_kind = "complete"
        omissions = 0
    elif all(actor == 2 for actor in actors):
        stream_kind = "bot_only"
        omissions = len(actors)
    else:
        prefix_length = 0
        while prefix_length < len(actors) and actors[prefix_length] == 2:
            prefix_length += 1
        suffix = actors[prefix_length:]
        suffix_pattern = [1 if index % 2 == 0 else 2 for index in range(len(suffix))]
        if prefix_length == 0 or not suffix or suffix != suffix_pattern:
            return tuple(), "mixed_actor_events_invalid"
        stream_kind = "migrated_prefix"
        omissions = prefix_length

    base_depth = len(actors) + omissions
    finish_events = [event for _, event in events if event.get("event") == "match_finished"]
    final_event = finish_events[-1] if finish_events else None
    depths = set()

    if final_event is None:
        if stream_kind != "bot_only":
            depths.add(base_depth)
        else:
            current_turn = position(match.get("currentTurnPlayerPosition"))
            if current_turn == 1:
                depths.add(base_depth)
            elif current_turn == 2:
                depths.add(base_depth + 1)
            else:
                depths.update((base_depth, base_depth + 1))
    elif has_abandonment_marker(match, final_event):
        # Abandonment closes the match but does not itself play a move.
        if stream_kind != "bot_only":
            depths.add(base_depth)
        else:
            abandoned = position(final_event.get("abandonedPlayerPosition"))
            if abandoned == 1:
                depths.add(base_depth)
            elif abandoned == 2:
                depths.add(base_depth + 1)
            else:
                depths.update((base_depth, base_depth + 1))
    elif stream_kind != "bot_only":
        # A normal match_finished replaces the final your_turn notification.
        depths.add(base_depth + 1)
    else:
        final_actor = actor_position(final_event, match)
        if final_actor == 1:
            depths.add(base_depth + 1)
        elif final_actor == 2:
            depths.add(base_depth + 2)
        else:
            depths.update((base_depth + 1, base_depth + 2))

    return tuple(sorted(depth for depth in depths if depth >= 0)), "mixed_actor_events"


def expected_last_case(match: Mapping[str, Any]) -> Optional[int]:
    # lastPlayedCaseId is a legacy/UI field: it can be invalid or stale after
    # migrations. lastMoveCaseId is the server's revision-bound field.
    return case_id(match.get("lastMoveCaseId"))


def has_sequential_move_events(
    match: Mapping[str, Any], events: Sequence[Tuple[str, Mapping[str, Any]]]
) -> bool:
    move_index = 0
    for _, event in events:
        if event.get("event") != "your_turn":
            continue
        actor = actor_position(event, match)
        next_player = position(event.get("currentTurnPlayerPosition"))
        expected_actor = 1 if move_index % 2 == 0 else 2
        expected_next = 2 if expected_actor == 1 else 1
        if actor != expected_actor:
            return False
        if next_player is not None and next_player != expected_next:
            return False
        move_index += 1
    return True


def has_consistent_depth_evidence(spec: MatchSpec) -> bool:
    if spec.depth_source != "revision":
        return True
    # Bot/fake event streams can omit moves made on the real player's client.
    # The revision is still usable because every candidate path is replayed
    # from the initial board and must reproduce the exported final board.
    if spec.fake_player_positions:
        return True
    revision = spec.candidate_depths[0]
    your_turn_count = sum(1 for _, event in spec.events if event.get("event") == "your_turn")
    # A terminal move can replace its your_turn notification with
    # match_finished, hence the one-event tolerance.
    return revision in (your_turn_count, your_turn_count + 1)


def build_match_specs(
    matches: Mapping[str, Any],
    event_groups: Mapping[str, Any],
    stats: Counter,
) -> List[MatchSpec]:
    specs: List[MatchSpec] = []
    stats["source_matches"] = len(matches)
    stats["source_event_groups"] = len(event_groups)

    for raw_match_id, raw_match in matches.items():
        match_id = str(raw_match_id)
        if not isinstance(raw_match, dict):
            stats["invalid_match_record"] += 1
            continue
        events = ordered_events(event_groups.get(raw_match_id, event_groups.get(match_id)))
        if not events:
            stats["matches_without_events_ignored"] += 1
            continue
        stats["matches_joined_with_events"] += 1
        fake_positions = fake_player_positions(raw_match, events)
        marked_fake = is_marked_fake_or_bot(raw_match, events)
        if marked_fake and not fake_positions:
            stats["fake_or_bot_matches_unknown_position_ignored"] += 1
            continue
        if len(fake_positions) == 2:
            stats["all_fake_matches_ignored"] += 1
            continue
        if fake_positions:
            stats["mixed_real_fake_matches_considered"] += 1
        else:
            stats["human_only_matches_considered"] += 1
        stats["matches_with_real_player_considered"] += 1

        final_board = normalize_board(raw_match.get("boardCounts"))
        if final_board is None:
            stats["matches_with_real_player_invalid_final_board"] += 1
            continue
        depths, depth_source = infer_candidate_depths(raw_match, events, fake_positions)
        specs.append(
            MatchSpec(
                match_id=match_id,
                match=raw_match,
                events=events,
                final_board=final_board,
                candidate_depths=depths,
                depth_source=depth_source,
                # In mixed matches lastMoveCaseId can remain on the previous
                # human move after the bot has played. The final board/path
                # validation is authoritative instead.
                expected_last_case_id=None if fake_positions else expected_last_case(raw_match),
                event_sequence_valid=has_sequential_move_events(raw_match, events),
                fake_player_positions=fake_positions,
            )
        )
    return specs


def add_path_solution(solution: PathSolution, count: int, path: Optional[PathMoves], depth: int) -> None:
    if count <= 0:
        return
    if solution.count == 0 and count == 1 and path is not None:
        solution.count = 1
        solution.path = path
        solution.depth = depth
        return
    solution.count = 2
    solution.path = None
    solution.depth = None


def merge_state(
    states: Dict[StateKey, Tuple[int, Optional[PathMoves]]],
    key: StateKey,
    count: int,
    path: Optional[PathMoves],
) -> None:
    existing = states.get(key)
    if existing is None:
        states[key] = (count, path) if count == 1 else (2, None)
        return
    # Distinct prefixes reached the same state, so every continuation from
    # this state is ambiguous too.
    states[key] = (2, None)


def reconstruct_unique_paths(
    specs: Sequence[MatchSpec], rules: Any, max_depth: int, stats: Counter
) -> Dict[str, Tuple[MatchSpec, PathMoves]]:
    targets: Dict[int, Dict[Board, List[MatchSpec]]] = defaultdict(lambda: defaultdict(list))
    solutions: Dict[str, PathSolution] = {spec.match_id: PathSolution() for spec in specs}

    for spec in specs:
        if spec.depth_source == "mixed_actor_events_invalid":
            stats["mixed_event_streams_not_reconstructable"] += 1
            continue
        if spec.depth_source == "events" and not spec.event_sequence_valid:
            stats["nonsequential_event_matches_not_reconstructed"] += 1
            continue
        if not has_consistent_depth_evidence(spec):
            stats["revision_event_depth_mismatch_not_reconstructed"] += 1
            continue
        positive_depths = [depth for depth in spec.candidate_depths if depth > 0]
        if not positive_depths:
            stats["zero_move_matches"] += 1
            continue
        usable_depths = [depth for depth in positive_depths if depth <= max_depth]
        if not usable_depths:
            stats["full_path_depth_above_limit"] += 1
            continue
        for depth in usable_depths:
            targets[depth][spec.final_board].append(spec)

    initial_key: StateKey = (INITIAL_BOARD, 1, False, None)
    states: Dict[StateKey, Tuple[int, Optional[PathMoves]]] = {initial_key: (1, tuple())}

    for depth in range(1, max_depth + 1):
        next_states: Dict[StateKey, Tuple[int, Optional[PathMoves]]] = {}
        depth_targets = targets.get(depth, {})

        for (board, turn, finished, winner), (path_count, path) in states.items():
            game = rules.SongoLegacyGame.from_board(board, turn)
            game.finished = finished
            game.winner = winner
            for move in game.legal_moves():
                child = game.clone()
                result = child.play(move)
                child_board = tuple(result.board)
                child_path = path + (move,) if path_count == 1 and path is not None else None

                for exported_board in possible_exported_boards(result):
                    for spec in depth_targets.get(exported_board, []):
                        if spec.expected_last_case_id is not None and move != spec.expected_last_case_id:
                            continue
                        add_path_solution(solutions[spec.match_id], path_count, child_path, depth)

                key: StateKey = (child_board, child.turn, child.finished, child.winner)
                merge_state(next_states, key, path_count, child_path)
        states = next_states
        stats["states_at_depth_{}".format(depth)] = len(states)

    recovered: Dict[str, Tuple[MatchSpec, PathMoves]] = {}
    for spec in specs:
        solution = solutions[spec.match_id]
        if solution.count == 1 and solution.path is not None:
            recovered[spec.match_id] = (spec, solution.path)
            stats["unique_full_path_matches"] += 1
            stats["unique_full_path_moves"] += len(solution.path)
        elif solution.count >= 2:
            stats["ambiguous_full_path_matches_ignored"] += 1
        elif (
            spec.depth_source != "mixed_actor_events_invalid"
            and
            (spec.depth_source != "events" or spec.event_sequence_valid)
            and has_consistent_depth_evidence(spec)
            and any(0 < depth <= max_depth for depth in spec.candidate_depths)
        ):
            if any(depth > max_depth for depth in spec.candidate_depths):
                stats["full_path_inconclusive_above_limit"] += 1
            else:
                stats["unreachable_full_path_matches_ignored"] += 1
    return recovered


def swept_famine_board(result: Any) -> Optional[Board]:
    if not result.finished or result.reason != "famine_after_move":
        return None
    board = tuple(result.board)
    score_1 = board[14] + sum(board[0:7])
    score_2 = board[15] + sum(board[7:14])
    return tuple([0] * 14 + [score_1, score_2])


def possible_exported_boards(result: Any) -> Tuple[Board, ...]:
    ordinary = tuple(result.board)
    swept = swept_famine_board(result)
    if swept is None or swept == ordinary:
        return (ordinary,)
    return ordinary, swept


def canonicalize_board(board: Sequence[int], player: int) -> List[int]:
    if player == 1:
        return list(board)
    return list(board[7:14]) + list(board[0:7]) + [board[15], board[14]]


def transition_record(
    match_id: str,
    ply: int,
    played_at_unix_ms: Optional[int],
    board_before: Sequence[int],
    legal_case_ids: Sequence[int],
    move: int,
    result: Any,
    method: str,
    board_after_override: Optional[Sequence[int]] = None,
    timestamp_source: Optional[str] = None,
) -> Dict[str, Any]:
    player = int(result.player)
    offset = 0 if player == 1 else 7
    legal_mask = [offset + local_action in legal_case_ids for local_action in range(7)]
    local_action = move - offset
    if not (0 <= local_action <= 6) or not legal_mask[local_action]:
        raise ValueError("Recovered move is not legal for its player")
    board_after = list(result.board) if board_after_override is None else list(board_after_override)
    return {
        "schema_version": SCHEMA_VERSION,
        "rules_version": RULES_VERSION,
        "match_id": match_id,
        "ply": ply,
        "played_at_unix_ms": played_at_unix_ms,
        "timestamp_source": timestamp_source,
        "player_position": player,
        "board_before": list(board_before),
        "state": canonicalize_board(board_before, player),
        "legal_case_ids": list(legal_case_ids),
        "legal_mask": legal_mask,
        "case_id": move,
        "action_local": local_action,
        "board_after": board_after,
        "captured": int(result.captured),
        "next_player_position": result.next_player,
        "finished_after": bool(result.finished),
        "winner_after": result.winner,
        "terminal_board_sweep_applied": board_after != list(result.board),
        "recovery_certainty": "exact",
        "recovery_methods": [method],
    }


def actor_position(event: Mapping[str, Any], match: Mapping[str, Any]) -> Optional[int]:
    actor_id = event.get("actorId")
    if not isinstance(actor_id, str) or not actor_id:
        return None
    if actor_id == match.get("player1Id"):
        return 1
    if actor_id == match.get("player2Id"):
        return 2
    return None


def align_event_timestamps(spec: MatchSpec, records: Sequence[Dict[str, Any]]) -> List[Optional[int]]:
    depth = len(records)
    your_turn_events = [event for _, event in spec.events if event.get("event") == "your_turn"]
    finish_events = [event for _, event in spec.events if event.get("event") == "match_finished"]
    if depth not in (len(your_turn_events), len(your_turn_events) + 1):
        return [None] * depth
    if depth == len(your_turn_events) + 1 and not finish_events:
        return [None] * depth

    timestamps: List[Optional[int]] = [None] * depth
    for index, event in enumerate(your_turn_events):
        record = records[index]
        event_actor = actor_position(event, spec.match)
        if event_actor is not None and event_actor != record["player_position"]:
            return [None] * depth
        event_next = position(event.get("currentTurnPlayerPosition"))
        if event_next is not None and event_next != record["next_player_position"]:
            return [None] * depth
        timestamps[index] = event_timestamp(event)

    if depth == len(your_turn_events) + 1:
        final_event = finish_events[-1]
        final_record = records[-1]
        event_actor = actor_position(final_event, spec.match)
        if event_actor is not None and event_actor != final_record["player_position"]:
            return [None] * depth
        timestamps[-1] = event_timestamp(final_event)
    return timestamps


def records_from_full_path(spec: MatchSpec, path: PathMoves, rules: Any) -> List[Dict[str, Any]]:
    game = rules.SongoLegacyGame()
    records: List[Dict[str, Any]] = []
    method = "unique_full_path_{}".format(spec.depth_source)
    for ply, move in enumerate(path, start=1):
        board_before = list(game.board)
        legal_case_ids = game.legal_moves()
        result = game.play(move)
        is_last = ply == len(path)
        exported_after = spec.final_board if is_last and spec.final_board in possible_exported_boards(result) else None
        records.append(
            transition_record(
                spec.match_id,
                ply,
                None,
                board_before,
                legal_case_ids,
                move,
                result,
                method,
                exported_after,
            )
        )
    if spec.final_board not in possible_exported_boards(result):
        raise ValueError("Full path does not reach exported board for {}".format(spec.match_id))
    for record, timestamp in zip(records, align_event_timestamps(spec, records)):
        record["played_at_unix_ms"] = timestamp
        record["timestamp_source"] = "event_created_at" if timestamp is not None else None
        record["excluded_fake_player_positions"] = list(spec.fake_player_positions)
    return records


def exact_replay_record(spec: MatchSpec, rules: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    match = spec.match
    before = normalize_board(match.get("replaySeedBoardCounts"))
    if before is None:
        return None, "missing_replay_seed"
    move = case_id(match.get("lastMoveCaseId"))
    if move is None:
        return None, "missing_replay_move"
    if match.get("replaySeedFinished") is True:
        return None, "replay_seed_already_finished"

    owner = 1 if move <= 6 else 2
    if owner in spec.fake_player_positions:
        return None, "fake_player_move_ignored"
    stated_positions = [
        candidate
        for candidate in (
            position(match.get("replaySeedCurrentTurnPlayerPosition")),
            position(match.get("lastMovePlayerPosition")),
        )
        if candidate is not None
    ]
    if any(candidate != owner for candidate in stated_positions):
        return None, "replay_player_mismatch"

    revision = integer(match.get("revision"))
    target_revision = integer(match.get("replayTargetRevision"))
    seed_revision = integer(match.get("replaySeedRevision"))
    known_targets = [value for value in (revision, target_revision) if value is not None and value >= 1]
    if len(set(known_targets)) > 1:
        return None, "replay_target_revision_mismatch"
    ply = known_targets[0] if known_targets else (seed_revision + 1 if seed_revision is not None else None)
    if ply is None or ply < 1:
        return None, "missing_replay_ply"
    if seed_revision is not None and seed_revision + 1 != ply:
        return None, "replay_seed_revision_mismatch"

    game = rules.SongoLegacyGame.from_board(before, owner)
    legal_case_ids = game.legal_moves()
    if move not in legal_case_ids:
        return None, "replay_move_illegal"
    result = game.play(move)
    if spec.final_board not in possible_exported_boards(result):
        return None, "replay_result_board_mismatch"

    timestamp = None
    timestamp_source = None
    # updatedAtUnix is deliberately excluded: later maintenance writes can
    # update the match days after the actual move.
    for field in ("lastMoveAtUnix", "replayTargetUpdatedAtUnix"):
        candidate = unix_milliseconds(match.get(field))
        if candidate is not None:
            timestamp = candidate
            timestamp_source = field
            break
    record = transition_record(
        spec.match_id,
        ply,
        timestamp,
        before,
        legal_case_ids,
        move,
        result,
        "exact_server_replay_seed",
        spec.final_board,
        timestamp_source,
    )
    record["excluded_fake_player_positions"] = list(spec.fake_player_positions)
    return record, "ok"


def record_signature(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        tuple(record["board_before"]),
        record["player_position"],
        record["case_id"],
        tuple(record["board_after"]),
    )


def merge_records(
    full_records: Sequence[Dict[str, Any]], replay_records: Sequence[Dict[str, Any]], stats: Counter
) -> List[Dict[str, Any]]:
    full_by_key = {(row["match_id"], row["ply"]): row for row in full_records}
    replay_by_key = {(row["match_id"], row["ply"]): row for row in replay_records}
    conflict_matches = {
        match_id
        for (match_id, ply), full in full_by_key.items()
        if (match_id, ply) in replay_by_key
        and record_signature(full) != record_signature(replay_by_key[(match_id, ply)])
    }
    if conflict_matches:
        stats["matches_with_reconstruction_conflict"] = len(conflict_matches)
        full_by_key = {
            key: value for key, value in full_by_key.items() if key[0] not in conflict_matches
        }

    merged: Dict[Tuple[str, int], Dict[str, Any]] = dict(full_by_key)
    for key, replay in replay_by_key.items():
        existing = merged.get(key)
        if existing is None:
            merged[key] = replay
            continue
        existing["recovery_methods"] = sorted(
            set(existing["recovery_methods"]) | set(replay["recovery_methods"])
        )
        if replay["played_at_unix_ms"] is not None:
            existing["played_at_unix_ms"] = replay["played_at_unix_ms"]
            existing["timestamp_source"] = replay["timestamp_source"]

    rows = list(merged.values())
    rows.sort(key=lambda row: (str(row["match_id"]), int(row["ply"])))
    return rows


def validate_output(
    rows: Sequence[Mapping[str, Any]], rules: Any, specs_by_id: Mapping[str, MatchSpec]
) -> None:
    seen = set()
    for row in rows:
        key = (row["match_id"], row["ply"])
        if key in seen:
            raise ValueError("Duplicate match/ply in output: {}".format(key))
        seen.add(key)
        before = normalize_board(row["board_before"])
        after = normalize_board(row["board_after"])
        if before is None or after is None:
            raise ValueError("Invalid board in output: {}".format(key))
        player = position(row["player_position"])
        move = case_id(row["case_id"])
        if player is None or move is None:
            raise ValueError("Invalid player/move in output: {}".format(key))
        spec = specs_by_id.get(str(row["match_id"]))
        if spec is None:
            raise ValueError("Output references an unknown match: {}".format(key))
        if player in spec.fake_player_positions:
            raise ValueError("Fake/bot player move leaked into output: {}".format(key))
        if list(row.get("excluded_fake_player_positions", [])) != list(spec.fake_player_positions):
            raise ValueError("Fake-player metadata mismatch: {}".format(key))
        game = rules.SongoLegacyGame.from_board(before, player)
        if move not in game.legal_moves():
            raise ValueError("Illegal move in output: {}".format(key))
        result = game.play(move)
        if after not in possible_exported_boards(result):
            raise ValueError("board_after validation failed: {}".format(key))
        expected_sweep = tuple(result.board) != after
        if bool(row.get("terminal_board_sweep_applied")) != expected_sweep:
            raise ValueError("Terminal sweep marker validation failed: {}".format(key))
        if row["action_local"] != (move if player == 1 else move - 7):
            raise ValueError("Local action validation failed: {}".format(key))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def write_json(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.max_reconstruction_depth < 0:
        raise ValueError("--max-reconstruction-depth cannot be negative")

    matches = load_json_object(args.matches)
    event_groups = load_json_object(args.events)
    rules = load_rules_module(args.rules)
    stats: Counter = Counter()
    specs = build_match_specs(matches, event_groups, stats)

    recovered_paths = reconstruct_unique_paths(specs, rules, args.max_reconstruction_depth, stats)
    full_records: List[Dict[str, Any]] = []
    for spec, path in recovered_paths.values():
        path_records = records_from_full_path(spec, path, rules)
        for record in path_records:
            if record["player_position"] in spec.fake_player_positions:
                stats["full_path_fake_player_moves_ignored"] += 1
                continue
            full_records.append(record)
            stats["full_path_real_player_moves"] += 1

    replay_records: List[Dict[str, Any]] = []
    for spec in specs:
        record, reason = exact_replay_record(spec, rules)
        stats["replay_{}".format(reason)] += 1
        if record is not None:
            replay_records.append(record)
    stats["exact_replay_seed_moves"] = len(replay_records)

    rows = merge_records(full_records, replay_records, stats)
    specs_by_id = {spec.match_id: spec for spec in specs}
    validate_output(rows, rules, specs_by_id)
    write_jsonl(args.output, rows)
    write_json(args.json_output, rows)

    stats["output_moves"] = len(rows)
    stats["output_matches"] = len({row["match_id"] for row in rows})
    stats["output_moves_from_mixed_matches"] = sum(
        1 for row in rows if row["excluded_fake_player_positions"]
    )
    stats["output_moves_from_human_only_matches"] = sum(
        1 for row in rows if not row["excluded_fake_player_positions"]
    )
    dated = sum(1 for row in rows if row["played_at_unix_ms"] is not None)
    stats["output_moves_with_timestamp"] = dated

    print(json.dumps(dict(sorted(stats.items())), indent=2, ensure_ascii=False))
    print("JSONL output: {}".format(args.output.resolve()))
    print("JSON output: {}".format(args.json_output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
