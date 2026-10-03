#!/usr/bin/env python3
"""Diagnostic reproductible du Lot 1A, sans modification des regles.

Le script caracterise le comportement courant de rules.py, fast_rules.py et
de la sandbox historique docs/songo_legacy_single.py. Il ne decide pas quelle
regle metier est correcte.

Usage:
    python apps/trainer/scripts/diagnose_lot1a_engine.py
    python apps/trainer/scripts/diagnose_lot1a_engine.py --json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from songo_ai.dataset.schema import canonicalize_board
from songo_ai.songo.fast_rules import FastSongoGame, capture as fast_capture, sow as fast_sow
from songo_ai.songo.rules import (
    DRAW,
    P1_STORE,
    P2_STORE,
    PLAYER_ONE,
    PLAYER_TWO,
    RepetitionTracker,
    SongoLegacyGame,
    local_action_to_pit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_PATH = PROJECT_ROOT / "docs" / "songo_legacy_single.py"

FULL_LAP_P1 = [14, 0, 0, 0, 0, 51, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0]
FULL_LAP_P2 = [0, 0, 0, 0, 0, 4, 1, 14, 0, 0, 0, 0, 51, 0, 0, 0]

TERMINAL_CASES = {
    "store_over_35_imported": ([14, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 36, 0], PLAYER_ONE),
    "stores_35_35": ([0] * 14 + [35, 35], PLAYER_ONE),
    "famine_no_transmit": ([6, 5, 4, 3, 2, 1, 1] + [0] * 7 + [24, 24], PLAYER_ONE),
    "current_camp_empty": ([0] * 7 + [20, 0, 0, 0, 0, 0, 0] + [25, 25], PLAYER_ONE),
}


def load_historical_legacy():
    spec = importlib.util.spec_from_file_location("songo_historical_legacy", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load historical legacy module: {LEGACY_PATH}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resout les annotations via sys.modules pendant l'import.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _jsonable_board(board) -> list[int]:
    return [int(v) for v in board]


def _changed_to_zero(before, after) -> list[int]:
    return [i for i in range(14) if int(before[i]) > 0 and int(after[i]) == 0]


def _reference_full_lap(board: list[int], player: int) -> dict[str, Any]:
    action = 0
    physical = local_action_to_pit(player, action)
    game = SongoLegacyGame.from_board(board, player)
    store = P1_STORE if player == PLAYER_ONE else P2_STORE
    store_before = int(game.board[store])
    initial = list(game.board)
    returned_arrival = int(game._sow(physical))
    trace = list(game.last_sow_trace)
    after_sow = list(game.board)
    capture_input = returned_arrival
    captured = int(game._capture(physical, capture_input))
    captured_pits = list(game.last_capture_trace)

    played = SongoLegacyGame.from_board(board, player)
    result = played.play_local(action)
    return {
        "engine": "rules.py",
        "initial_state": initial,
        "turn": player,
        "local_action": action,
        "physical_pit": physical,
        "initial_seed_count": initial[physical],
        "seed_trajectory": trace,
        "real_last_arrival": trace[-1],
        "sow_return_value": returned_arrival,
        "capture_input": capture_input,
        "captured_pits": captured_pits,
        "captured_seeds": captured,
        "store_before": store_before,
        "store_after": int(played.board[store]),
        "after_sow": after_sow,
        "final_state": list(played.board),
        "finished": played.finished,
        "winner": played.winner,
        "reason": result.reason,
    }


def _fast_full_lap(board: list[int], player: int, reference_trace: list[int]) -> dict[str, Any]:
    action = 0
    physical = local_action_to_pit(player, action)
    store = P1_STORE if player == PLAYER_ONE else P2_STORE
    working = np.array(board, dtype=np.int64)
    store_before = int(working[store])
    returned_arrival = int(fast_sow(working, physical))
    after_sow = working.tolist()
    before_capture = working.copy()
    captured = int(fast_capture(working, physical, returned_arrival))
    captured_pits = _changed_to_zero(before_capture, working)

    played = FastSongoGame.from_board(board, player)
    result = played.play_local(action)
    return {
        "engine": "fast_rules.py",
        "initial_state": list(board),
        "turn": player,
        "local_action": action,
        "physical_pit": physical,
        "initial_seed_count": board[physical],
        "seed_trajectory": reference_trace,
        "trajectory_note": "trace fournie par rules.py; etat post-semis fast_rules identique",
        "real_last_arrival": reference_trace[-1],
        "sow_return_value": returned_arrival,
        "capture_input": returned_arrival,
        "captured_pits": captured_pits,
        "captured_seeds": captured,
        "store_before": store_before,
        "store_after": int(played.board[store]),
        "after_sow": after_sow,
        "final_state": played.board.tolist(),
        "finished": played.finished,
        "winner": played.winner,
        "reason": result.reason,
    }


def _historical_full_lap(module, board: list[int], player: int, reference_trace: list[int]) -> dict[str, Any]:
    action = 0
    physical = local_action_to_pit(player, action)
    store = P1_STORE if player == PLAYER_ONE else P2_STORE
    game = module.SongoLegacyGame.from_board(board, player)
    store_before = int(game.board[store])
    returned_arrival = int(game._sow(physical))
    after_sow = list(game.board)
    before_capture = list(game.board)
    captured = int(game._capture(physical, returned_arrival))
    captured_pits = _changed_to_zero(before_capture, game.board)

    played = module.SongoLegacyGame.from_board(board, player)
    result = played.play(physical)
    return {
        "engine": "docs/songo_legacy_single.py",
        "initial_state": list(board),
        "turn": player,
        "local_action": action,
        "physical_pit": physical,
        "initial_seed_count": board[physical],
        "seed_trajectory": reference_trace,
        "trajectory_note": "trace reconstruite et confirmee par un etat post-semis identique",
        "real_last_arrival": reference_trace[-1],
        "sow_return_value": returned_arrival,
        "capture_input": returned_arrival,
        "captured_pits": captured_pits,
        "captured_seeds": captured,
        "store_before": store_before,
        "store_after": int(played.board[store]),
        "after_sow": after_sow,
        "final_state": list(played.board),
        "finished": played.finished,
        "winner": played.winner,
        "reason": result.reason,
    }


def full_lap_diagnostics() -> dict[str, Any]:
    legacy = load_historical_legacy()
    result: dict[str, Any] = {}
    for label, board, player in (
        ("J1", FULL_LAP_P1, PLAYER_ONE),
        ("J2", FULL_LAP_P2, PLAYER_TWO),
    ):
        reference = _reference_full_lap(board, player)
        fast = _fast_full_lap(board, player, reference["seed_trajectory"])
        historical = _historical_full_lap(legacy, board, player, reference["seed_trajectory"])
        result[label] = {
            "rules": reference,
            "fast_rules": fast,
            "historical_legacy": historical,
            "post_sow_equivalent": reference["after_sow"] == fast["after_sow"] == historical["after_sow"],
            "final_equivalent": reference["final_state"] == fast["final_state"] == historical["final_state"],
        }
    return result


def _swap_player(player: int) -> int:
    return PLAYER_TWO if player == PLAYER_ONE else PLAYER_ONE


def _relative_player(player: int, perspective: int) -> int:
    return player if perspective == PLAYER_ONE else _swap_player(player)


def _relative_winner(winner, perspective: int):
    if winner is None or winner == DRAW:
        return winner
    return _relative_player(winner, perspective)


def _normalize_reason(reason: str, perspective: int) -> str:
    if perspective == PLAYER_ONE:
        return reason
    return reason.replace("player_1", "PLAYER_TMP").replace("player_2", "player_1").replace("PLAYER_TMP", "player_2")


def compare_canonical_transition(game_cls, board: list[int], turn: int, action: int, tags: list[str]) -> dict[str, Any]:
    original = game_cls.from_board(board, turn)
    transformed_board = canonicalize_board(tuple(board), turn)
    canonical = game_cls.from_board(transformed_board, PLAYER_ONE)

    failures: list[str] = []
    original_pre_mask = tuple(original.legal_mask())
    canonical_pre_mask = tuple(canonical.legal_mask())
    if original_pre_mask != canonical_pre_mask:
        failures.append("pre_legal_mask")
    if action not in original.legal_local_actions():
        return {"skipped": True, "reason": "original action illegal", "tags": tags}
    if action not in canonical.legal_local_actions():
        failures.append("canonical_action_illegal")
        return {"skipped": False, "success": False, "failures": failures, "tags": tags}

    original_transmit_before = bool(original.can_transmit_for_player(turn))
    canonical_transmit_before = bool(canonical.can_transmit_for_player(PLAYER_ONE))
    if original_transmit_before != canonical_transmit_before:
        failures.append("transmission_before")

    original_result = original.play_local(action)
    canonical_result = canonical.play_local(action)
    expected_board = tuple(canonicalize_board(tuple(_jsonable_board(original.board)), turn))
    actual_board = tuple(_jsonable_board(canonical.board))
    if expected_board[:14] != actual_board[:14]:
        failures.append("pits")
    if expected_board[14:16] != actual_board[14:16]:
        failures.append("stores")

    expected_turn = _relative_player(original.turn, turn)
    if canonical.turn != expected_turn:
        failures.append("turn")
    if original_result.captured != canonical_result.captured:
        failures.append("captured")
    if original.finished != canonical.finished:
        failures.append("finished")
    if _relative_winner(original.winner, turn) != canonical.winner:
        failures.append("winner")
    if _normalize_reason(original_result.reason, turn) != canonical_result.reason:
        failures.append("reason")

    original_post_mask = tuple(original.legal_mask())
    canonical_post_mask = tuple(canonical.legal_mask())
    if original_post_mask != canonical_post_mask:
        failures.append("post_legal_mask")

    original_current_relative = _relative_player(original.turn, turn)
    original_transmit_after = bool(original.can_transmit_for_player(original.turn))
    canonical_transmit_after = bool(canonical.can_transmit_for_player(original_current_relative))
    if original_transmit_after != canonical_transmit_after:
        failures.append("transmission_after")

    return {
        "skipped": False,
        "success": not failures,
        "failures": failures,
        "engine": game_cls.__name__,
        "tags": tags,
        "initial_board": list(board),
        "turn": turn,
        "action": action,
        "physical_action": local_action_to_pit(turn, action),
        "seed_count": board[local_action_to_pit(turn, action)],
        "original_result": {
            "board": _jsonable_board(original.board),
            "turn": original.turn,
            "captured": original_result.captured,
            "finished": original.finished,
            "winner": original.winner,
            "reason": original_result.reason,
        },
        "canonical_result": {
            "board": _jsonable_board(canonical.board),
            "turn": canonical.turn,
            "captured": canonical_result.captured,
            "finished": canonical.finished,
            "winner": canonical.winner,
            "reason": canonical_result.reason,
        },
        "expected_canonical_board": list(expected_board),
    }


def _manual_canonical_cases() -> list[tuple[list[int], int, int, list[str]]]:
    initial = [5] * 14 + [0, 0]
    cases = [(initial, PLAYER_TWO, a, ["manual", "simple"]) for a in range(7)]
    cases.extend(
        [
            (FULL_LAP_P2, PLAYER_TWO, 0, ["manual", "limit", "full_lap", "store", "capture"]),
            (
                [1] * 7 + [0, 0, 0, 0, 0, 8, 0] + [27, 28],
                PLAYER_TWO,
                5,
                ["manual", "capture", "terminal"],
            ),
            (
                [0] * 7 + [0, 0, 0, 0, 0, 0, 2] + [34, 34],
                PLAYER_TWO,
                6,
                ["manual", "transmission", "near_famine"],
            ),
            (
                [0] * 7 + [6, 5, 4, 3, 2, 1, 2] + [23, 24],
                PLAYER_TWO,
                6,
                ["manual", "transmission", "near_famine", "limit"],
            ),
        ]
    )
    return cases


def _generated_canonical_cases(seed: int = 20260924, games: int = 40, max_moves: int = 250):
    rng = random.Random(seed)
    cases = []
    for _ in range(games):
        game = SongoLegacyGame()
        for _ply in range(max_moves):
            if game.finished:
                break
            legal = game.legal_local_actions()
            if not legal:
                game.normalize_terminal()
                break
            if game.turn == PLAYER_TWO:
                action = rng.choice(legal)
                probe = game.clone_for_search()
                result = probe.play_local(action)
                physical = local_action_to_pit(game.turn, action)
                tags = ["generated"]
                if game.board[physical] >= 14:
                    tags.extend(["full_lap", "store"])
                elif physical + game.board[physical] >= 14:
                    tags.append("store_passage")
                if result.captured:
                    tags.append("capture")
                if "famine" in result.reason or sum(game.board[0:7]) == 0:
                    tags.append("near_famine")
                if result.finished:
                    tags.append("terminal")
                cases.append((_jsonable_board(game.board), game.turn, action, tags))
            game.play_local(rng.choice(legal))
    return cases


def canonicalization_diagnostics() -> dict[str, Any]:
    cases = _manual_canonical_cases() + _generated_canonical_cases()
    report: dict[str, Any] = {}
    for game_cls in (SongoLegacyGame, FastSongoGame):
        records = [compare_canonical_transition(game_cls, *case) for case in cases]
        tested = [r for r in records if not r.get("skipped")]
        failures = [r for r in tested if not r["success"]]
        categories: dict[str, dict[str, int]] = {}
        for tag in sorted({tag for r in tested for tag in r["tags"]}):
            tagged = [r for r in tested if tag in r["tags"]]
            categories[tag] = {
                "tested": len(tagged),
                "successes": sum(1 for r in tagged if r["success"]),
                "failures": sum(1 for r in tagged if not r["success"]),
            }
        failure_kinds = Counter(kind for r in failures for kind in r["failures"])
        report[game_cls.__name__] = {
            "tested": len(tested),
            "successes": len(tested) - len(failures),
            "failures": len(failures),
            "failure_kinds": dict(sorted(failure_kinds.items())),
            "categories": categories,
            "minimal_failure_examples": failures[:5],
        }
    return report


def legal_moves_diagnostics() -> list[dict[str, Any]]:
    states = [
        ("initial_J1", [5] * 14 + [0, 0], PLAYER_ONE),
        ("initial_J2", [5] * 14 + [0, 0], PLAYER_TWO),
        ("full_lap_J1", FULL_LAP_P1, PLAYER_ONE),
        ("full_lap_J2", FULL_LAP_P2, PLAYER_TWO),
    ]
    rows = []
    for game_cls in (SongoLegacyGame, FastSongoGame):
        for name, board, turn in states:
            game = game_cls.from_board(board, turn)
            rows.append(
                {
                    "engine": game_cls.__name__,
                    "state": name,
                    "turn": turn,
                    "legal_moves_default": game.legal_moves(),
                    "legal_moves_player_one": game.legal_moves(PLAYER_ONE),
                    "legal_moves_player_two": game.legal_moves(PLAYER_TWO),
                    "legal_mask": list(game.legal_mask()),
                }
            )
    return rows


def terminal_diagnostics() -> list[dict[str, Any]]:
    rows = []
    for game_cls in (SongoLegacyGame, FastSongoGame):
        for name, (board, turn) in TERMINAL_CASES.items():
            game = game_cls.from_board(board, turn)

            def snapshot():
                return {
                    "finished": bool(game.finished),
                    "winner": game.winner,
                    "legal_mask": list(game.legal_mask()),
                    "legal_moves": game.legal_moves(),
                    "final_score_with_territory": list(game.final_score_with_territory()),
                }

            before = snapshot()
            game.normalize_terminal()
            after = snapshot()
            game.normalize_terminal()
            after_second_call = snapshot()
            rows.append(
                {
                    "engine": game_cls.__name__,
                    "case": name,
                    "board": list(board),
                    "turn": turn,
                    "before": before,
                    "after_normalize": after,
                    "after_second_normalize": after_second_call,
                    "idempotent": after == after_second_call,
                }
            )
    return rows


def cycle_and_truncation_diagnostics() -> dict[str, Any]:
    tracker = RepetitionTracker()
    state = SongoLegacyGame().to_state()
    counts = [tracker.record(state), tracker.record(state), tracker.record(state)]
    return {
        "repetition_tracker_counts": counts,
        "tracker_changes_game_state": False,
        "rules_integrate_repetition_as_terminal": False,
        "generation_max_moves_behavior": "stops trajectory without outcome classification",
        "tournament_max_moves_behavior": "counts unfinished game as draw",
        "rl_contract_recommendation": ["WIN", "LOSS", "DRAW", "TRUNCATED"],
        "truncated_to_z_zero_allowed": False,
    }


def build_report() -> dict[str, Any]:
    return {
        "lot": "1A",
        "rules_modified": False,
        "full_lap": full_lap_diagnostics(),
        "canonicalization": canonicalization_diagnostics(),
        "legal_moves": legal_moves_diagnostics(),
        "terminality": terminal_diagnostics(),
        "cycles_and_truncation": cycle_and_truncation_diagnostics(),
    }


def print_human(report: dict[str, Any]) -> None:
    print("LOT 1A - DIAGNOSTIC DU MOTEUR (AUCUNE CORRECTION DE REGLE)")
    print("=" * 72)
    for player in ("J1", "J2"):
        print(f"\nTOUR COMPLET {player}")
        for key in ("rules", "fast_rules", "historical_legacy"):
            row = report["full_lap"][player][key]
            print(f"\n[{row['engine']}]")
            for field in (
                "initial_state",
                "turn",
                "local_action",
                "physical_pit",
                "initial_seed_count",
                "seed_trajectory",
                "real_last_arrival",
                "sow_return_value",
                "capture_input",
                "captured_pits",
                "captured_seeds",
                "store_before",
                "store_after",
                "final_state",
            ):
                print(f"  {field}: {row[field]}")
        print(f"  post_sow_equivalent: {report['full_lap'][player]['post_sow_equivalent']}")
        print(f"  final_equivalent: {report['full_lap'][player]['final_equivalent']}")

    print("\nCANONICALISATION C(T(S,a)) == T(C(S),a')")
    for engine, row in report["canonicalization"].items():
        print(
            f"  {engine}: tested={row['tested']} successes={row['successes']} "
            f"failures={row['failures']} kinds={row['failure_kinds']}"
        )
        print(f"    categories={row['categories']}")
        for example in row["minimal_failure_examples"]:
            print(
                f"    failure turn={example['turn']} action={example['action']} "
                f"seed_count={example['seed_count']} tags={example['tags']} "
                f"fields={example['failures']}"
            )

    print("\nLEGAL_MOVES(PLAYER)")
    for row in report["legal_moves"]:
        print(
            f"  {row['engine']} {row['state']} turn={row['turn']} "
            f"default={row['legal_moves_default']} P1={row['legal_moves_player_one']} "
            f"P2={row['legal_moves_player_two']} mask={row['legal_mask']}"
        )

    print("\nTERMINALITE")
    for row in report["terminality"]:
        print(
            f"  {row['engine']} {row['case']}: before={row['before']} "
            f"after={row['after_normalize']} idempotent={row['idempotent']}"
        )

    print("\nCYCLES ET TRONCATURE")
    for key, value in report["cycles_and_truncation"].items():
        print(f"  {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="sortie JSON stable pour archivage")
    args = parser.parse_args()
    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
