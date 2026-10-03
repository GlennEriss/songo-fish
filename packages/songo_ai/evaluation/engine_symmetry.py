"""Transformations P1/P2 et audit d'equivariance du moteur Songo courant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from songo_ai.dataset.selfplay_schema import RawSongoState
from songo_ai.search.mcts import terminal_value
from songo_ai.songo.rules import (
    DRAW,
    NUM_ACTIONS,
    P1_STORE,
    P2_STORE,
    PLAYER_ONE,
    PLAYER_TWO,
    SongoLegacyGame,
    local_action_to_pit,
)


def swap_player(player: int) -> int:
    if player == PLAYER_ONE:
        return PLAYER_TWO
    if player == PLAYER_TWO:
        return PLAYER_ONE
    raise ValueError("player must be PLAYER_ONE or PLAYER_TWO")


@dataclass(frozen=True)
class PlayerSwapTransform:
    """Echange P1/P2 avec ordre local conserve ou inverse."""

    name: str
    reverse_local_order: bool = False

    def transform_state(self, state: RawSongoState) -> RawSongoState:
        first = state.board[7:14]
        second = state.board[0:7]
        if self.reverse_local_order:
            first = tuple(reversed(first))
            second = tuple(reversed(second))
        return RawSongoState(
            tuple(first) + tuple(second) + (state.board[P2_STORE], state.board[P1_STORE]),
            swap_player(state.player_to_move),
        )

    def transform_action(self, action: int) -> int:
        if not 0 <= action < NUM_ACTIONS:
            raise ValueError("action must be in 0..6")
        return NUM_ACTIONS - 1 - action if self.reverse_local_order else action

    def transform_vector(self, values: Sequence) -> tuple:
        if len(values) != NUM_ACTIONS:
            raise ValueError("action vector must contain seven entries")
        transformed = [None] * NUM_ACTIONS
        for action, value in enumerate(values):
            transformed[self.transform_action(action)] = value
        return tuple(transformed)

    def transform_mask(self, mask: Sequence[bool]) -> tuple[bool, ...]:
        return tuple(bool(value) for value in self.transform_vector(mask))

    def transform_policy(self, policy: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(value) for value in self.transform_vector(policy))

    @staticmethod
    def transform_winner(winner: int | None) -> int | None:
        if winner in (None, DRAW):
            return winner
        return swap_player(winner)


SWAP_KEEP_LOCAL = PlayerSwapTransform("swap_keep_local", False)
SWAP_REVERSE_LOCAL = PlayerSwapTransform("swap_reverse_local", True)
PLAYER_SWAP_CANDIDATES = (SWAP_KEEP_LOCAL, SWAP_REVERSE_LOCAL)


def _normalized_game(state: RawSongoState) -> SongoLegacyGame:
    game = SongoLegacyGame.from_state(state.to_engine_state())
    game.normalize_terminal()
    return game


def audit_legal_equivariance(
    state: RawSongoState, transform: PlayerSwapTransform
) -> dict:
    original = _normalized_game(state)
    transformed = _normalized_game(transform.transform_state(state))
    original_mask = (False,) * NUM_ACTIONS if original.finished else original.legal_mask()
    transformed_mask = (
        (False,) * NUM_ACTIONS if transformed.finished else transformed.legal_mask()
    )
    expected_mask = transform.transform_mask(original_mask)
    expected_winner = transform.transform_winner(original.winner)
    failures = []
    if transformed.finished != original.finished:
        failures.append("terminal")
    if transformed.winner != expected_winner:
        failures.append("winner")
    if transformed_mask != expected_mask:
        failures.append("legal_mask")
    return {
        "success": not failures,
        "failures": failures,
        "original_mask": original_mask,
        "expected_mask": expected_mask,
        "transformed_mask": transformed_mask,
        "original_terminal": original.finished,
        "transformed_terminal": transformed.finished,
        "original_winner": original.winner,
        "transformed_winner": transformed.winner,
    }


def audit_transition_equivariance(
    state: RawSongoState,
    action: int,
    transform: PlayerSwapTransform,
) -> dict:
    """Compare ``T(F(S,a))`` a ``F(T(S),T_action(a))``."""

    original = _normalized_game(state)
    if original.finished or not original.legal_mask()[action]:
        raise ValueError("transition audit requires a legal non-terminal action")
    transformed_before = transform.transform_state(state)
    transformed = _normalized_game(transformed_before)
    transformed_action = transform.transform_action(action)
    failures = []
    if transformed.finished or not transformed.legal_mask()[transformed_action]:
        failures.append("transformed_action_illegal")
        return {
            "success": False,
            "failures": failures,
            "action": action,
            "transformed_action": transformed_action,
            "state": state,
            "transformed_state": transformed_before,
        }

    physical_action = local_action_to_pit(state.player_to_move, action)
    transformed_physical_action = local_action_to_pit(
        transformed_before.player_to_move, transformed_action
    )
    seed_count = state.board[physical_action]
    transformed_seed_count = transformed_before.board[transformed_physical_action]
    original_result = original.play_local(action)
    transformed_result = transformed.play_local(transformed_action)
    expected_state = transform.transform_state(RawSongoState.from_game(original))
    actual_state = RawSongoState.from_game(transformed)

    if expected_state.board[:14] != actual_state.board[:14]:
        failures.append("pits")
    if expected_state.board[14:] != actual_state.board[14:]:
        failures.append("stores")
    if expected_state.player_to_move != actual_state.player_to_move:
        failures.append("player_to_move")
    if original.finished != transformed.finished:
        failures.append("terminal")
    if transform.transform_winner(original.winner) != transformed.winner:
        failures.append("winner")

    original_store = P1_STORE if state.player_to_move == PLAYER_ONE else P2_STORE
    transformed_store = (
        P1_STORE if transformed_before.player_to_move == PLAYER_ONE else P2_STORE
    )
    original_final_store_deposit = bool(
        original.last_sow_trace and original.last_sow_trace[-1] == original_store
    )
    transformed_final_store_deposit = bool(
        transformed.last_sow_trace and transformed.last_sow_trace[-1] == transformed_store
    )
    z_original = None
    z_transformed = None
    if original.finished and transformed.finished:
        z_original = terminal_value(original.winner, state.player_to_move)
        z_transformed = terminal_value(
            transformed.winner, transformed_before.player_to_move
        )
        if z_original != z_transformed:
            failures.append("value_target")

    return {
        "success": not failures,
        "failures": failures,
        "action": action,
        "transformed_action": transformed_action,
        "physical_action": physical_action,
        "transformed_physical_action": transformed_physical_action,
        "seed_count": seed_count,
        "transformed_seed_count": transformed_seed_count,
        "original_final_store_deposit": original_final_store_deposit,
        "transformed_final_store_deposit": transformed_final_store_deposit,
        "original_capture": original_result.captured,
        "transformed_capture": transformed_result.captured,
        "original_sow_trace": tuple(original.last_sow_trace),
        "transformed_sow_trace": tuple(transformed.last_sow_trace),
        "original_state": state,
        "transformed_state": transformed_before,
        "expected_successor": expected_state,
        "actual_successor": actual_state,
        "original_terminal": original.finished,
        "transformed_terminal": transformed.finished,
        "original_winner": original.winner,
        "transformed_winner": transformed.winner,
        "z_original": z_original,
        "z_transformed": z_transformed,
    }


def audit_state_collection(
    named_states: Iterable[tuple[str, RawSongoState]],
    transform: PlayerSwapTransform,
    *,
    max_examples: int = 20,
) -> dict:
    """Agrège légalité, transitions et fréquence par groupe/trajectoire."""

    states = list(named_states)
    legal_failures = 0
    transition_count = 0
    transition_failures = 0
    terminal_transition_count = 0
    terminal_z_failures = 0
    states_with_failure = 0
    groups_with_failure: set[str] = set()
    all_groups = {group for group, _ in states}
    failure_kinds: dict[str, int] = {}
    family_counts: dict[str, int] = {
        "final_store_deposit": 0,
        "capture_difference": 0,
        "other": 0,
    }
    examples = []

    for group, state in states:
        legal = audit_legal_equivariance(state, transform)
        if not legal["success"]:
            legal_failures += 1
        game = _normalized_game(state)
        state_failed = not legal["success"]
        if game.finished:
            if state_failed:
                groups_with_failure.add(group)
                states_with_failure += 1
            continue
        for action, is_legal in enumerate(game.legal_mask()):
            if not is_legal:
                continue
            transition_count += 1
            record = audit_transition_equivariance(state, action, transform)
            if record.get("original_terminal"):
                terminal_transition_count += 1
                if record.get("z_original") != record.get("z_transformed"):
                    terminal_z_failures += 1
            if record["success"]:
                continue
            transition_failures += 1
            state_failed = True
            groups_with_failure.add(group)
            for kind in record["failures"]:
                failure_kinds[kind] = failure_kinds.get(kind, 0) + 1
            if record.get("original_final_store_deposit") or record.get(
                "transformed_final_store_deposit"
            ):
                family_counts["final_store_deposit"] += 1
            elif record.get("original_capture") != record.get("transformed_capture"):
                family_counts["capture_difference"] += 1
            else:
                family_counts["other"] += 1
            if len(examples) < max_examples:
                examples.append(_jsonable_transition(group, record))
        if state_failed:
            states_with_failure += 1

    return {
        "transform": transform.name,
        "states_tested": len(states),
        "legal_equivariance_successes": len(states) - legal_failures,
        "legal_equivariance_failures": legal_failures,
        "legal_equivariance_failure_rate": legal_failures / len(states) if states else 0.0,
        "states_with_transition_failure": states_with_failure,
        "state_failure_rate": states_with_failure / len(states) if states else 0.0,
        "transitions_tested": transition_count,
        "transition_successes": transition_count - transition_failures,
        "transition_failures": transition_failures,
        "transition_failure_rate": (
            transition_failures / transition_count if transition_count else 0.0
        ),
        "groups_tested": len(all_groups),
        "groups_with_failure": len(groups_with_failure),
        "group_failure_rate": len(groups_with_failure) / len(all_groups) if all_groups else 0.0,
        "terminal_transitions_tested": terminal_transition_count,
        "terminal_z_failures": terminal_z_failures,
        "failure_kinds": dict(sorted(failure_kinds.items())),
        "failure_families": family_counts,
        "minimal_failure_examples": examples,
    }


def transition_is_mirror_eligible(
    state: RawSongoState, action: int, transform: PlayerSwapTransform = SWAP_KEEP_LOCAL
) -> bool:
    """Prédicat exact, fondé sur les deux transitions du moteur gelé."""

    return bool(audit_transition_equivariance(state, action, transform)["success"])


def state_is_mirror_eligible(
    state: RawSongoState, transform: PlayerSwapTransform = SWAP_KEEP_LOCAL
) -> bool:
    """Vrai si légalité et tous les successeurs légaux sont équivariants."""

    legal = audit_legal_equivariance(state, transform)
    if not legal["success"]:
        return False
    game = _normalized_game(state)
    if game.finished:
        return True
    return all(
        transition_is_mirror_eligible(state, action, transform)
        for action, is_legal in enumerate(game.legal_mask())
        if is_legal
    )


def _jsonable_transition(group: str, record: dict) -> dict:
    keys = (
        "success",
        "failures",
        "action",
        "transformed_action",
        "physical_action",
        "transformed_physical_action",
        "seed_count",
        "transformed_seed_count",
        "original_final_store_deposit",
        "transformed_final_store_deposit",
        "original_capture",
        "transformed_capture",
        "original_sow_trace",
        "transformed_sow_trace",
        "original_terminal",
        "transformed_terminal",
        "original_winner",
        "transformed_winner",
        "z_original",
        "z_transformed",
    )
    result = {"group": group}
    for key in keys:
        value = record.get(key)
        result[key] = list(value) if isinstance(value, tuple) else value
    for key in (
        "original_state",
        "transformed_state",
        "expected_successor",
        "actual_successor",
    ):
        state = record.get(key)
        if state is not None:
            result[key] = {
                "board": list(state.board),
                "player_to_move": state.player_to_move,
            }
    return result
