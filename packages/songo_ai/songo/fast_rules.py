"""Moteur de regles accelere Numba (section 4.3 : "Numba : compiler les
boucles critiques sur des tableaux numeriques compacts").

Traduction terme a terme de `songo_ai.songo.rules` (seule source de verite
logique du projet) sur des tableaux numpy int64, sans classes ni exceptions
dans les fonctions compilees @njit. Le profilage de l'etape 3
(scripts/bench_search.py) a montre qu'apres suppression du gaspillage
Python evident, le cout restant est de l'overhead d'interpreteur pur
(des centaines de milliers d'appels a de petites fonctions) : c'est
exactement la classe de probleme que Numba resout.

Pour respecter le principe "une seule implementation des regles"
(section 12.1), ce module n'est PAS une nouvelle regle inventee : sa seule
garantie de correction est le test differentiel
`songo_ai/tests/test_fast_rules_matches_reference.py`, qui rejoue des
milliers de coups aleatoires sur ce moteur et sur `SongoLegacyGame` et exige
un etat identique apres chaque coup. Tant que ce test est vert, les deux
implementations sont interchangeables ; FastSongoGame est celle utilisee en
pratique par la recherche/le professeur des que la performance compte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numba import njit

from .rules import (
    BOARD_SIZE,
    P1_STORE,
    P2_STORE,
    PLAYER_ONE,
    PLAYER_TWO,
    IllegalMove,
    MoveResult,
    State,
    local_action_to_pit,
    normalize_player,
    side_range,
    zobrist_hash,
)

REASON_FINISHED_BEFORE_MOVE = 0
REASON_PLAYER_1_STORE_OVER_35 = 1
REASON_PLAYER_2_STORE_OVER_35 = 2
REASON_STORES_35_35 = 3
REASON_FAMINE_AFTER_MOVE = 4
REASON_NEXT_TURN = 5

_REASON_STRINGS = {
    REASON_FINISHED_BEFORE_MOVE: "finished_before_move",
    REASON_PLAYER_1_STORE_OVER_35: "player_1_store_over_35",
    REASON_PLAYER_2_STORE_OVER_35: "player_2_store_over_35",
    REASON_STORES_35_35: "stores_35_35",
    REASON_FAMINE_AFTER_MOVE: "famine_after_move",
    REASON_NEXT_TURN: "next_turn",
}


@njit(cache=True)
def _owner_of_pit(index: int) -> int:
    return 1 if index <= 6 else 2


@njit(cache=True)
def _side_range(player: int) -> Tuple[int, int]:
    return (7, 13) if player == 2 else (0, 6)


@njit(cache=True)
def _sum_range(board: np.ndarray, start: int, end: int) -> int:
    total = 0
    for i in range(start, end + 1):
        total += board[i]
    return total


@njit(cache=True)
def _can_transmit_for_player(board: np.ndarray, player: int) -> bool:
    required = 1
    if player == 1:
        if board[6] > required:
            return True
        for i in range(5, -1, -1):
            if board[i] > required:
                return True
            required += 1
    else:
        if board[13] > required:
            return True
        for i in range(12, 6, -1):
            if board[i] > required:
                return True
            required += 1
    return False


@njit(cache=True)
def _can_transmit_from_side(board: np.ndarray, move: int) -> bool:
    return _can_transmit_for_player(board, _owner_of_pit(move))


@njit(cache=True)
def _sum_opponent(board: np.ndarray, move: int) -> int:
    owner = _owner_of_pit(move)
    opp = 2 if owner == 1 else 1
    start, end = _side_range(opp)
    return _sum_range(board, start, end)


@njit(cache=True)
def _sum_player_side(board: np.ndarray, move: int) -> int:
    start, end = _side_range(_owner_of_pit(move))
    return _sum_range(board, start, end)


@njit(cache=True)
def is_legal_move(board: np.ndarray, turn: int, move: int, finished: bool) -> bool:
    if finished:
        return False
    if move < 0 or move > 13:
        return False
    if _owner_of_pit(move) != turn:
        return False
    count = board[move]
    if count == 0:
        return False

    can_transmit = _can_transmit_from_side(board, move)
    opponent_side_empty = _sum_opponent(board, move) == 0

    if (move == 6 or move == 13) and count < 2 and can_transmit and opponent_side_empty:
        return False

    cannot_reach_opponent = (move < 6 and count - 6 + move <= 0) or (6 < move < 13 and count - 13 + move <= 0)
    if cannot_reach_opponent and can_transmit and opponent_side_empty:
        return False

    if (move == 6 or move == 13) and count == 1 and _sum_player_side(board, move) > 1:
        return False

    if not can_transmit and opponent_side_empty:
        return False

    return True


@njit(cache=True)
def legal_mask(board: np.ndarray, turn: int, finished: bool) -> np.ndarray:
    mask = np.zeros(7, dtype=np.bool_)
    start, _ = _side_range(turn)
    for a in range(7):
        mask[a] = is_legal_move(board, turn, start + a, finished)
    return mask


@njit(cache=True)
def _can_capture_all_opponent(board: np.ndarray, move: int) -> bool:
    owner = _owner_of_pit(move)
    opp = 2 if owner == 1 else 1
    start, end = _side_range(opp)
    for i in range(start, end + 1):
        v = board[i]
        if not (1 < v <= 4):
            return False
    return True


@njit(cache=True)
def sow(board: np.ndarray, move: int) -> int:
    seed_count = board[move]
    next_index = move + 1

    if seed_count == 1:
        if move == 6:
            next_index = P1_STORE
        elif move == 13:
            next_index = P2_STORE
        board[next_index] += 1
        board[move] = 0
        return next_index

    moved = 0
    full_turn_reference = move
    while moved < seed_count:
        if next_index == P1_STORE:
            next_index = 0

        full_turn = next_index == full_turn_reference
        if full_turn:
            if move < 7:
                full_turn_reference = 0
                if seed_count - moved == 1:
                    board[P1_STORE] += 1
                    next_index = P1_STORE
                    break
                next_index = 7
            else:
                full_turn_reference = 7
                if seed_count - moved == 1:
                    board[P2_STORE] += 1
                    next_index = P2_STORE
                    break
                next_index = 0

        board[next_index] += 1
        moved += 1
        next_index += 1

    board[move] = 0
    return next_index - 1


@njit(cache=True)
def capture(board: np.ndarray, move: int, arrival: int) -> int:
    if arrival < 0 or arrival >= BOARD_SIZE:
        return 0

    seed_count = board[arrival]
    if seed_count <= 1 or seed_count > 4 or arrival == P1_STORE or arrival == P2_STORE:
        return 0

    cross_side = (move < 7 and arrival >= 7) or (move >= 7 and arrival < 7)
    if not cross_side:
        return 0

    if arrival == 0 or arrival == 7:
        return 0

    # Protection contre le videment total (cf. songo.rules._capture pour
    # le detail) : on protege la derniere case de la cascade plutot que de
    # bloquer toute la capture.
    protected_pit = -2  # aucune valeur de `current` ne vaut jamais -2
    if (arrival == 6 or arrival == 13) and _can_capture_all_opponent(board, move):
        protected_pit = 0 if move >= 7 else 7

    captured = 0
    current = arrival
    while (current != -1 and move >= 7) or (current != 6 and move < 7):
        if current == protected_pit:
            break
        seeds = board[current]
        if seeds <= 1 or seeds > 4:
            break
        captured += seeds
        if current < 7:
            board[P2_STORE] += seeds
        else:
            board[P1_STORE] += seeds
        board[current] = 0
        current -= 1

    return captured


@njit(cache=True)
def finish_by_final_territory_and_stores(board: np.ndarray) -> int:
    score_1 = board[P1_STORE] + _sum_range(board, 0, 6)
    score_2 = board[P2_STORE] + _sum_range(board, 7, 13)
    if score_1 > score_2:
        return 1
    if score_2 > score_1:
        return 2
    return 0


@njit(cache=True)
def finish_if_current_player_has_no_seeds(board: np.ndarray, turn: int) -> bool:
    start, end = _side_range(turn)
    return _sum_range(board, start, end) == 0


@njit(cache=True)
def finish_if_current_player_wins_by_famine_no_transmit(board: np.ndarray, turn: int) -> bool:
    opp = 2 if turn == 1 else 1
    ostart, oend = _side_range(opp)
    if _sum_range(board, ostart, oend) > 0:
        return False
    cstart, cend = _side_range(turn)
    if _sum_range(board, cstart, cend) == 0:
        return False
    if _can_transmit_for_player(board, turn):
        return False
    return True


@njit(cache=True)
def finish_after_move_for_famine(board: np.ndarray, move: int) -> bool:
    next_player = 1 if move >= 7 else 2
    nstart, nend = _side_range(next_player)
    if _sum_range(board, nstart, nend) == 0:
        return True
    opp = 2 if next_player == 1 else 1
    ostart, oend = _side_range(opp)
    if _sum_range(board, ostart, oend) > 0:
        return False
    if _can_transmit_for_player(board, next_player):
        return False
    return True


@njit(cache=True)
def apply_move(board: np.ndarray, move: int) -> Tuple[int, bool, int, int]:
    """Mutates board in place. L'appelant doit avoir deja verifie la
    legalite et gere les cas "finished avant le coup". Renvoie
    (captured, finished, winner_or_-1, reason_code)."""
    arrival = sow(board, move)
    captured = capture(board, move, arrival)

    if board[P1_STORE] > 35:
        return captured, True, 1, REASON_PLAYER_1_STORE_OVER_35
    if board[P2_STORE] > 35:
        return captured, True, 2, REASON_PLAYER_2_STORE_OVER_35
    if board[P1_STORE] == 35 and board[P2_STORE] == 35:
        return captured, True, 0, REASON_STORES_35_35

    if finish_after_move_for_famine(board, move):
        winner = finish_by_final_territory_and_stores(board)
        return captured, True, winner, REASON_FAMINE_AFTER_MOVE

    return captured, False, -1, REASON_NEXT_TURN


# ---------------------------------------------------------------------------
# Wrapper Python : meme API publique que SongoLegacyGame (section 12.1 :
# duck-typing suffisant, songo_ai.search.negamax n'a pas besoin d'un
# changement pour utiliser ce moteur a la place du moteur de reference).
# ---------------------------------------------------------------------------


@dataclass
class FastSongoGame:
    board: np.ndarray
    turn: int = PLAYER_ONE
    finished: bool = False
    winner: Optional[int] = None
    history: List[MoveResult] = field(default_factory=list)
    record_history: bool = True

    @classmethod
    def initial(cls) -> "FastSongoGame":
        return cls(np.array([5] * 14 + [0, 0], dtype=np.int64), PLAYER_ONE)

    @classmethod
    def from_board(cls, board, turn: int = PLAYER_ONE) -> "FastSongoGame":
        arr = np.array(list(board), dtype=np.int64)
        if arr.shape[0] != BOARD_SIZE:
            raise ValueError("board must contain exactly 16 integers")
        if (arr < 0).any():
            raise ValueError("board cannot contain negative seed counts")
        if turn not in (PLAYER_ONE, PLAYER_TWO):
            raise ValueError("turn must be 1 or 2")
        return cls(arr, turn)

    @classmethod
    def from_state(cls, state: State) -> "FastSongoGame":
        return cls.from_board(state.board, state.turn)

    def clone(self) -> "FastSongoGame":
        return FastSongoGame(self.board.copy(), self.turn, self.finished, self.winner, list(self.history), self.record_history)

    def clone_for_search(self) -> "FastSongoGame":
        return FastSongoGame(self.board.copy(), self.turn, self.finished, self.winner, [], record_history=False)

    def to_state(self) -> State:
        return State(tuple(self.board.tolist()), self.turn)

    def zobrist_hash(self) -> int:
        return zobrist_hash(self.to_state())

    def normalize_terminal(self) -> None:
        if self.finished:
            return
        if finish_if_current_player_has_no_seeds(self.board, self.turn):
            self.finished = True
            self.winner = finish_by_final_territory_and_stores(self.board)
            return
        if finish_if_current_player_wins_by_famine_no_transmit(self.board, self.turn):
            self.finished = True
            self.winner = finish_by_final_territory_and_stores(self.board)

    def is_legal_move(self, move: int) -> bool:
        return bool(is_legal_move(self.board, self.turn, move, self.finished))

    def validate_move(self, move: int) -> None:
        if not self.is_legal_move(move):
            raise IllegalMove(f"illegal move {move}")

    def legal_moves(self, player: Optional[int] = None) -> List[int]:
        player = self.turn if player is None else normalize_player(player)
        start, end = side_range(player)
        return [i for i in range(start, end + 1) if self.is_legal_move(i)]

    def legal_mask(self) -> Tuple[bool, ...]:
        return tuple(bool(x) for x in legal_mask(self.board, self.turn, self.finished))

    def legal_local_actions(self) -> List[int]:
        mask = self.legal_mask()
        return [a for a in range(7) if mask[a]]

    def play(self, move: int) -> MoveResult:
        if not self.finished:
            if finish_if_current_player_has_no_seeds(self.board, self.turn):
                self.finished = True
                self.winner = finish_by_final_territory_and_stores(self.board)
            elif finish_if_current_player_wins_by_famine_no_transmit(self.board, self.turn):
                self.finished = True
                self.winner = finish_by_final_territory_and_stores(self.board)

        if self.finished:
            return self._record(move, self.turn, None, 0, True, self.winner, "finished_before_move")

        self.validate_move(move)
        player = self.turn
        captured, finished, winner_code, reason_code = apply_move(self.board, move)
        reason = _REASON_STRINGS[reason_code]

        if finished:
            self.finished = True
            self.winner = winner_code
            next_player = None
        else:
            self.turn = PLAYER_ONE if move >= 7 else PLAYER_TWO
            next_player = self.turn

        return self._record(move, player, next_player, captured, finished, self.winner, reason)

    def play_local(self, local_action: int) -> MoveResult:
        return self.play(local_action_to_pit(self.turn, local_action))

    def can_transmit_for_player(self, player: int) -> bool:
        return bool(_can_transmit_for_player(self.board, normalize_player(player)))

    def can_transmit_from_side(self, move: int) -> bool:
        return bool(_can_transmit_from_side(self.board, move))

    def can_capture_all_opponent(self, move: int) -> bool:
        return bool(_can_capture_all_opponent(self.board, move))

    def sum_opponent(self, move: int) -> int:
        return int(_sum_opponent(self.board, move))

    def sum_player_side(self, move: int) -> int:
        return int(_sum_player_side(self.board, move))

    def score(self) -> Tuple[int, int]:
        return int(self.board[P1_STORE]), int(self.board[P2_STORE])

    def final_score_with_territory(self) -> Tuple[int, int]:
        return (
            int(self.board[P1_STORE] + _sum_range(self.board, 0, 6)),
            int(self.board[P2_STORE] + _sum_range(self.board, 7, 13)),
        )

    def _record(self, move, player, next_player, captured, finished, winner, reason) -> MoveResult:
        # .tolist() est une conversion native numpy (C), bien plus rapide
        # qu'une comprehension Python avec int() element par element.
        result = MoveResult(move, player, next_player, int(captured), finished, winner, reason, self.board.tolist())
        if self.record_history:
            self.history.append(result)
        return result


def warmup() -> None:
    """Force la compilation JIT une fois (a appeler avant tout benchmark)."""
    game = FastSongoGame.initial()
    for _ in range(3):
        legal = game.legal_local_actions()
        if not legal or game.finished:
            game = FastSongoGame.initial()
            continue
        game.play_local(legal[0])
