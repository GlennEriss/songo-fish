"""Moteur de regles Songo (etape 1 du plan directeur SongoFish).

Portage durci de la sandbox de reference `songo/docs/songo_legacy_single.py`
(source de verite unique pour la sowing/capture/famine/transmission logic ;
la partie C# Unity ne fait plus foi ici). Ce module ajoute par-dessus la
logique identique de la sandbox :

- une normalisation systematique de l'etat terminal avant tout calcul de
  coups legaux (section 4.2, point 1) ;
- un masque de legalite booleen sans exceptions dans le chemin chaud
  (section 4.2, point 4), la levee d'exception restant reservee a l'API
  amicale `validate_move` ;
- un etat compact et immuable (`State`) pour la recherche, copiable a cout
  quasi nul, et une variante de jeu sans historique (section 4.2, points 3
  et 5) ;
- un hachage Zobrist deterministe pour la table de transposition
  (section 4.2, point 6) ;
- un detecteur de repetition separe de l'etat de jeu (section 4.2, point 2).
  La regle de cycle n'existe pas dans le jeu de production actuel : ce
  detecteur est fourni comme mecanisme optionnel pour la recherche/le
  self-play, il ne modifie pas play()/is_terminal() par defaut.

Invariants controles par les tests (section 4.3) : taille 16, valeurs >= 0,
somme des 16 compteurs == 70 apres chaque coup depuis la position initiale,
determinisme (meme etat + meme coup -> meme resultat), aucun coup hors du
masque legal, aucun coup applique sur une position terminale.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, List, NamedTuple, Optional, Tuple

PLAYER_ONE = 1
PLAYER_TWO = 2
DRAW = 0

P1_STORE = 14
P2_STORE = 15
BOARD_SIZE = 16
NUM_ACTIONS = 7
TOTAL_SEEDS = 70


class IllegalMove(ValueError):
    pass


def normalize_player(player: int) -> int:
    return PLAYER_TWO if player == PLAYER_TWO else PLAYER_ONE


def opponent(player: int) -> int:
    return PLAYER_ONE if normalize_player(player) == PLAYER_TWO else PLAYER_TWO


def owner_of_pit(index: int) -> int:
    if 0 <= index <= 6:
        return PLAYER_ONE
    if 7 <= index <= 13:
        return PLAYER_TWO
    raise ValueError("stores do not belong to a playable side")


def side_range(player: int) -> Tuple[int, int]:
    return (7, 13) if normalize_player(player) == PLAYER_TWO else (0, 6)


def local_action_to_pit(player: int, local_action: int) -> int:
    """Action locale 0..6 (section 4.1) -> index absolu 0..13 du plateau."""
    if not 0 <= local_action <= 6:
        raise ValueError("local_action must be in 0..6")
    start, _ = side_range(player)
    return start + local_action


def pit_to_local_action(pit: int) -> int:
    start, _ = side_range(owner_of_pit(pit))
    return pit - start


class State(NamedTuple):
    """Etat compact et immuable : copiable a cout quasi nul pour la recherche."""

    board: Tuple[int, ...]
    turn: int

    @staticmethod
    def initial() -> "State":
        return State(tuple([5] * 14 + [0, 0]), PLAYER_ONE)


def assert_invariants(board: Iterable[int]) -> None:
    values = list(board)
    assert len(values) == BOARD_SIZE, f"board must have {BOARD_SIZE} counters, got {len(values)}"
    assert all(v >= 0 for v in values), "board counters must be >= 0"
    assert sum(values) == TOTAL_SEEDS, f"seed conservation violated: sum={sum(values)}"


# ---------------------------------------------------------------------------
# Zobrist hashing (deterministe : graine fixe, stable d'un run a l'autre).
# ---------------------------------------------------------------------------

_ZOBRIST_SEED = 0x536F6E676F  # "Songo" en hexa, arbitraire mais fixe.
_ZOBRIST_RNG = random.Random(_ZOBRIST_SEED)
# Une valeur par (case, nombre de graines possibles jusqu'a 70) + une par joueur.
_ZOBRIST_CELL = [[_ZOBRIST_RNG.getrandbits(64) for _ in range(TOTAL_SEEDS + 1)] for _ in range(BOARD_SIZE)]
_ZOBRIST_TURN = {PLAYER_ONE: _ZOBRIST_RNG.getrandbits(64), PLAYER_TWO: _ZOBRIST_RNG.getrandbits(64)}


def zobrist_hash(state: State) -> int:
    h = _ZOBRIST_TURN[normalize_player(state.turn)]
    for index, count in enumerate(state.board):
        h ^= _ZOBRIST_CELL[index][count]
    return h


class RepetitionTracker:
    """Compteur separe (position, joueur au trait) -> occurrences.

    Non branche par defaut sur les regles de fin de partie : la regle de
    cycle est encore a trancher (cf. section 14, risque "Regle de cycle non
    formalisee"). A utiliser explicitement cote recherche/self-play si une
    regle de nulle par repetition est decidee.
    """

    def __init__(self) -> None:
        self._counts: dict[int, int] = {}

    def record(self, state: State) -> int:
        key = zobrist_hash(state)
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def count(self, state: State) -> int:
        return self._counts.get(zobrist_hash(state), 0)

    def reset(self) -> None:
        self._counts.clear()


@dataclass(frozen=True)
class MoveResult:
    move: int
    player: int
    next_player: Optional[int]
    captured: int
    finished: bool
    winner: Optional[int]
    reason: str
    board: List[int]


@dataclass
class SongoLegacyGame:
    """API de jeu avec historique (compatible `songo_legacy_single.py`).

    Toute la logique de sowing/capture/famine reste identique bit a bit a la
    sandbox de reference : seule l'organisation (masque booleen, clone leger,
    hash) a ete ajoutee autour.
    """

    board: List[int] = field(default_factory=lambda: [5] * 14 + [0, 0])
    turn: int = PLAYER_ONE
    finished: bool = False
    winner: Optional[int] = None
    history: List[MoveResult] = field(default_factory=list)
    record_history: bool = True
    # Trace du dernier coup joue (case par case, dans l'ordre), pour toute
    # UI qui veut animer la distribution/capture au lieu de sauter
    # directement a l'etat final (ex: apps/table). Purement additif : ne
    # change ni la signature ni le comportement de _sow()/_capture().
    last_sow_trace: List[int] = field(default_factory=list)
    last_capture_trace: List[int] = field(default_factory=list)

    @classmethod
    def from_board(cls, board: Iterable[int], turn: int = PLAYER_ONE) -> "SongoLegacyGame":
        values = list(board)
        if len(values) != BOARD_SIZE:
            raise ValueError("board must contain exactly 16 integers")
        if any(v < 0 for v in values):
            raise ValueError("board cannot contain negative seed counts")
        if turn not in (PLAYER_ONE, PLAYER_TWO):
            raise ValueError("turn must be 1 or 2")
        return cls(values, turn)

    @classmethod
    def from_state(cls, state: State) -> "SongoLegacyGame":
        return cls.from_board(state.board, state.turn)

    def clone(self) -> "SongoLegacyGame":
        return SongoLegacyGame(
            list(self.board), self.turn, self.finished, self.winner, list(self.history), self.record_history
        )

    def clone_for_search(self) -> "SongoLegacyGame":
        """Variante sans historique : pas d'allocation de liste par coup (section 4.2)."""
        return SongoLegacyGame(list(self.board), self.turn, self.finished, self.winner, [], record_history=False)

    def to_state(self) -> State:
        return State(tuple(self.board), self.turn)

    def zobrist_hash(self) -> int:
        return zobrist_hash(self.to_state())

    # -- Terminal state normalization -------------------------------------

    def normalize_terminal(self) -> None:
        """Recalcule finished/winner sans jouer de coup (section 4.2, point 1).

        A appeler avant toute requete de legalite sur un etat potentiellement
        importe d'un snapshot externe (ex: reconstruit depuis Firestore).
        """
        if self.finished:
            return
        if self._finish_if_current_player_has_no_seeds():
            return
        self._finish_if_current_player_wins_by_famine_no_transmit()

    # -- Legality (boolean fast path + friendly exception path) ----------

    def _illegality_reason(self, move: int) -> Optional[str]:
        if self.finished:
            return "match already finished"
        if move < 0 or move > 13:
            return "move must be a pit index from 0 to 13"
        if owner_of_pit(move) != self.turn:
            return f"pit {move} does not belong to player {self.turn}"
        count = self.board[move]
        if count == 0:
            return "empty pit"

        # can_transmit_from_side()/sum_opponent() sont des fonctions pures de
        # l'etat courant (pas d'effet de bord) : les calculer une seule fois
        # evite jusqu'a 3 recalculs identiques (mesure par profilage, cf.
        # scripts/bench_search.py) sans changer le resultat.
        can_transmit = self.can_transmit_from_side(move)
        opponent_side_empty = self.sum_opponent(move) == 0

        if move in (6, 13) and count < 2 and can_transmit and opponent_side_empty:
            return "must transmit seeds to opponent"

        cannot_reach_opponent = (move < 6 and count - 6 + move <= 0) or (6 < move < 13 and count - 13 + move <= 0)
        if cannot_reach_opponent and can_transmit and opponent_side_empty:
            return "must transmit seeds to opponent"

        if move in (6, 13) and count == 1 and self.sum_player_side(move) > 1:
            return "single seed in edge pit is forbidden while side has other seeds"

        if not can_transmit and opponent_side_empty:
            return "must transmit seeds to opponent"

        return None

    def is_legal_move(self, move: int) -> bool:
        return self._illegality_reason(move) is None

    def validate_move(self, move: int) -> None:
        reason = self._illegality_reason(move)
        if reason is not None:
            raise IllegalMove(reason)

    def legal_moves(self, player: Optional[int] = None) -> List[int]:
        player = self.turn if player is None else normalize_player(player)
        start, end = side_range(player)
        return [i for i in range(start, end + 1) if self.is_legal_move(i)]

    def legal_mask(self) -> Tuple[bool, ...]:
        """Masque booleen sur les 7 actions locales (0..6) du joueur au trait."""
        start, _ = side_range(self.turn)
        return tuple(self.is_legal_move(start + a) for a in range(NUM_ACTIONS))

    def legal_local_actions(self) -> List[int]:
        return [a for a in range(NUM_ACTIONS) if self.legal_mask()[a]]

    # -- Play --------------------------------------------------------------

    def play(self, move: int) -> MoveResult:
        self._finish_if_current_player_has_no_seeds()
        self._finish_if_current_player_wins_by_famine_no_transmit()
        if self.finished:
            return self._record(move, self.turn, None, 0, True, self.winner, "finished_before_move")

        self.validate_move(move)
        player = self.turn
        arrival = self._sow(move)
        captured = self._capture(move, arrival)

        if self.board[P1_STORE] > 35:
            self.finished = True
            self.winner = PLAYER_ONE
            reason = "player_1_store_over_35"
        elif self.board[P2_STORE] > 35:
            self.finished = True
            self.winner = PLAYER_TWO
            reason = "player_2_store_over_35"
        elif self.board[P1_STORE] == 35 and self.board[P2_STORE] == 35:
            self.finished = True
            self.winner = DRAW
            reason = "stores_35_35"
        elif self._finish_after_move_for_famine(move):
            reason = "famine_after_move"
        else:
            self.turn = PLAYER_ONE if move >= 7 else PLAYER_TWO
            reason = "next_turn"

        return self._record(move, player, None if self.finished else self.turn, captured, self.finished, self.winner, reason)

    def play_local(self, local_action: int) -> MoveResult:
        return self.play(local_action_to_pit(self.turn, local_action))

    def _sow(self, move: int) -> int:
        self.last_sow_trace = []
        seed_count = self.board[move]
        next_index = move + 1

        if seed_count == 1:
            if move in (6, 13):
                next_index = P1_STORE if move == 6 else P2_STORE
            self.board[next_index] += 1
            self.board[move] = 0
            self.last_sow_trace.append(next_index)
            return next_index

        moved = 0
        full_turn_reference = move
        for _ in range(seed_count):
            if next_index == P1_STORE:
                next_index = 0

            full_turn = next_index == full_turn_reference
            if full_turn:
                if move < 7:
                    full_turn_reference = 0
                    if seed_count - moved == 1:
                        self.board[P1_STORE] += 1
                        next_index = P1_STORE
                        self.last_sow_trace.append(next_index)
                        break
                    next_index = 7
                else:
                    full_turn_reference = 7
                    if seed_count - moved == 1:
                        self.board[P2_STORE] += 1
                        next_index = P2_STORE
                        self.last_sow_trace.append(next_index)
                        break
                    next_index = 0

            self.board[next_index] += 1
            self.last_sow_trace.append(next_index)
            moved += 1
            next_index += 1

        self.board[move] = 0
        return next_index - 1

    def _capture(self, move: int, arrival: int) -> int:
        self.last_capture_trace = []

        if arrival < 0 or arrival >= BOARD_SIZE:
            return 0

        seed_count = self.board[arrival]
        if seed_count <= 1 or seed_count > 4 or arrival in (P1_STORE, P2_STORE):
            return 0

        cross_side = (move < 7 <= arrival) or (move >= 7 and arrival < 7)
        if not cross_side:
            return 0

        if arrival in (0, 7):
            return 0

        if arrival in (6, 13) and self.can_capture_all_opponent(move):
            return 0

        captured = 0
        current = arrival
        while (current != -1 and move >= 7) or (current != 6 and move < 7):
            seeds = self.board[current]
            if seeds <= 1 or seeds > 4:
                break

            captured += seeds
            if current < 7:
                self.board[P2_STORE] += seeds
            else:
                self.board[P1_STORE] += seeds
            self.board[current] = 0
            self.last_capture_trace.append(current)
            current -= 1

        return captured

    def _finish_if_current_player_has_no_seeds(self) -> bool:
        if self.finished:
            return False
        start, end = side_range(self.turn)
        if sum(self.board[start : end + 1]) > 0:
            return False
        self._finish_by_final_territory_and_stores()
        return True

    def _finish_if_current_player_wins_by_famine_no_transmit(self) -> bool:
        if self.finished:
            return False
        current_start, current_end = side_range(self.turn)
        opponent_start, opponent_end = side_range(opponent(self.turn))
        if sum(self.board[opponent_start : opponent_end + 1]) > 0:
            return False
        if sum(self.board[current_start : current_end + 1]) == 0:
            return False
        if self.can_transmit_for_player(self.turn):
            return False
        self._finish_by_final_territory_and_stores()
        return True

    def _finish_after_move_for_famine(self, move: int) -> bool:
        next_player = PLAYER_ONE if move >= 7 else PLAYER_TWO
        next_start, next_end = side_range(next_player)
        if sum(self.board[next_start : next_end + 1]) == 0:
            self._finish_by_final_territory_and_stores()
            return True

        opponent_start, opponent_end = side_range(opponent(next_player))
        if sum(self.board[opponent_start : opponent_end + 1]) > 0:
            return False
        if self.can_transmit_for_player(next_player):
            return False

        self._finish_by_final_territory_and_stores()
        return True

    def _finish_by_final_territory_and_stores(self) -> None:
        score_1 = self.board[P1_STORE] + sum(self.board[0:7])
        score_2 = self.board[P2_STORE] + sum(self.board[7:14])
        self.finished = True
        if score_1 > score_2:
            self.winner = PLAYER_ONE
        elif score_2 > score_1:
            self.winner = PLAYER_TWO
        else:
            self.winner = DRAW

    def can_transmit_for_player(self, player: int) -> bool:
        player = normalize_player(player)
        required = 1
        if player == PLAYER_ONE:
            if self.board[6] > required:
                return True
            for i in range(5, -1, -1):
                if self.board[i] > required:
                    return True
                required += 1
        else:
            if self.board[13] > required:
                return True
            for i in range(12, 6, -1):
                if self.board[i] > required:
                    return True
                required += 1
        return False

    def can_transmit_from_side(self, move: int) -> bool:
        return self.can_transmit_for_player(owner_of_pit(move))

    def can_capture_all_opponent(self, move: int) -> bool:
        start, end = side_range(opponent(owner_of_pit(move)))
        return all(1 < self.board[i] <= 4 for i in range(start, end + 1))

    def sum_opponent(self, move: int) -> int:
        start, end = side_range(opponent(owner_of_pit(move)))
        return sum(self.board[start : end + 1])

    def sum_player_side(self, move: int) -> int:
        start, end = side_range(owner_of_pit(move))
        return sum(self.board[start : end + 1])

    def score(self) -> Tuple[int, int]:
        return self.board[P1_STORE], self.board[P2_STORE]

    def final_score_with_territory(self) -> Tuple[int, int]:
        return self.board[P1_STORE] + sum(self.board[0:7]), self.board[P2_STORE] + sum(self.board[7:14])

    def _record(
        self,
        move: int,
        player: int,
        next_player: Optional[int],
        captured: int,
        finished: bool,
        winner: Optional[int],
        reason: str,
    ) -> MoveResult:
        result = MoveResult(move, player, next_player, captured, finished, winner, reason, list(self.board))
        if self.record_history:
            self.history.append(result)
        return result


def format_board(board: List[int]) -> str:
    top = " ".join(f"{board[i]:2d}" for i in range(13, 6, -1))
    bottom = " ".join(f"{board[i]:2d}" for i in range(0, 7))
    return (
        f"          P2 pits: {top}\n"
        f"P2 store {board[P2_STORE]:2d}                         P1 store {board[P1_STORE]:2d}\n"
        f"          P1 pits: {bottom}"
    )
