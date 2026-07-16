#!/usr/bin/env python3
"""
Songo legacy rules sandbox.

Single-file Python port of the Unity legacy board logic, without minimax,
Unity objects, UI, audio, persistence, or ads.

Board indices:
  0..6   player 1 pits
  7..13  player 2 pits
  14     player 1 store
  15     player 2 store
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional


PLAYER_ONE = 1
PLAYER_TWO = 2
DRAW = 0

P1_STORE = 14
P2_STORE = 15
BOARD_SIZE = 16


class IllegalMove(ValueError):
    pass


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
    # Default legacy setup: 14 playable pits with 5 seeds each, then two stores.
    board: List[int] = field(default_factory=lambda: [5] * 14 + [0, 0])
    turn: int = PLAYER_ONE
    finished: bool = False
    winner: Optional[int] = None
    history: List[MoveResult] = field(default_factory=list)

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

    def clone(self) -> "SongoLegacyGame":
        return SongoLegacyGame(list(self.board), self.turn, self.finished, self.winner, list(self.history))

    def legal_moves(self, player: Optional[int] = None) -> List[int]:
        # Compute legal moves by running the same validation used before play().
        player = self.turn if player is None else normalize_player(player)
        start, end = side_range(player)
        return [i for i in range(start, end + 1) if self.is_legal_move(i)]

    def is_legal_move(self, move: int) -> bool:
        try:
            self.validate_move(move)
            return True
        except IllegalMove:
            return False

    def validate_move(self, move: int) -> None:
        # Port of Table.TryValidateLegacyMove. It raises instead of calling
        # Unity's message/replay UI flow.
        if self.finished:
            raise IllegalMove("match already finished")
        if move < 0 or move > 13:
            raise IllegalMove("move must be a pit index from 0 to 13")
        if owner_of_pit(move) != self.turn:
            raise IllegalMove(f"pit {move} does not belong to player {self.turn}")
        count = self.board[move]
        if count == 0:
            raise IllegalMove("empty pit")

        # Legacy Table.TryValidateLegacyMove.
        if move in (6, 13) and count < 2 and self.can_transmit_from_side(move) and self.sum_opponent(move) == 0:
            raise IllegalMove("must transmit seeds to opponent")

        cannot_reach_opponent = (move < 6 and count - 6 + move <= 0) or (6 < move < 13 and count - 13 + move <= 0)
        if cannot_reach_opponent and self.can_transmit_from_side(move) and self.sum_opponent(move) == 0:
            raise IllegalMove("must transmit seeds to opponent")

        if move in (6, 13) and count == 1 and self.sum_player_side(move) > 1:
            raise IllegalMove("single seed in edge pit is forbidden while side has other seeds")

        if not self.can_transmit_from_side(move) and self.sum_opponent(move) == 0:
            raise IllegalMove("must transmit seeds to opponent")

    def play(self, move: int) -> MoveResult:
        # Port of Match.jouerTable + Table.parcourirLaTable + Case.deplacerLesPionsCase
        # + Table.mangerLesPions + Match.verifierEtatDuMatch.
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

    def _sow(self, move: int) -> int:
        # Port of Case.deplacerLesPionsCase, reduced to counts.
        # Returns the arrival index of the last deposited seed.
        seed_count = self.board[move]
        next_index = move + 1

        if seed_count == 1:
            # Legacy special case: one seed from edge pit goes into the player's store.
            if move in (6, 13):
                next_index = P1_STORE if move == 6 else P2_STORE
            self.board[next_index] += 1
            self.board[move] = 0
            return next_index

        moved = 0
        full_turn_reference = move
        for _ in range(seed_count):
            # In the legacy board loop, index 14 wraps to 0. Index 15 is only
            # reached by the "last seed after full turn" special case.
            if next_index == P1_STORE:
                next_index = 0

            full_turn = next_index == full_turn_reference
            if full_turn:
                # When a full lap reaches the original side, jump to opponent
                # territory unless this is the final seed, which goes to store.
                if move < 7:
                    full_turn_reference = 0
                    if seed_count - moved == 1:
                        self.board[P1_STORE] += 1
                        next_index = P1_STORE
                        break
                    next_index = 7
                else:
                    full_turn_reference = 7
                    if seed_count - moved == 1:
                        self.board[P2_STORE] += 1
                        next_index = P2_STORE
                        break
                    next_index = 0

            self.board[next_index] += 1
            moved += 1
            next_index += 1

        self.board[move] = 0
        return next_index - 1

    def _capture(self, move: int, arrival: int) -> int:
        # Port of Table.mangerLesPions. Captures run backward from the arrival
        # pit while pits contain 2..4 seeds.
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
            # Legacy safeguard: do not capture if it would take the whole
            # opponent row from the edge condition.
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
            current -= 1

        return captured

    def _finish_if_current_player_has_no_seeds(self) -> bool:
        # Port of Match.TryFinishIfCurrentTurnPlayerHasNoSeeds.
        if self.finished:
            return False
        start, end = side_range(self.turn)
        if sum(self.board[start : end + 1]) > 0:
            return False
        self._finish_by_final_territory_and_stores()
        return True

    def _finish_if_current_player_wins_by_famine_no_transmit(self) -> bool:
        # Port of Match.TryFinishIfCurrentTurnPlayerWinsByFamineNoTransmit.
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
        # Port of Match.TryFinishImmediatelyAfterMoveForFamineRules.
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
        # Legacy final famine score: stores + seeds remaining on each territory.
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
        # Port of Table.peutTransmettrePion / Match.CanTransmitForPlayer.
        # It checks the whole side, not only a single selected pit.
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

    def score(self) -> tuple[int, int]:
        return self.board[P1_STORE], self.board[P2_STORE]

    def final_score_with_territory(self) -> tuple[int, int]:
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
        self.history.append(result)
        return result


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


def side_range(player: int) -> tuple[int, int]:
    return (7, 13) if normalize_player(player) == PLAYER_TWO else (0, 6)


def format_board(board: List[int]) -> str:
    # Human-oriented display: player 2 is printed in reverse, like a board view.
    top = " ".join(f"{board[i]:2d}" for i in range(13, 6, -1))
    bottom = " ".join(f"{board[i]:2d}" for i in range(0, 7))
    return (
        f"          P2 pits: {top}\n"
        f"P2 store {board[P2_STORE]:2d}                         P1 store {board[P1_STORE]:2d}\n"
        f"          P1 pits: {bottom}"
    )


def run_cli() -> None:
    # Tiny REPL for working outside Unity:
    #   number -> play pit
    #   moves  -> list legal moves
    #   board  -> print raw 16-int board
    #   quit   -> exit
    game = SongoLegacyGame()
    print("Songo legacy sandbox. Type a pit index, 'moves', 'board', or 'quit'.")
    while True:
        print()
        print(format_board(game.board))
        if game.finished:
            label = "draw" if game.winner == DRAW else f"player {game.winner}"
            print(f"Finished: winner={label}, final_score={game.final_score_with_territory()}")
            return

        moves = game.legal_moves()
        raw = input(f"Player {game.turn} move {moves}> ").strip().lower()
        if raw in ("q", "quit", "exit"):
            return
        if raw == "moves":
            print(moves)
            continue
        if raw == "board":
            print(game.board)
            continue
        try:
            move = int(raw)
            result = game.play(move)
            print(
                f"played={result.move} captured={result.captured} "
                f"reason={result.reason} next={result.next_player}"
            )
        except (ValueError, IllegalMove) as exc:
            print(f"Illegal move: {exc}")


if __name__ == "__main__":
    run_cli()
