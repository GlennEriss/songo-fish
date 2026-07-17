"""Anime un coup case par case au lieu de sauter directement au resultat
final : chaque graine deposee l'une apres l'autre (sowing), puis chaque
case capturee l'une apres l'autre (capture). Consomme les traces exposees
par songo_ai.songo.rules.SongoLegacyGame (last_sow_trace/last_capture_trace)
-- aucune logique de regles dupliquee ici, juste du rythme d'affichage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from songo_ai.songo.rules import P1_STORE, P2_STORE

SOW_STEP_SECONDS = 0.22
CAPTURE_STEP_SECONDS = 0.30


@dataclass
class MoveAnimation:
    board: List[int]
    final_board: List[int]
    source_pit: int
    sow_trace: List[int]
    capture_trace: List[int]
    phase: str = "sowing"  # "sowing" | "capturing" | "done"
    step_index: int = 0
    elapsed: float = 0.0
    receiving_pit: Optional[int] = None

    @classmethod
    def start(
        cls,
        board_before: List[int],
        source_pit: int,
        sow_trace: List[int],
        capture_trace: List[int],
        final_board: List[int],
    ) -> "MoveAnimation":
        board = list(board_before)
        board[source_pit] = 0  # les graines sont "ramassees" d'un coup, avant d'etre semees une a une

        anim = cls(
            board=board,
            final_board=list(final_board),
            source_pit=source_pit,
            sow_trace=list(sow_trace),
            capture_trace=list(capture_trace),
        )
        if not anim.sow_trace:
            anim.phase = "capturing" if anim.capture_trace else "done"
        return anim

    @property
    def finished(self) -> bool:
        return self.phase == "done"

    def update(self, dt: float) -> None:
        if self.phase == "done":
            return
        self.elapsed += dt

        if self.phase == "sowing":
            self._advance_sowing()
        elif self.phase == "capturing":
            self._advance_capturing()

        if self.phase == "done":
            # Filet de securite : l'etat affiche doit toujours finir par
            # correspondre exactement a l'etat reel du moteur de regles.
            self.board = list(self.final_board)

    def _advance_sowing(self) -> None:
        while self.elapsed >= SOW_STEP_SECONDS and self.step_index < len(self.sow_trace):
            pit = self.sow_trace[self.step_index]
            self.board[pit] += 1
            self.receiving_pit = pit
            self.step_index += 1
            self.elapsed -= SOW_STEP_SECONDS

        if self.step_index >= len(self.sow_trace):
            self.receiving_pit = None
            if self.capture_trace:
                self.phase = "capturing"
                self.step_index = 0
                self.elapsed = 0.0
            else:
                self.phase = "done"

    def _advance_capturing(self) -> None:
        while self.elapsed >= CAPTURE_STEP_SECONDS and self.step_index < len(self.capture_trace):
            pit = self.capture_trace[self.step_index]
            seeds = self.board[pit]
            self.board[pit] = 0
            store_index = P2_STORE if pit < 7 else P1_STORE
            self.board[store_index] += seeds
            self.receiving_pit = pit
            self.step_index += 1
            self.elapsed -= CAPTURE_STEP_SECONDS

        if self.step_index >= len(self.capture_trace):
            self.receiving_pit = None
            self.phase = "done"
