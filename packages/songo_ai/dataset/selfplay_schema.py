"""Contrat de donnees du futur pipeline RL Songo.

Ce module est volontairement independant du schema teacher historique de
``dataset.schema``. Il conserve l'etat moteur brut (16 compteurs et joueur au
trait) et autorise les exemples incomplets produits pendant une partie, avant
que MCTS ou le resultat terminal ne soient disponibles.

Il ne contient ni conversion depuis les annotations Minimax, ni generation de
pseudo-cibles. ``D_LAB`` reste le corpus historique/laboratoire ; ``D_RL``
designera uniquement les experiences produites par le futur self-play.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple

from songo_ai.songo.rules import (
    BOARD_SIZE,
    NUM_ACTIONS,
    PLAYER_ONE,
    PLAYER_TWO,
    TOTAL_SEEDS,
    State,
)


class DatasetFamily(str, Enum):
    """Familles de donnees maintenues separement par le projet."""

    LAB = "D_LAB"
    RL = "D_RL"


D_LAB = DatasetFamily.LAB
D_RL = DatasetFamily.RL


@dataclass(frozen=True)
class RawSongoState:
    """Etat moteur non canonicalise.

    ``board`` conserve les cases physiques 0..13 puis les magasins 14 et 15.
    ``player_to_move`` conserve explicitement le joueur physique au trait.
    """

    board: Tuple[int, ...]
    player_to_move: int

    def __post_init__(self) -> None:
        if len(self.board) != BOARD_SIZE:
            raise ValueError(f"board must contain {BOARD_SIZE} counters")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in self.board):
            raise TypeError("board counters must be integers")
        if any(value < 0 for value in self.board):
            raise ValueError("board counters must be non-negative")
        if sum(self.board) != TOTAL_SEEDS:
            raise ValueError(f"board must conserve {TOTAL_SEEDS} seeds")
        if self.player_to_move not in (PLAYER_ONE, PLAYER_TWO):
            raise ValueError("player_to_move must be PLAYER_ONE or PLAYER_TWO")

    @classmethod
    def from_engine_state(cls, state: State) -> "RawSongoState":
        return cls(tuple(int(value) for value in state.board), int(state.turn))

    @classmethod
    def from_game(cls, game) -> "RawSongoState":
        """Construit l'etat depuis l'interface commune des deux moteurs."""

        return cls(tuple(int(value) for value in game.board), int(game.turn))

    def to_engine_state(self) -> State:
        return State(self.board, self.player_to_move)


def _tuple_of_bools(values: Sequence[bool], field_name: str) -> Tuple[bool, ...]:
    if len(values) != NUM_ACTIONS:
        raise ValueError(f"{field_name} must contain {NUM_ACTIONS} entries")
    return tuple(bool(value) for value in values)


def _optional_float_tuple(
    values: Optional[Sequence[float]], field_name: str
) -> Optional[Tuple[float, ...]]:
    if values is None:
        return None
    if len(values) != NUM_ACTIONS:
        raise ValueError(f"{field_name} must contain {NUM_ACTIONS} entries")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{field_name} must contain only finite values")
    return result


def _optional_int_tuple(
    values: Optional[Sequence[int]], field_name: str
) -> Optional[Tuple[int, ...]]:
    if values is None:
        return None
    if len(values) != NUM_ACTIONS:
        raise ValueError(f"{field_name} must contain {NUM_ACTIONS} entries")
    result = tuple(int(value) for value in values)
    if any(value < 0 for value in result):
        raise ValueError(f"{field_name} must contain non-negative counts")
    return result


@dataclass(frozen=True)
class RLTrainingExample:
    """Contrat conceptuel ``(S, M, pi, z, metadata)`` du futur ``D_RL``.

    ``policy_target`` et ``value_target`` sont optionnels : pendant une partie,
    le resultat terminal n'est pas encore connu et MCTS n'existe pas encore
    dans le depot. ``visit_counts`` est egalement optionnel jusqu'a
    l'implementation de la recherche.
    """

    state: RawSongoState
    legal_mask: Tuple[bool, ...]
    policy_target: Optional[Tuple[float, ...]] = None
    value_target: Optional[float] = None
    visit_counts: Optional[Tuple[int, ...]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    dataset_family: DatasetFamily = field(default=DatasetFamily.RL, init=False)

    def __post_init__(self) -> None:
        legal_mask = _tuple_of_bools(self.legal_mask, "legal_mask")
        policy_target = _optional_float_tuple(self.policy_target, "policy_target")
        visit_counts = _optional_int_tuple(self.visit_counts, "visit_counts")
        metadata = dict(self.metadata)

        if policy_target is not None:
            if any(value < 0.0 for value in policy_target):
                raise ValueError("policy_target cannot contain negative probabilities")
            if not math.isclose(sum(policy_target), 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("policy_target must sum to 1")
            if any(policy_target[a] != 0.0 for a, legal in enumerate(legal_mask) if not legal):
                raise ValueError("policy_target must be zero on illegal actions")

        value_target = self.value_target
        if value_target is not None:
            value_target = float(value_target)
            if not math.isfinite(value_target) or not -1.0 <= value_target <= 1.0:
                raise ValueError("value_target must be finite and in [-1, 1]")

        if visit_counts is not None:
            if any(visit_counts[a] != 0 for a, legal in enumerate(legal_mask) if not legal):
                raise ValueError("visit_counts must be zero on illegal actions")

        object.__setattr__(self, "legal_mask", legal_mask)
        object.__setattr__(self, "policy_target", policy_target)
        object.__setattr__(self, "value_target", value_target)
        object.__setattr__(self, "visit_counts", visit_counts)
        object.__setattr__(self, "metadata", metadata)

    @property
    def is_search_complete(self) -> bool:
        return self.visit_counts is not None and self.policy_target is not None

    @property
    def is_terminally_labeled(self) -> bool:
        return self.value_target is not None

