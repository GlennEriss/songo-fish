"""Self-play local et production controlee d'exemples ``D_RL``.

Ce module est separe des trajectoires teacher historiques. Il orchestre le
moteur gele, MCTS et le contrat ``RLTrainingExample`` sans effectuer
d'entrainement ni gerer de Replay Buffer.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Protocol, Sequence

from songo_ai.search.mcts import MCTSConfig, SongoMCTS, terminal_value, visit_counts_to_policy
from songo_ai.songo.rules import DRAW, NUM_ACTIONS, IllegalMove, SongoLegacyGame

if TYPE_CHECKING:
    from torch import nn

    from songo_ai.dataset.selfplay_schema import RLTrainingExample, RawSongoState
    from songo_ai.model.srn_graph import SongoGraphBuilder
    from songo_ai.search.mcts import MCTSResult


class SelfPlayStatus(str, Enum):
    TERMINAL_WIN = "TERMINAL_WIN"
    TERMINAL_DRAW = "TERMINAL_DRAW"
    TRUNCATED_MAX_PLIES = "TRUNCATED_MAX_PLIES"
    TRUNCATED_REPETITION = "TRUNCATED_REPETITION"

    @property
    def is_terminal(self) -> bool:
        return self in (SelfPlayStatus.TERMINAL_WIN, SelfPlayStatus.TERMINAL_DRAW)

    @property
    def is_truncated(self) -> bool:
        return not self.is_terminal


@dataclass(frozen=True)
class SelfPlayConfig:
    """Configuration locale du self-play ; ``mcts`` n'est pas dupliquee."""

    games: int = 1
    max_game_plies: int = 400
    repetition_limit: int = 3
    target_temperature: float = 1.0
    action_temperature: float = 1.0
    temperature_drop_ply: int = 30
    late_action_temperature: float = 0.0
    seed: int = 0
    generation: int = 0
    checkpoint_id: str = "untrained"
    provenance: Mapping[str, object] = field(default_factory=dict)
    include_truncated_examples: bool = True
    mcts: MCTSConfig = field(
        default_factory=lambda: MCTSConfig(add_root_noise=True)
    )

    def __post_init__(self) -> None:
        if self.games <= 0:
            raise ValueError("games must be positive")
        if self.max_game_plies <= 0:
            raise ValueError("max_game_plies must be positive")
        if self.repetition_limit < 2:
            raise ValueError("repetition_limit must be at least 2")
        for name, temperature in (
            ("target_temperature", self.target_temperature),
            ("action_temperature", self.action_temperature),
            ("late_action_temperature", self.late_action_temperature),
        ):
            if not math.isfinite(temperature) or temperature < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.temperature_drop_ply < 0:
            raise ValueError("temperature_drop_ply must be non-negative")
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must not be empty")
        object.__setattr__(self, "provenance", dict(self.provenance))

    def action_temperature_at(self, ply: int) -> float:
        if ply < 0:
            raise ValueError("ply must be non-negative")
        return (
            self.action_temperature
            if ply < self.temperature_drop_ply
            else self.late_action_temperature
        )


@dataclass(frozen=True)
class PendingSelfPlayStep:
    """Position temporaire avant attribution du resultat terminal ``z``."""

    state: "RawSongoState"
    legal_mask: tuple[bool, ...]
    visit_counts: tuple[int, ...]
    policy_target: tuple[float, ...]
    play_policy: tuple[float, ...]
    action_played: int
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class SelfPlayGameResult:
    game_id: str
    status: SelfPlayStatus
    winner: Optional[int]
    final_score: Optional[tuple[int, int]]
    num_plies: int
    examples: tuple["RLTrainingExample", ...]
    action_sequence: tuple[int, ...]
    total_mcts_simulations: int
    total_network_evaluations: int
    average_branching_factor: float
    elapsed_s: float
    truncation_reason: Optional[str] = None


@dataclass(frozen=True)
class SelfPlayStatistics:
    games_completed: int
    games_truncated: int
    wins_p1: int
    wins_p2: int
    draws: int
    truncations_repetition: int
    truncations_max_plies: int
    total_positions: int
    positions_encountered: int
    average_game_length: float
    median_game_length: float
    min_game_length: int
    max_game_length: int
    average_branching_factor: float
    total_mcts_simulations: int
    total_network_evaluations: int
    elapsed_s: float
    positions_per_second: float
    length_distribution: Mapping[int, int]


@dataclass(frozen=True)
class SelfPlayRunResult:
    games: tuple[SelfPlayGameResult, ...]
    examples: tuple["RLTrainingExample", ...]
    statistics: SelfPlayStatistics


class SearchLike(Protocol):
    def search(self, state: "RawSongoState", *, policy_temperature: float = 1.0) -> "MCTSResult": ...


SearchFactory = Callable[[MCTSConfig], SearchLike]


def select_action_from_policy(
    policy: Sequence[float],
    legal_mask: Sequence[bool],
    rng: random.Random,
) -> int:
    """Echantillonne strictement parmi les actions legales."""

    if len(policy) != NUM_ACTIONS or len(legal_mask) != NUM_ACTIONS:
        raise ValueError(f"policy and legal_mask must contain {NUM_ACTIONS} entries")
    probabilities = tuple(float(value) for value in policy)
    mask = tuple(bool(value) for value in legal_mask)
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError("policy must contain finite non-negative probabilities")
    if any(probabilities[action] != 0.0 for action, legal in enumerate(mask) if not legal):
        raise IllegalMove("play policy assigns probability to an illegal action")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("play policy must sum to 1")

    threshold = rng.random()
    cumulative = 0.0
    last_legal_with_mass: Optional[int] = None
    for action, probability in enumerate(probabilities):
        if mask[action] and probability > 0.0:
            last_legal_with_mass = action
            cumulative += probability
            if threshold < cumulative:
                return action
    if last_legal_with_mass is None:
        raise ValueError("play policy has no probability mass on legal actions")
    return last_legal_with_mass  # Tolerance aux seuls arrondis flottants.


def finalize_selfplay_steps(
    steps: Sequence[PendingSelfPlayStep],
    *,
    status: SelfPlayStatus,
    winner: Optional[int],
    final_score: Optional[tuple[int, int]],
    include_truncated_examples: bool,
) -> tuple["RLTrainingExample", ...]:
    """Attribue ``z`` uniquement depuis un resultat terminal reel."""

    from songo_ai.dataset.selfplay_schema import RLTrainingExample

    if status.is_terminal:
        if winner not in (1, 2, DRAW):
            raise ValueError("a terminal game must expose winner P1, P2 or DRAW")
    elif winner is not None:
        raise ValueError("a truncated game must not expose a winner")
    if status.is_truncated and not include_truncated_examples:
        return ()

    examples = []
    for step in steps:
        value_target = (
            terminal_value(winner, step.state.player_to_move)
            if status.is_terminal
            else None
        )
        metadata = dict(step.metadata)
        metadata.update(
            {
                "status": status.value,
                "winner": winner,
                "terminal_result": winner if status.is_terminal else None,
                "final_score": list(final_score) if final_score is not None else None,
                "play_policy": list(step.play_policy),
            }
        )
        examples.append(
            RLTrainingExample(
                state=step.state,
                legal_mask=step.legal_mask,
                visit_counts=step.visit_counts,
                policy_target=step.policy_target,
                value_target=value_target,
                metadata=metadata,
            )
        )
    return tuple(examples)


class SelfPlayRunner:
    """Orchestrateur d'une ou plusieurs parties locales de self-play."""

    def __init__(
        self,
        model: "nn.Module",
        config: Optional[SelfPlayConfig] = None,
        graph_builder: Optional["SongoGraphBuilder"] = None,
        game_factory: Callable[[], SongoLegacyGame] = SongoLegacyGame,
        search_factory: Optional[SearchFactory] = None,
        models_by_player: Optional[Mapping[int, "nn.Module"]] = None,
    ) -> None:
        self.model = model
        self.config = config or SelfPlayConfig()
        if graph_builder is None:
            from songo_ai.model.srn_graph import SongoGraphBuilder

            graph_builder = SongoGraphBuilder()
        self.graph_builder = graph_builder
        self.game_factory = game_factory
        self.search_factory = search_factory or (
            lambda mcts_config: SongoMCTS(self.model, self.graph_builder, mcts_config)
        )
        self.models_by_player = dict(models_by_player or {})
        if self.models_by_player and set(self.models_by_player) != {1, 2}:
            raise ValueError("models_by_player must define exactly P1 and P2")

    def play_game(self, game_index: int = 0) -> SelfPlayGameResult:
        if game_index < 0:
            raise ValueError("game_index must be non-negative")
        started_at = time.perf_counter()
        game_id = self._game_id(game_index)
        action_rng = random.Random(self._derived_seed(game_index, 0, "actions"))
        game = self.game_factory()
        game.normalize_terminal()
        repetitions = Counter({self._repetition_key(game): 1})
        steps: list[PendingSelfPlayStep] = []
        actions: list[int] = []
        total_simulations = 0
        total_network_evaluations = 0
        total_legal_actions = 0
        status: Optional[SelfPlayStatus] = self._terminal_status(game) if game.finished else None
        truncation_reason: Optional[str] = None

        for ply in range(self.config.max_game_plies):
            if status is not None:
                break
            legal_mask = tuple(game.legal_mask())
            if not any(legal_mask):
                game.normalize_terminal()
                if not game.finished:
                    raise RuntimeError("engine exposed no legal action without a terminal state")
                status = self._terminal_status(game)
                break

            from songo_ai.dataset.selfplay_schema import RawSongoState

            state = RawSongoState.from_game(game)
            search_seed = self._derived_seed(game_index, ply, "mcts")
            search_config = replace(self.config.mcts, seed=search_seed)
            search = (
                SongoMCTS(self.models_by_player[state.player_to_move], self.graph_builder, search_config)
                if self.models_by_player
                else self.search_factory(search_config)
            )
            search_result = search.search(
                state,
                policy_temperature=self.config.target_temperature,
            )
            visit_counts = tuple(int(value) for value in search_result.visit_counts)
            policy_target = visit_counts_to_policy(
                visit_counts,
                legal_mask,
                self.config.target_temperature,
            )
            action_temperature = self.config.action_temperature_at(ply)
            play_policy = visit_counts_to_policy(
                visit_counts,
                legal_mask,
                action_temperature,
            )
            action = select_action_from_policy(play_policy, legal_mask, action_rng)
            if not legal_mask[action]:
                raise IllegalMove(f"self-play selected illegal local action {action}")

            metadata = {
                "game_id": game_id,
                "ply": ply,
                "generation": self.config.generation,
                "checkpoint_id": self.config.checkpoint_id,
                "player_to_move": state.player_to_move,
                "action_played": action,
                "mcts_simulations": search_result.num_simulations,
                "action_temperature": action_temperature,
                "target_temperature": self.config.target_temperature,
                "root_value": search_result.root_value,
                "search_nodes": search_result.num_nodes,
                "network_evaluations": search_result.network_evaluations,
                "selfplay_seed": self.config.seed,
                "c_puct": search_config.c_puct,
                "dirichlet_alpha": search_config.dirichlet_alpha,
                "dirichlet_epsilon": search_config.dirichlet_epsilon,
                "root_noise": search_config.add_root_noise,
                "acting_model_id": self.config.provenance.get(
                    "p1_model_id" if state.player_to_move == 1 else "p2_model_id",
                    self.config.checkpoint_id,
                ),
            }
            metadata.update(self.config.provenance)
            steps.append(
                PendingSelfPlayStep(
                    state=state,
                    legal_mask=legal_mask,
                    visit_counts=visit_counts,
                    policy_target=policy_target,
                    play_policy=play_policy,
                    action_played=action,
                    metadata=metadata,
                )
            )
            actions.append(action)
            total_legal_actions += sum(legal_mask)
            total_simulations += search_result.num_simulations
            total_network_evaluations += search_result.network_evaluations

            game.play_local(action)
            if game.finished:
                status = self._terminal_status(game)
                break

            key = self._repetition_key(game)
            repetitions[key] += 1
            if repetitions[key] >= self.config.repetition_limit:
                status = SelfPlayStatus.TRUNCATED_REPETITION
                truncation_reason = (
                    f"state repeated {repetitions[key]} times "
                    f"(technical limit={self.config.repetition_limit})"
                )
                break

        if status is None:
            status = SelfPlayStatus.TRUNCATED_MAX_PLIES
            truncation_reason = f"max_game_plies={self.config.max_game_plies} reached"

        winner = game.winner if status.is_terminal else None
        final_score = game.final_score_with_territory() if status.is_terminal else None
        examples = finalize_selfplay_steps(
            steps,
            status=status,
            winner=winner,
            final_score=final_score,
            include_truncated_examples=self.config.include_truncated_examples,
        )
        elapsed = time.perf_counter() - started_at
        return SelfPlayGameResult(
            game_id=game_id,
            status=status,
            winner=winner,
            final_score=final_score,
            num_plies=len(steps),
            examples=examples,
            action_sequence=tuple(actions),
            total_mcts_simulations=total_simulations,
            total_network_evaluations=total_network_evaluations,
            average_branching_factor=total_legal_actions / len(steps) if steps else 0.0,
            elapsed_s=elapsed,
            truncation_reason=truncation_reason,
        )

    def generate(self) -> SelfPlayRunResult:
        started_at = time.perf_counter()
        games = tuple(self.play_game(index) for index in range(self.config.games))
        examples = tuple(example for game in games for example in game.examples)
        elapsed = time.perf_counter() - started_at
        lengths = [game.num_plies for game in games]
        completed = [game for game in games if game.status.is_terminal]
        truncated = [game for game in games if game.status.is_truncated]
        positions_encountered = sum(lengths)
        branching_total = sum(game.average_branching_factor * game.num_plies for game in games)
        statistics_result = SelfPlayStatistics(
            games_completed=len(completed),
            games_truncated=len(truncated),
            wins_p1=sum(game.winner == 1 for game in completed),
            wins_p2=sum(game.winner == 2 for game in completed),
            draws=sum(game.winner == DRAW for game in completed),
            truncations_repetition=sum(
                game.status is SelfPlayStatus.TRUNCATED_REPETITION for game in games
            ),
            truncations_max_plies=sum(
                game.status is SelfPlayStatus.TRUNCATED_MAX_PLIES for game in games
            ),
            total_positions=len(examples),
            positions_encountered=positions_encountered,
            average_game_length=statistics.fmean(lengths),
            median_game_length=float(statistics.median(lengths)),
            min_game_length=min(lengths),
            max_game_length=max(lengths),
            average_branching_factor=(
                branching_total / positions_encountered if positions_encountered else 0.0
            ),
            total_mcts_simulations=sum(game.total_mcts_simulations for game in games),
            total_network_evaluations=sum(
                game.total_network_evaluations for game in games
            ),
            elapsed_s=elapsed,
            positions_per_second=len(examples) / elapsed if elapsed > 0.0 else float("inf"),
            length_distribution=dict(sorted(Counter(lengths).items())),
        )
        return SelfPlayRunResult(games=games, examples=examples, statistics=statistics_result)

    def _game_id(self, game_index: int) -> str:
        payload = (
            f"{self.config.seed}:{self.config.generation}:"
            f"{self.config.checkpoint_id}:{game_index}"
        )
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
        return f"selfplay-g{self.config.generation}-{game_index}-{digest}"

    def _derived_seed(self, game_index: int, ply: int, purpose: str) -> int:
        payload = f"{self.config.seed}:{game_index}:{ply}:{purpose}"
        return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")

    @staticmethod
    def _repetition_key(game: SongoLegacyGame) -> tuple[tuple[int, ...], int]:
        return tuple(int(value) for value in game.board), int(game.turn)

    @staticmethod
    def _terminal_status(game: SongoLegacyGame) -> SelfPlayStatus:
        if not game.finished or game.winner not in (1, 2, DRAW):
            raise ValueError("engine terminal state must expose a valid winner")
        return (
            SelfPlayStatus.TERMINAL_DRAW
            if game.winner == DRAW
            else SelfPlayStatus.TERMINAL_WIN
        )
