"""Evaluation strategique controlee des agents SRN + MCTS.

Ce module est volontairement distinct du tournoi historique. Il conserve les
troncatures comme des resultats techniques, construit des matches apparies a
partir de positions initiales fixees et reechantillonne les *positions de
depart* (pas les parties individuelles) pour estimer l'incertitude.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Mapping, Optional, Protocol, Sequence

from songo_ai.dataset.selfplay_schema import RawSongoState
from songo_ai.search.mcts import MCTSConfig, SongoMCTS
from songo_ai.songo.rules import DRAW, PLAYER_ONE, PLAYER_TWO, SongoLegacyGame


class ArenaStatus(str, Enum):
    TERMINAL_WIN = "TERMINAL_WIN"
    TERMINAL_DRAW = "TERMINAL_DRAW"
    TRUNCATED_MAX_PLIES = "TRUNCATED_MAX_PLIES"
    TRUNCATED_REPETITION = "TRUNCATED_REPETITION"

    @property
    def is_terminal(self) -> bool:
        return self in (ArenaStatus.TERMINAL_WIN, ArenaStatus.TERMINAL_DRAW)


@dataclass(frozen=True)
class ArenaOpening:
    opening_id: str
    state: RawSongoState
    prefix_actions: tuple[int, ...]
    requested_prefix_length: int


@dataclass(frozen=True)
class ArenaDecision:
    action: int
    mcts_simulations: int = 0
    network_evaluations: int = 0
    network_forward_calls: int = 0
    search_nodes: int = 0
    search_time_s: float = field(default=0.0, compare=False)


class ArenaAgent(Protocol):
    name: str

    def select_action(self, state: RawSongoState, *, seed: int) -> ArenaDecision: ...


@dataclass
class SRNMCTSAgent:
    """Agent d'evaluation : aucun bruit, action = argmax des visites."""

    name: str
    model: object
    num_simulations: int
    c_puct: float = 1.5
    graph_builder: object | None = None

    def __post_init__(self) -> None:
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if not math.isfinite(self.c_puct) or self.c_puct < 0.0:
            raise ValueError("c_puct must be finite and non-negative")

    @property
    def mcts_config(self) -> MCTSConfig:
        return MCTSConfig(
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            add_root_noise=False,
        )

    def select_action(self, state: RawSongoState, *, seed: int) -> ArenaDecision:
        config = MCTSConfig(
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            add_root_noise=False,
            seed=seed,
        )
        search = SongoMCTS(self.model, self.graph_builder, config)
        counters_before = getattr(self.model, "counters", None)
        result = search.search(state, policy_temperature=0.0)
        counters_after = getattr(self.model, "counters", None)
        if result.selected_action is None:
            raise RuntimeError("MCTS did not return an action for a non-terminal state")
        return ArenaDecision(
            action=result.selected_action,
            mcts_simulations=result.num_simulations,
            network_evaluations=result.network_evaluations,
            network_forward_calls=(
                counters_after.network_forward_calls - counters_before.network_forward_calls
                if counters_before is not None and counters_after is not None
                else result.network_evaluations
            ),
            search_nodes=result.num_nodes,
            search_time_s=result.elapsed_s,
        )


@dataclass(frozen=True)
class RandomLegalAgent:
    name: str = "random-legal"

    def select_action(self, state: RawSongoState, *, seed: int) -> ArenaDecision:
        game = SongoLegacyGame.from_state(state.to_engine_state())
        game.normalize_terminal()
        legal = game.legal_local_actions()
        if not legal:
            raise RuntimeError("random agent received a terminal state")
        return ArenaDecision(random.Random(seed).choice(legal))


@dataclass(frozen=True)
class ArenaConfig:
    max_plies: int = 400
    repetition_limit: int = 3
    seed: int = 0
    bootstrap_samples: int = 10_000
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.max_plies <= 0:
            raise ValueError("max_plies must be positive")
        if self.repetition_limit < 2:
            raise ValueError("repetition_limit must be at least 2")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")


@dataclass(frozen=True)
class ArenaGameResult:
    opening_id: str
    agent_a: str
    agent_b: str
    a_player: int
    status: ArenaStatus
    winner: Optional[int]
    continuation_plies: int
    prefix_plies: int
    action_sequence: tuple[int, ...]
    total_mcts_simulations: int
    total_network_evaluations: int
    elapsed_s: float
    total_network_forward_calls: int = 0
    search_nodes_by_agent: Mapping[str, int] = None
    search_time_s_by_agent: Mapping[str, float] = None

    @property
    def a_outcome(self) -> Optional[float]:
        """Score de A (1/0.5/0) ou ``None`` si la partie est tronquee."""

        if not self.status.is_terminal:
            return None
        if self.winner == DRAW:
            return 0.5
        return 1.0 if self.winner == self.a_player else 0.0


@dataclass(frozen=True)
class SideSummary:
    games: int
    terminal_games: int
    wins: int
    draws: int
    losses: int
    truncated_repetition: int
    truncated_max_plies: int


@dataclass(frozen=True)
class LengthSummary:
    mean: float
    median: float
    minimum: int
    maximum: int


@dataclass(frozen=True)
class ArenaSummary:
    agent_a: str
    agent_b: str
    openings: int
    games: int
    by_a_side: Mapping[str, SideSummary]
    aggregate: SideSummary
    score_rate_a_terminal: Optional[float]
    paired_bootstrap_ci: Optional[tuple[float, float]]
    bootstrap_unit: str
    effective_opening_clusters: int
    lengths: LengthSummary
    distinct_trajectories: int
    total_mcts_simulations: int
    total_network_evaluations: int
    total_network_forward_calls: int
    elapsed_s: float


def _derived_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join(str(part) for part in (base_seed,) + parts)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def generate_deterministic_openings(
    *,
    prefix_lengths: Sequence[int],
    seed: int,
) -> tuple[ArenaOpening, ...]:
    """Genere des positions legales par prefixes uniformes fixes.

    Chaque position utilise un RNG propre derive de ``(seed, index,
    longueur)``. Aucun modele ni resultat de match n'intervient dans leur
    construction. La position initiale (prefixe 0) est acceptee au plus une
    fois ; les autres etats doivent etre distincts et non terminaux.
    """

    if not prefix_lengths:
        raise ValueError("prefix_lengths must not be empty")
    if any(length < 0 for length in prefix_lengths):
        raise ValueError("prefix lengths must be non-negative")
    openings: list[ArenaOpening] = []
    seen_states: set[tuple[tuple[int, ...], int]] = set()
    for index, requested_length in enumerate(prefix_lengths):
        game = SongoLegacyGame()
        rng = random.Random(_derived_seed(seed, "opening", index, requested_length))
        actions: list[int] = []
        for _ in range(requested_length):
            game.normalize_terminal()
            if game.finished:
                raise ValueError(
                    f"prefix {index} reached a terminal before {requested_length} plies"
                )
            legal = game.legal_local_actions()
            action = rng.choice(legal)
            game.play_local(action)
            actions.append(action)
        game.normalize_terminal()
        if game.finished:
            raise ValueError(f"prefix {index} ends on a terminal state")
        key = (tuple(game.board), game.turn)
        if key in seen_states:
            raise ValueError("deterministic opening generation produced a duplicate state")
        seen_states.add(key)
        digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:12]
        openings.append(
            ArenaOpening(
                opening_id=f"opening-{index:02d}-p{requested_length}-{digest}",
                state=RawSongoState.from_game(game),
                prefix_actions=tuple(actions),
                requested_prefix_length=requested_length,
            )
        )
    return tuple(openings)


def generate_unique_deterministic_openings(
    *,
    count: int,
    seed: int,
    max_prefix_length: int = 40,
    max_attempts: int = 10_000,
) -> tuple[ArenaOpening, ...]:
    """Génère ``count`` ouvertures uniques avec rejet déterministe.

    Les candidats dépendent uniquement de la seed et de leur indice. Les états
    terminaux ou déjà vus sont ignorés ; aucune sortie de modèle n'intervient.
    """

    if count <= 0 or max_prefix_length <= 0 or max_attempts < count:
        raise ValueError("invalid unique opening generation bounds")
    openings: list[ArenaOpening] = []
    seen_states: set[tuple[tuple[int, ...], int]] = set()
    for candidate_index in range(max_attempts):
        requested_length = 0 if candidate_index == 0 else 1 + (
            candidate_index % max_prefix_length
        )
        game = SongoLegacyGame()
        rng = random.Random(
            _derived_seed(seed, "unique-opening", candidate_index, requested_length)
        )
        actions: list[int] = []
        valid = True
        for _ in range(requested_length):
            game.normalize_terminal()
            if game.finished:
                valid = False
                break
            action = rng.choice(game.legal_local_actions())
            game.play_local(action)
            actions.append(action)
        game.normalize_terminal()
        if not valid or game.finished:
            continue
        key = (tuple(game.board), game.turn)
        if key in seen_states:
            continue
        seen_states.add(key)
        digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:12]
        openings.append(
            ArenaOpening(
                opening_id=(
                    f"opening-{len(openings):03d}-candidate-{candidate_index}-"
                    f"p{requested_length}-{digest}"
                ),
                state=RawSongoState.from_game(game),
                prefix_actions=tuple(actions),
                requested_prefix_length=requested_length,
            )
        )
        if len(openings) == count:
            return tuple(openings)
    raise RuntimeError(
        f"could generate only {len(openings)} unique openings in {max_attempts} attempts"
    )


def validate_opening(opening: ArenaOpening) -> None:
    """Rejoue le prefixe et exige l'identite exacte de l'etat annonce."""

    game = SongoLegacyGame()
    for action in opening.prefix_actions:
        if action not in game.legal_local_actions():
            raise ValueError(f"opening {opening.opening_id} contains an illegal action")
        game.play_local(action)
    game.normalize_terminal()
    if game.finished:
        raise ValueError(f"opening {opening.opening_id} is terminal")
    if RawSongoState.from_game(game) != opening.state:
        raise ValueError(f"opening {opening.opening_id} does not match its prefix")


def play_arena_game(
    opening: ArenaOpening,
    *,
    p1_agent: ArenaAgent,
    p2_agent: ArenaAgent,
    agent_a_name: str,
    agent_b_name: str,
    a_player: int,
    config: ArenaConfig,
) -> ArenaGameResult:
    if a_player not in (PLAYER_ONE, PLAYER_TWO):
        raise ValueError("a_player must be PLAYER_ONE or PLAYER_TWO")
    if {p1_agent.name, p2_agent.name} != {agent_a_name, agent_b_name}:
        raise ValueError("physical agents do not match agent A/B names")
    validate_opening(opening)
    game = SongoLegacyGame.from_state(opening.state.to_engine_state())
    repetitions = Counter({(tuple(game.board), game.turn): 1})
    actions: list[int] = []
    total_simulations = 0
    total_network_evaluations = 0
    total_network_forward_calls = 0
    search_nodes_by_agent = Counter()
    search_time_s_by_agent = Counter()
    status: Optional[ArenaStatus] = None
    started_at = time.perf_counter()

    for ply in range(config.max_plies):
        game.normalize_terminal()
        if game.finished:
            status = (
                ArenaStatus.TERMINAL_DRAW
                if game.winner == DRAW
                else ArenaStatus.TERMINAL_WIN
            )
            break
        legal = game.legal_local_actions()
        if not legal:
            raise RuntimeError("engine exposed no legal action without a terminal state")
        agent = p1_agent if game.turn == PLAYER_ONE else p2_agent
        # Meme seed de tie-break pour les deux inversions de cote d'une paire.
        # Les trajectoires peuvent ensuite diverger, mais pas la configuration.
        decision_seed = _derived_seed(config.seed, opening.opening_id, ply)
        decision = agent.select_action(RawSongoState.from_game(game), seed=decision_seed)
        if decision.action not in legal:
            raise ValueError(
                f"agent {agent.name!r} returned illegal action {decision.action}; legal={legal}"
            )
        game.play_local(decision.action)
        actions.append(decision.action)
        total_simulations += decision.mcts_simulations
        total_network_evaluations += decision.network_evaluations
        total_network_forward_calls += decision.network_forward_calls
        search_nodes_by_agent[agent.name] += decision.search_nodes
        search_time_s_by_agent[agent.name] += decision.search_time_s
        if game.finished:
            status = (
                ArenaStatus.TERMINAL_DRAW
                if game.winner == DRAW
                else ArenaStatus.TERMINAL_WIN
            )
            break
        key = (tuple(game.board), game.turn)
        repetitions[key] += 1
        if repetitions[key] >= config.repetition_limit:
            status = ArenaStatus.TRUNCATED_REPETITION
            break
    if status is None:
        status = ArenaStatus.TRUNCATED_MAX_PLIES

    winner = game.winner if status.is_terminal else None
    return ArenaGameResult(
        opening_id=opening.opening_id,
        agent_a=agent_a_name,
        agent_b=agent_b_name,
        a_player=a_player,
        status=status,
        winner=winner,
        continuation_plies=len(actions),
        prefix_plies=len(opening.prefix_actions),
        action_sequence=tuple(actions),
        total_mcts_simulations=total_simulations,
        total_network_evaluations=total_network_evaluations,
        elapsed_s=time.perf_counter() - started_at,
        total_network_forward_calls=total_network_forward_calls,
        search_nodes_by_agent=dict(search_nodes_by_agent),
        search_time_s_by_agent=dict(search_time_s_by_agent),
    )


def run_paired_arena(
    agent_a: ArenaAgent,
    agent_b: ArenaAgent,
    openings: Sequence[ArenaOpening],
    *,
    config: ArenaConfig,
) -> tuple[ArenaGameResult, ...]:
    """Joue exactement deux parties par depart en inversant P1/P2."""

    if agent_a.name == agent_b.name:
        raise ValueError("agent names must be distinct")
    if not openings:
        raise ValueError("openings must not be empty")
    if isinstance(agent_a, SRNMCTSAgent) and isinstance(agent_b, SRNMCTSAgent):
        if agent_a.num_simulations != agent_b.num_simulations:
            raise ValueError("paired SRN agents must use the same MCTS simulation budget")
        if agent_a.c_puct != agent_b.c_puct:
            raise ValueError("paired SRN agents must use the same c_puct")
    results: list[ArenaGameResult] = []
    for opening in openings:
        results.append(
            play_arena_game(
                opening,
                p1_agent=agent_a,
                p2_agent=agent_b,
                agent_a_name=agent_a.name,
                agent_b_name=agent_b.name,
                a_player=PLAYER_ONE,
                config=config,
            )
        )
        results.append(
            play_arena_game(
                opening,
                p1_agent=agent_b,
                p2_agent=agent_a,
                agent_a_name=agent_a.name,
                agent_b_name=agent_b.name,
                a_player=PLAYER_TWO,
                config=config,
            )
        )
    return tuple(results)


def _side_summary(results: Sequence[ArenaGameResult]) -> SideSummary:
    outcomes = [result.a_outcome for result in results]
    terminal = [outcome for outcome in outcomes if outcome is not None]
    return SideSummary(
        games=len(results),
        terminal_games=len(terminal),
        wins=sum(outcome == 1.0 for outcome in terminal),
        draws=sum(outcome == 0.5 for outcome in terminal),
        losses=sum(outcome == 0.0 for outcome in terminal),
        truncated_repetition=sum(
            result.status == ArenaStatus.TRUNCATED_REPETITION for result in results
        ),
        truncated_max_plies=sum(
            result.status == ArenaStatus.TRUNCATED_MAX_PLIES for result in results
        ),
    )


def paired_bootstrap_interval(
    results: Sequence[ArenaGameResult],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> Optional[tuple[float, float]]:
    """IC percentile par cluster ``opening_id``.

    Les deux inversions de cote d'un meme depart restent toujours ensemble.
    Une troncature n'est jamais transformee en 0.5 et est exclue du
    denominateur terminal de son cluster.
    """

    by_opening: dict[str, list[float]] = {}
    for result in results:
        outcome = result.a_outcome
        if outcome is not None:
            by_opening.setdefault(result.opening_id, []).append(outcome)
        else:
            by_opening.setdefault(result.opening_id, [])
    cluster_scores = [
        sum(values) / len(values) for _, values in sorted(by_opening.items()) if values
    ]
    if not cluster_scores:
        return None
    rng = random.Random(seed)
    estimates = []
    n = len(cluster_scores)
    for _ in range(samples):
        sampled = [cluster_scores[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(sampled) / n)
    estimates.sort()
    alpha = (1.0 - confidence_level) / 2.0
    low_index = max(0, min(samples - 1, int(math.floor(alpha * samples))))
    high_index = max(0, min(samples - 1, int(math.ceil((1.0 - alpha) * samples)) - 1))
    return estimates[low_index], estimates[high_index]


def summarize_arena(
    results: Sequence[ArenaGameResult],
    *,
    config: ArenaConfig,
) -> ArenaSummary:
    if not results:
        raise ValueError("cannot summarize an empty arena")
    names = {(result.agent_a, result.agent_b) for result in results}
    if len(names) != 1:
        raise ValueError("all results must belong to one confrontation")
    agent_a, agent_b = next(iter(names))
    p1_results = [result for result in results if result.a_player == PLAYER_ONE]
    p2_results = [result for result in results if result.a_player == PLAYER_TWO]
    aggregate = _side_summary(results)
    score_sum = sum(result.a_outcome for result in results if result.a_outcome is not None)
    score_rate = score_sum / aggregate.terminal_games if aggregate.terminal_games else None
    lengths = [result.continuation_plies for result in results]
    opening_ids = {result.opening_id for result in results}
    terminal_opening_ids = {
        result.opening_id for result in results if result.a_outcome is not None
    }
    trajectories = {
        (result.opening_id, result.a_player, result.action_sequence) for result in results
    }
    return ArenaSummary(
        agent_a=agent_a,
        agent_b=agent_b,
        openings=len(opening_ids),
        games=len(results),
        by_a_side={"P1": _side_summary(p1_results), "P2": _side_summary(p2_results)},
        aggregate=aggregate,
        score_rate_a_terminal=score_rate,
        paired_bootstrap_ci=paired_bootstrap_interval(
            results,
            samples=config.bootstrap_samples,
            confidence_level=config.confidence_level,
            seed=_derived_seed(config.seed, agent_a, agent_b, "bootstrap"),
        ),
        bootstrap_unit="opening_pair",
        effective_opening_clusters=len(terminal_opening_ids),
        lengths=LengthSummary(
            mean=statistics.fmean(lengths),
            median=statistics.median(lengths),
            minimum=min(lengths),
            maximum=max(lengths),
        ),
        distinct_trajectories=len(trajectories),
        total_mcts_simulations=sum(result.total_mcts_simulations for result in results),
        total_network_evaluations=sum(result.total_network_evaluations for result in results),
        total_network_forward_calls=sum(
            result.total_network_forward_calls for result in results
        ),
        elapsed_s=sum(result.elapsed_s for result in results),
    )


def opening_to_dict(opening: ArenaOpening) -> dict:
    return {
        "opening_id": opening.opening_id,
        "state": {
            "board": list(opening.state.board),
            "player_to_move": opening.state.player_to_move,
        },
        "prefix_actions": list(opening.prefix_actions),
        "requested_prefix_length": opening.requested_prefix_length,
    }


def game_result_to_dict(result: ArenaGameResult) -> dict:
    payload = asdict(result)
    payload["status"] = result.status.value
    payload["a_outcome"] = result.a_outcome
    return payload
