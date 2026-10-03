"""MCTS/PUCT minimal pour le SRN Songo.

Le moteur reste l'unique autorite sur les transitions, la legalite et les
resultats terminaux. Le reseau fournit uniquement les priors de politique et
une Value exprimee du point de vue du joueur au trait dans l'etat evalue.

Convention centrale de perspective
----------------------------------
``Q(S, a)`` et ``W(S, a)`` sont toujours stockes du point de vue du joueur au
trait dans le noeud parent ``S``. Une valeur de feuille est convertie vers ce
joueur avec :func:`convert_value_perspective`, en comparant les identites des
joueurs. Le code ne suppose donc pas qu'une transition alterne toujours le
trait, meme si c'est le comportement observe pour les transitions non
terminales du moteur actuel.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn

from songo_ai.songo.rules import (
    DRAW,
    NUM_ACTIONS,
    PLAYER_ONE,
    PLAYER_TWO,
    SongoLegacyGame,
)

if TYPE_CHECKING:
    from songo_ai.dataset.selfplay_schema import RawSongoState
    from songo_ai.model.srn_graph import SongoGraphBuilder


@dataclass(frozen=True)
class MCTSConfig:
    """Hyperparametres experimentaux de la recherche PUCT."""

    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    add_root_noise: bool = False
    collect_simulation_trace: bool = False
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if not math.isfinite(self.c_puct) or self.c_puct < 0.0:
            raise ValueError("c_puct must be finite and non-negative")
        if not math.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0.0:
            raise ValueError("dirichlet_alpha must be finite and positive")
        if not math.isfinite(self.dirichlet_epsilon) or not 0.0 <= self.dirichlet_epsilon <= 1.0:
            raise ValueError("dirichlet_epsilon must be in [0, 1]")


@dataclass
class MCTSNode:
    """Noeud de recherche et statistiques des sept actions locales."""

    state: "RawSongoState"
    legal_mask: tuple[bool, ...]
    terminal: bool = False
    winner: Optional[int] = None
    expanded: bool = False
    priors: list[float] = field(default_factory=lambda: [0.0] * NUM_ACTIONS)
    visit_counts: list[int] = field(default_factory=lambda: [0] * NUM_ACTIONS)
    value_sums: list[float] = field(default_factory=lambda: [0.0] * NUM_ACTIONS)
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)
    network_value: Optional[float] = None
    search_depth: int = 0

    def __post_init__(self) -> None:
        if len(self.legal_mask) != NUM_ACTIONS:
            raise ValueError(f"legal_mask must contain {NUM_ACTIONS} entries")
        self.legal_mask = tuple(bool(value) for value in self.legal_mask)
        for name, values in (
            ("priors", self.priors),
            ("visit_counts", self.visit_counts),
            ("value_sums", self.value_sums),
        ):
            if len(values) != NUM_ACTIONS:
                raise ValueError(f"{name} must contain {NUM_ACTIONS} entries")

    @property
    def player_to_move(self) -> int:
        return self.state.player_to_move

    @property
    def total_visits(self) -> int:
        return sum(self.visit_counts)

    @property
    def q_values(self) -> tuple[float, ...]:
        """Moyenne ``W/N`` ; une action non visitee a explicitement Q=0."""

        return tuple(
            self.value_sums[action] / count if count else 0.0
            for action, count in enumerate(self.visit_counts)
        )


@dataclass(frozen=True)
class MCTSResult:
    """Resultat exploitable par le futur pipeline de self-play."""

    visit_counts: tuple[int, ...]
    policy: tuple[float, ...]
    root_value: float
    selected_action: Optional[int]
    num_simulations: int
    num_nodes: int
    nodes_expanded: int
    network_evaluations: int
    elapsed_s: float
    simulations_per_second: float
    root_priors: tuple[float, ...]
    root_q_values: tuple[float, ...]
    legal_mask: tuple[bool, ...]
    simulation_trace: tuple[dict, ...] = ()


def convert_value_perspective(value: float, from_player: int, to_player: int) -> float:
    """Convertit une valeur zero-sum entre les perspectives physiques P1/P2."""

    if from_player not in (PLAYER_ONE, PLAYER_TWO):
        raise ValueError("from_player must be PLAYER_ONE or PLAYER_TWO")
    if to_player not in (PLAYER_ONE, PLAYER_TWO):
        raise ValueError("to_player must be PLAYER_ONE or PLAYER_TWO")
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return float(value if from_player == to_player else -value)


def terminal_value(winner: int, perspective_player: int) -> float:
    """Convertit le vainqueur moteur en valeur depuis ``perspective_player``."""

    if perspective_player not in (PLAYER_ONE, PLAYER_TWO):
        raise ValueError("perspective_player must be PLAYER_ONE or PLAYER_TWO")
    if winner == DRAW:
        return 0.0
    if winner not in (PLAYER_ONE, PLAYER_TWO):
        raise ValueError("a terminal state must have winner P1, P2 or DRAW")
    return 1.0 if winner == perspective_player else -1.0


def puct_scores(node: MCTSNode, c_puct: float) -> tuple[float, ...]:
    """Calcule ``Q + c*P*sqrt(1+N_parent)/(1+N_a)`` pour chaque action."""

    if not math.isfinite(c_puct) or c_puct < 0.0:
        raise ValueError("c_puct must be finite and non-negative")
    parent_scale = math.sqrt(1.0 + node.total_visits)
    q_values = node.q_values
    scores = []
    for action in range(NUM_ACTIONS):
        if not node.legal_mask[action]:
            scores.append(float("-inf"))
            continue
        exploration = c_puct * node.priors[action] * parent_scale / (1 + node.visit_counts[action])
        scores.append(q_values[action] + exploration)
    return tuple(scores)


def select_puct_action(node: MCTSNode, c_puct: float, rng: random.Random) -> int:
    """Selectionne une action legale ; les egalites exactes utilisent ``rng``."""

    legal_actions = [action for action, legal in enumerate(node.legal_mask) if legal]
    if not legal_actions:
        raise ValueError("cannot select an action from a node without legal moves")
    scores = puct_scores(node, c_puct)
    best_score = max(scores[action] for action in legal_actions)
    tied_actions = [action for action in legal_actions if scores[action] == best_score]
    return rng.choice(tied_actions)


def visit_counts_to_policy(
    visit_counts: Sequence[int],
    legal_mask: Sequence[bool],
    temperature: float = 1.0,
) -> tuple[float, ...]:
    """Transforme les visites brutes en cible ``pi`` sans modifier les comptes.

    ``temperature=0`` produit un argmax deterministe ; en cas d'egalite,
    l'action locale de plus petit indice est choisie. Si tous les comptes sont
    nuls et ``temperature>0``, la distribution est uniforme sur les coups
    legaux.
    """

    if len(visit_counts) != NUM_ACTIONS or len(legal_mask) != NUM_ACTIONS:
        raise ValueError(f"visit_counts and legal_mask must contain {NUM_ACTIONS} entries")
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    counts = tuple(int(count) for count in visit_counts)
    mask = tuple(bool(legal) for legal in legal_mask)
    if any(count < 0 for count in counts):
        raise ValueError("visit counts must be non-negative")
    if any(count != 0 for count, legal in zip(counts, mask) if not legal):
        raise ValueError("illegal actions must have zero visit counts")
    legal_actions = [action for action, legal in enumerate(mask) if legal]
    if not legal_actions:
        return (0.0,) * NUM_ACTIONS

    policy = [0.0] * NUM_ACTIONS
    if temperature == 0.0:
        best_count = max(counts[action] for action in legal_actions)
        best_action = next(action for action in legal_actions if counts[action] == best_count)
        policy[best_action] = 1.0
        return tuple(policy)

    positive_actions = [action for action in legal_actions if counts[action] > 0]
    if not positive_actions:
        uniform = 1.0 / len(legal_actions)
        for action in legal_actions:
            policy[action] = uniform
        return tuple(policy)

    log_weights = {
        action: math.log(counts[action]) / temperature
        for action in positive_actions
    }
    maximum = max(log_weights.values())
    weights = {action: math.exp(value - maximum) for action, value in log_weights.items()}
    denominator = sum(weights.values())
    for action, weight in weights.items():
        policy[action] = weight / denominator
    return tuple(policy)


class SongoMCTS:
    """Recherche MCTS/PUCT sequentielle, sans self-play ni cache complexe."""

    def __init__(
        self,
        model: nn.Module,
        graph_builder: Optional["SongoGraphBuilder"] = None,
        config: Optional[MCTSConfig] = None,
    ) -> None:
        if graph_builder is None:
            # Import differe : le depot historique fait transiter
            # dataset -> generation -> search pendant certains imports.
            from songo_ai.model.srn_graph import SongoGraphBuilder

            graph_builder = SongoGraphBuilder()
        self.model = model
        self.graph_builder = graph_builder
        self.config = config or MCTSConfig()

    def search(self, state: "RawSongoState", *, policy_temperature: float = 1.0) -> MCTSResult:
        """Execute la recherche depuis un etat moteur brut non canonicalise.

        ``root_value`` est la moyenne des retours sauvegardes aux aretes de la
        racine, donc une estimation MCTS dans la perspective du joueur au trait
        a la racine. Ce n'est pas la Value brute du SRN lors de l'expansion.
        """

        root = self._root_node(state)
        started_at = time.perf_counter()
        if root.terminal:
            elapsed = time.perf_counter() - started_at
            return MCTSResult(
                visit_counts=tuple(root.visit_counts),
                policy=(0.0,) * NUM_ACTIONS,
                root_value=terminal_value(root.winner, root.player_to_move),
                selected_action=None,
                num_simulations=0,
                num_nodes=1,
                nodes_expanded=0,
                network_evaluations=0,
                elapsed_s=elapsed,
                simulations_per_second=0.0,
                root_priors=tuple(root.priors),
                root_q_values=root.q_values,
                legal_mask=root.legal_mask,
            )

        rng = random.Random(self.config.seed)
        noise_rng = np.random.default_rng(self.config.seed)
        num_nodes = 1
        nodes_expanded = 0
        network_evaluations = 0
        completed = 0
        simulation_trace: list[dict] = []
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                self._expand(root)
                nodes_expanded += 1
                network_evaluations += 1
                if self.config.add_root_noise:
                    self._add_root_noise(root, noise_rng)

                for simulation_index in range(self.config.num_simulations):
                    node = root
                    path: list[tuple[MCTSNode, int]] = []

                    while node.expanded and not node.terminal:
                        action = select_puct_action(node, self.config.c_puct, rng)
                        path.append((node, action))
                        child = node.children.get(action)
                        if child is None:
                            child = self._transition(node, action)
                            node.children[action] = child
                            num_nodes += 1
                        node = child

                    if node.terminal:
                        leaf_value = terminal_value(node.winner, node.player_to_move)
                    else:
                        leaf_value = self._expand(node)
                        nodes_expanded += 1
                        network_evaluations += 1

                    self._backup(path, leaf_value, node.player_to_move)
                    if self.config.collect_simulation_trace:
                        root_action = path[0][1] if path else None
                        simulation_trace.append(
                            {
                                "simulation": simulation_index + 1,
                                "root_action": root_action,
                                "root_prior": (
                                    root.priors[root_action]
                                    if root_action is not None
                                    else None
                                ),
                                "root_action_visits": (
                                    root.visit_counts[root_action]
                                    if root_action is not None
                                    else None
                                ),
                                "root_action_q": (
                                    root.q_values[root_action]
                                    if root_action is not None
                                    else None
                                ),
                                "leaf_value": float(leaf_value),
                                "leaf_player": node.player_to_move,
                                "leaf_terminal": node.terminal,
                                "leaf_depth": len(path),
                                "leaf_state": {
                                    "board": list(node.state.board),
                                    "player_to_move": node.state.player_to_move,
                                },
                                "visit_counts_after_backup": list(root.visit_counts),
                                "root_priors_all": list(root.priors),
                                "root_q_values_after_backup": list(root.q_values),
                                "root_puct_scores_after_backup": list(
                                    puct_scores(root, self.config.c_puct)
                                ),
                            }
                        )
                    completed += 1
        finally:
            self.model.train(was_training)

        elapsed = time.perf_counter() - started_at
        policy = visit_counts_to_policy(root.visit_counts, root.legal_mask, policy_temperature)
        root_value = sum(root.value_sums) / root.total_visits
        max_visits = max(root.visit_counts[action] for action in range(NUM_ACTIONS) if root.legal_mask[action])
        selected_action = next(
            action
            for action in range(NUM_ACTIONS)
            if root.legal_mask[action] and root.visit_counts[action] == max_visits
        )
        return MCTSResult(
            visit_counts=tuple(root.visit_counts),
            policy=policy,
            root_value=float(root_value),
            selected_action=selected_action,
            num_simulations=completed,
            num_nodes=num_nodes,
            nodes_expanded=nodes_expanded,
            network_evaluations=network_evaluations,
            elapsed_s=elapsed,
            simulations_per_second=completed / elapsed if elapsed > 0.0 else float("inf"),
            root_priors=tuple(root.priors),
            root_q_values=root.q_values,
            legal_mask=root.legal_mask,
            simulation_trace=tuple(simulation_trace),
        )

    def search_many(
        self,
        states: Sequence["RawSongoState"],
        *,
        policy_temperature: float = 1.0,
        seeds: Optional[Sequence[Optional[int]]] = None,
    ) -> tuple[MCTSResult, ...]:
        """Execute plusieurs recherches independantes avec inference groupee.

        Une simulation est avancee par racine et par tour. Les feuilles non
        terminales obtenues sont ensuite rassemblees dans un unique graphe de
        batch. Les arbres, statistiques PUCT et sauvegardes restent strictement
        independants : seul le passage reseau est mutualise.

        Cette forme est particulierement adaptee aux arenes et au self-play ou
        plusieurs parties sont disponibles simultanement. Elle ne modifie ni
        le nombre de simulations par position, ni l'autorite du moteur sur les
        transitions et les terminaux.
        """

        if not states:
            return ()
        if seeds is None:
            seeds = tuple(
                None if self.config.seed is None else self.config.seed + index
                for index in range(len(states))
            )
        if len(seeds) != len(states):
            raise ValueError("seeds must contain one entry per state")

        roots = [self._root_node(state) for state in states]
        rngs = [random.Random(seed) for seed in seeds]
        noise_rngs = [np.random.default_rng(seed) for seed in seeds]
        node_counts = [1] * len(roots)
        expanded_counts = [0] * len(roots)
        evaluation_counts = [0] * len(roots)
        completed_counts = [0] * len(roots)
        traces: list[list[dict]] = [[] for _ in roots]
        started_at = time.perf_counter()
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                initial = [root for root in roots if not root.terminal]
                self._expand_many(initial)
                for index, root in enumerate(roots):
                    if not root.terminal:
                        expanded_counts[index] += 1
                        evaluation_counts[index] += 1
                        if self.config.add_root_noise:
                            self._add_root_noise(root, noise_rngs[index])

                for simulation_index in range(self.config.num_simulations):
                    pending: list[tuple[int, MCTSNode, list[tuple[MCTSNode, int]]]] = []
                    terminal_leaves: list[
                        tuple[int, MCTSNode, list[tuple[MCTSNode, int]], float]
                    ] = []
                    for index, root in enumerate(roots):
                        if root.terminal:
                            continue
                        node = root
                        path: list[tuple[MCTSNode, int]] = []
                        while node.expanded and not node.terminal:
                            action = select_puct_action(node, self.config.c_puct, rngs[index])
                            path.append((node, action))
                            child = node.children.get(action)
                            if child is None:
                                child = self._transition(node, action)
                                node.children[action] = child
                                node_counts[index] += 1
                            node = child
                        if node.terminal:
                            terminal_leaves.append(
                                (index, node, path, terminal_value(node.winner, node.player_to_move))
                            )
                        else:
                            pending.append((index, node, path))

                    values = self._expand_many([node for _, node, _ in pending])
                    leaves = terminal_leaves + [
                        (index, node, path, value)
                        for (index, node, path), value in zip(pending, values)
                    ]
                    for index, node, path, leaf_value in leaves:
                        if not node.terminal:
                            expanded_counts[index] += 1
                            evaluation_counts[index] += 1
                        self._backup(path, leaf_value, node.player_to_move)
                        if self.config.collect_simulation_trace:
                            traces[index].append(
                                self._trace_entry(
                                    roots[index], node, path, leaf_value, simulation_index
                                )
                            )
                        completed_counts[index] += 1
        finally:
            self.model.train(was_training)

        elapsed = time.perf_counter() - started_at
        results = []
        for index, root in enumerate(roots):
            if root.terminal:
                results.append(
                    MCTSResult(
                        visit_counts=tuple(root.visit_counts),
                        policy=(0.0,) * NUM_ACTIONS,
                        root_value=terminal_value(root.winner, root.player_to_move),
                        selected_action=None,
                        num_simulations=0,
                        num_nodes=1,
                        nodes_expanded=0,
                        network_evaluations=0,
                        elapsed_s=elapsed,
                        simulations_per_second=0.0,
                        root_priors=tuple(root.priors),
                        root_q_values=root.q_values,
                        legal_mask=root.legal_mask,
                    )
                )
                continue
            policy = visit_counts_to_policy(
                root.visit_counts, root.legal_mask, policy_temperature
            )
            root_value = sum(root.value_sums) / root.total_visits
            max_visits = max(
                root.visit_counts[action]
                for action in range(NUM_ACTIONS)
                if root.legal_mask[action]
            )
            selected_action = next(
                action
                for action in range(NUM_ACTIONS)
                if root.legal_mask[action] and root.visit_counts[action] == max_visits
            )
            results.append(
                MCTSResult(
                    visit_counts=tuple(root.visit_counts),
                    policy=policy,
                    root_value=float(root_value),
                    selected_action=selected_action,
                    num_simulations=completed_counts[index],
                    num_nodes=node_counts[index],
                    nodes_expanded=expanded_counts[index],
                    network_evaluations=evaluation_counts[index],
                    elapsed_s=elapsed,
                    simulations_per_second=(
                        completed_counts[index] / elapsed if elapsed > 0.0 else float("inf")
                    ),
                    root_priors=tuple(root.priors),
                    root_q_values=root.q_values,
                    legal_mask=root.legal_mask,
                    simulation_trace=tuple(traces[index]),
                )
            )
        return tuple(results)

    @staticmethod
    def _root_node(state: "RawSongoState") -> MCTSNode:
        game = SongoLegacyGame.from_state(state.to_engine_state())
        game.normalize_terminal()
        return SongoMCTS._node_from_game(game)

    @staticmethod
    def _node_from_game(game: SongoLegacyGame, search_depth: int = 0) -> MCTSNode:
        from songo_ai.dataset.selfplay_schema import RawSongoState

        legal_mask = (False,) * NUM_ACTIONS if game.finished else game.legal_mask()
        return MCTSNode(
            state=RawSongoState.from_game(game),
            legal_mask=legal_mask,
            terminal=game.finished,
            winner=game.winner,
            search_depth=search_depth,
        )

    @staticmethod
    def _transition(parent: MCTSNode, local_action: int) -> MCTSNode:
        if parent.terminal:
            raise ValueError("cannot transition from a terminal node")
        if not 0 <= local_action < NUM_ACTIONS or not parent.legal_mask[local_action]:
            raise ValueError(f"illegal local action for MCTS transition: {local_action}")
        game = SongoLegacyGame.from_state(parent.state.to_engine_state())
        game.play_local(local_action)
        return SongoMCTS._node_from_game(game, parent.search_depth + 1)

    def _expand(self, node: MCTSNode) -> float:
        if node.terminal:
            raise ValueError("terminal nodes must not be expanded by the network")
        if node.expanded:
            raise ValueError("node is already expanded")
        device = self._model_device()
        graph = self.graph_builder.build(node.state).as_batch().to(device)
        if hasattr(self.model, "forward_at_depth"):
            policy_logits, value = self.model.forward_at_depth(graph, node.search_depth)
        else:
            policy_logits, value = self.model(graph)
        legal_mask = torch.tensor([node.legal_mask], dtype=torch.bool, device=device)
        from songo_ai.model.srn_network import policy_probabilities

        priors = policy_probabilities(policy_logits, legal_mask)[0]
        scalar_value = float(value.reshape(-1)[0].item())
        if not math.isfinite(scalar_value):
            raise ValueError("network returned a non-finite value")
        if priors.shape != (NUM_ACTIONS,) or not torch.isfinite(priors).all():
            raise ValueError("network returned invalid policy priors")
        node.priors = [float(probability) for probability in priors.detach().cpu().tolist()]
        node.network_value = scalar_value
        node.expanded = True
        return scalar_value

    def _expand_many(self, nodes: Sequence[MCTSNode]) -> tuple[float, ...]:
        """Etend des feuilles dans un ou plusieurs forwards groupes."""

        if not nodes:
            return ()
        if any(node.terminal for node in nodes):
            raise ValueError("terminal nodes must not be expanded by the network")
        if any(node.expanded for node in nodes):
            raise ValueError("node is already expanded")
        device = self._model_device()
        from songo_ai.model.srn_network import policy_probabilities

        logits_by_index: list[Optional[torch.Tensor]] = [None] * len(nodes)
        values_by_index: list[Optional[torch.Tensor]] = [None] * len(nodes)
        if hasattr(self.model, "forward_at_depth"):
            depth_groups: dict[int, list[int]] = {}
            for index, node in enumerate(nodes):
                depth_groups.setdefault(node.search_depth, []).append(index)
            for depth, indices in depth_groups.items():
                graph = self.graph_builder.build_batch([nodes[i].state for i in indices]).to(device)
                logits, values = self.model.forward_at_depth(graph, depth)
                for local, original in enumerate(indices):
                    logits_by_index[original] = logits[local]
                    values_by_index[original] = values.reshape(-1)[local]
        else:
            graph = self.graph_builder.build_batch([node.state for node in nodes]).to(device)
            logits, values = self.model(graph)
            flat_values = values.reshape(-1)
            for index in range(len(nodes)):
                logits_by_index[index] = logits[index]
                values_by_index[index] = flat_values[index]

        policy_logits = torch.stack([item for item in logits_by_index if item is not None])
        legal_mask = torch.tensor(
            [node.legal_mask for node in nodes], dtype=torch.bool, device=device
        )
        priors_batch = policy_probabilities(policy_logits, legal_mask)
        scalar_values = []
        for index, node in enumerate(nodes):
            value_tensor = values_by_index[index]
            if value_tensor is None:
                raise RuntimeError("missing batched value output")
            scalar_value = float(value_tensor.item())
            priors = priors_batch[index]
            if not math.isfinite(scalar_value):
                raise ValueError("network returned a non-finite value")
            if priors.shape != (NUM_ACTIONS,) or not torch.isfinite(priors).all():
                raise ValueError("network returned invalid policy priors")
            node.priors = [float(x) for x in priors.detach().cpu().tolist()]
            node.network_value = scalar_value
            node.expanded = True
            scalar_values.append(scalar_value)
        return tuple(scalar_values)

    def _trace_entry(
        self,
        root: MCTSNode,
        node: MCTSNode,
        path: Sequence[tuple[MCTSNode, int]],
        leaf_value: float,
        simulation_index: int,
    ) -> dict:
        root_action = path[0][1] if path else None
        return {
            "simulation": simulation_index + 1,
            "root_action": root_action,
            "root_prior": root.priors[root_action] if root_action is not None else None,
            "root_action_visits": (
                root.visit_counts[root_action] if root_action is not None else None
            ),
            "root_action_q": root.q_values[root_action] if root_action is not None else None,
            "leaf_value": float(leaf_value),
            "leaf_player": node.player_to_move,
            "leaf_terminal": node.terminal,
            "leaf_depth": len(path),
            "leaf_state": {
                "board": list(node.state.board),
                "player_to_move": node.state.player_to_move,
            },
            "visit_counts_after_backup": list(root.visit_counts),
            "root_priors_all": list(root.priors),
            "root_q_values_after_backup": list(root.q_values),
            "root_puct_scores_after_backup": list(puct_scores(root, self.config.c_puct)),
        }

    def _model_device(self) -> torch.device:
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(self.model.buffers(), None)
        return buffer.device if buffer is not None else torch.device("cpu")

    def _add_root_noise(self, root: MCTSNode, rng: np.random.Generator) -> None:
        legal_actions = [action for action, legal in enumerate(root.legal_mask) if legal]
        noise = rng.dirichlet([self.config.dirichlet_alpha] * len(legal_actions))
        epsilon = self.config.dirichlet_epsilon
        for index, action in enumerate(legal_actions):
            root.priors[action] = (1.0 - epsilon) * root.priors[action] + epsilon * float(noise[index])

    @staticmethod
    def _backup(
        path: Sequence[tuple[MCTSNode, int]],
        leaf_value: float,
        leaf_player: int,
    ) -> None:
        for parent, action in reversed(path):
            parent_value = convert_value_perspective(
                leaf_value,
                from_player=leaf_player,
                to_player=parent.player_to_move,
            )
            parent.visit_counts[action] += 1
            parent.value_sums[action] += parent_value
