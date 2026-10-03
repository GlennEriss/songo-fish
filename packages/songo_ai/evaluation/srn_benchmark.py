"""Batterie D_LAB et comparaison descriptive des sorties brutes du SRN."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from songo_ai.dataset.selfplay_schema import RawSongoState
from songo_ai.model.srn_graph import SongoGraphBuilder
from songo_ai.model.srn_network import policy_probabilities
from songo_ai.songo.rules import PLAYER_ONE, PLAYER_TWO, SongoLegacyGame


@dataclass(frozen=True)
class LabBenchmarkPosition:
    position_id: str
    state: RawSongoState
    legal_mask: tuple[bool, ...]
    source_trajectory_id: str
    move_number: int
    phase_proxy: str

    @property
    def legal_count(self) -> int:
        return sum(self.legal_mask)

    @property
    def seeds_in_play(self) -> int:
        return sum(self.state.board[:14])


def _phase_proxy(move_number: int) -> str:
    if move_number <= 15:
        return "opening"
    if move_number <= 60:
        return "midgame"
    return "late"


def _physical_state_from_canonical(
    canonical_board: Sequence[int], player_to_move: int
) -> tuple[int, ...]:
    board = tuple(int(value) for value in canonical_board)
    if player_to_move == PLAYER_ONE:
        return board
    if player_to_move != PLAYER_TWO:
        raise ValueError("player_to_move must be PLAYER_ONE or PLAYER_TWO")
    # canonicalize_board est une involution : echange camps et magasins.
    return board[7:14] + board[0:7] + (board[15], board[14])


def select_d_lab_benchmark(
    path: str | Path,
    *,
    seed: int,
) -> tuple[LabBenchmarkPosition, ...]:
    """Selectionne une position par strate non vide ``phase x legal_count``.

    La source v001 est canonicalisee du point de vue du joueur au trait. Pour
    exposer les deux identites physiques au SRN brut, les strates triees sont
    alternativement materialisees comme P1 puis comme P2. Le choix interne a
    chaque strate minimise un hash de ``seed + identite source`` et ne lit
    aucune sortie de modele ni aucun label teacher.
    """

    path = Path(path)
    candidates: dict[tuple[str, int], list[dict]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            phase = _phase_proxy(int(record["move_number"]))
            legal_count = sum(bool(value) for value in record["legal_mask"])
            candidates.setdefault((phase, legal_count), []).append(record)
    if not candidates:
        raise ValueError("D_LAB source is empty")

    phase_order = {"opening": 0, "midgame": 1, "late": 2}
    selected: list[LabBenchmarkPosition] = []
    for index, key in enumerate(
        sorted(candidates, key=lambda value: (phase_order[value[0]], value[1]))
    ):
        phase, legal_count = key

        def rank(record: dict) -> str:
            identity = (
                f"{seed}:{record['trajectory_id']}:{record['move_number']}:"
                + ",".join(str(value) for value in record["state"])
            )
            return hashlib.sha256(identity.encode("utf-8")).hexdigest()

        record = min(candidates[key], key=rank)
        player = PLAYER_ONE if index % 2 == 0 else PLAYER_TWO
        board = _physical_state_from_canonical(record["state"], player)
        state = RawSongoState(board, player)
        game = SongoLegacyGame.from_state(state.to_engine_state())
        game.normalize_terminal()
        legal_mask = tuple(bool(value) for value in record["legal_mask"])
        if game.finished or tuple(game.legal_mask()) != legal_mask:
            raise ValueError("D_LAB canonical state cannot be reconstructed consistently")
        source_id = f"{record['trajectory_id']}:{record['move_number']}"
        selected.append(
            LabBenchmarkPosition(
                position_id=f"lab-{phase}-l{legal_count}-{source_id}-p{player}",
                state=state,
                legal_mask=legal_mask,
                source_trajectory_id=str(record["trajectory_id"]),
                move_number=int(record["move_number"]),
                phase_proxy=phase,
            )
        )
    return tuple(selected)


def select_extended_d_lab_benchmark(
    path: str | Path,
    *,
    seed: int,
    target_count: int = 350,
) -> tuple[LabBenchmarkPosition, ...]:
    """Echantillonnage equilibre par bins structurels, sans labels/modeles.

    Un round-robin deterministe parcourt les bins
    ``phase x legal_count x seeds_in_play x stores``. Les enregistrements de
    chaque bin sont tries par SHA-256 seede. Cette strategie donne du poids aux
    situations rares sans selectionner selon Minimax, Policy ou Value.
    """

    if target_count <= 0:
        raise ValueError("target_count must be positive")
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if target_count > len(records):
        raise ValueError("target_count exceeds the available D_LAB records")

    def bucket(value: int, cuts: tuple[int, ...]) -> int:
        return next((index for index, cut in enumerate(cuts) if value <= cut), len(cuts))

    bins: dict[tuple, list[dict]] = {}
    for record in records:
        seeds_in_play = sum(int(value) for value in record["state"][:14])
        stores_total = int(record["state"][14]) + int(record["state"][15])
        key = (
            _phase_proxy(int(record["move_number"])),
            sum(bool(value) for value in record["legal_mask"]),
            bucket(seeds_in_play, (14, 35, 55)),
            bucket(stores_total, (14, 35, 55)),
        )
        bins.setdefault(key, []).append(record)

    def rank(record: dict) -> str:
        identity = (
            f"{seed}:{record['trajectory_id']}:{record['move_number']}:"
            + ",".join(str(value) for value in record["state"])
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    queues = {key: sorted(values, key=rank) for key, values in bins.items()}
    ordered_keys = sorted(queues, key=repr)
    chosen: list[dict] = []
    while len(chosen) < target_count:
        progressed = False
        for key in ordered_keys:
            if queues[key]:
                chosen.append(queues[key].pop(0))
                progressed = True
                if len(chosen) == target_count:
                    break
        if not progressed:
            raise RuntimeError("structural round-robin exhausted unexpectedly")

    positions = []
    for index, record in enumerate(chosen):
        player = PLAYER_ONE if index % 2 == 0 else PLAYER_TWO
        board = _physical_state_from_canonical(record["state"], player)
        state = RawSongoState(board, player)
        legal_mask = tuple(bool(value) for value in record["legal_mask"])
        game = SongoLegacyGame.from_state(state.to_engine_state())
        game.normalize_terminal()
        if game.finished or tuple(game.legal_mask()) != legal_mask:
            raise ValueError("D_LAB canonical state cannot be reconstructed consistently")
        phase = _phase_proxy(int(record["move_number"]))
        source_id = f"{record['trajectory_id']}:{record['move_number']}"
        positions.append(
            LabBenchmarkPosition(
                position_id=(
                    f"labx-{index:03d}-{phase}-l{sum(legal_mask)}-{source_id}-p{player}"
                ),
                state=state,
                legal_mask=legal_mask,
                source_trajectory_id=str(record["trajectory_id"]),
                move_number=int(record["move_number"]),
                phase_proxy=phase,
            )
        )
    return tuple(positions)


def model_parameter_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0.0)


def _jensen_shannon(first: Sequence[float], second: Sequence[float]) -> float:
    midpoint = [(left + right) / 2.0 for left, right in zip(first, second)]

    def kl(values: Sequence[float]) -> float:
        return sum(
            value * math.log(value / middle)
            for value, middle in zip(values, midpoint)
            if value > 0.0
        )

    return 0.5 * kl(first) + 0.5 * kl(second)


def _kl_divergence(first: Sequence[float], second: Sequence[float]) -> float:
    return sum(
        left * math.log(left / right)
        for left, right in zip(first, second)
        if left > 0.0 and right > 0.0
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def evaluate_raw_network_outputs(
    models: Mapping[str, torch.nn.Module],
    positions: Sequence[LabBenchmarkPosition],
    *,
    graph_builder: SongoGraphBuilder | None = None,
) -> dict:
    if not models:
        raise ValueError("models must not be empty")
    if not positions:
        raise ValueError("positions must not be empty")
    graph_builder = graph_builder or SongoGraphBuilder()
    graph_cpu = graph_builder.build_batch(position.state for position in positions)
    legal_cpu = torch.tensor(
        [position.legal_mask for position in positions], dtype=torch.bool
    )
    outputs: dict[str, dict] = {}
    for name, model in models.items():
        parameter = next(model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                logits, values = model(graph_cpu.to(device))
                policies = policy_probabilities(logits, legal_cpu.to(device))
        finally:
            model.train(was_training)
        policy_rows = policies.detach().cpu().tolist()
        value_rows = values.detach().cpu().tolist()
        entropies = [_entropy(row) for row in policy_rows]
        margins = []
        for row, position in zip(policy_rows, positions):
            legal_probabilities = sorted(
                (row[action] for action, legal in enumerate(position.legal_mask) if legal),
                reverse=True,
            )
            margins.append(
                legal_probabilities[0] - legal_probabilities[1]
                if len(legal_probabilities) > 1
                else legal_probabilities[0]
            )
        outputs[name] = {
            "parameter_fingerprint": model_parameter_fingerprint(model),
            "policy_entropy": {
                "mean": statistics.fmean(entropies),
                "std": statistics.pstdev(entropies),
                "minimum": min(entropies),
                "maximum": max(entropies),
            },
            "value": {
                "mean": statistics.fmean(value_rows),
                "std": statistics.pstdev(value_rows),
                "minimum": min(value_rows),
                "maximum": max(value_rows),
            },
            "policy_top1_margin": {
                "mean": statistics.fmean(margins),
                "median": statistics.median(margins),
                "q05": _quantile(margins, 0.05),
                "q25": _quantile(margins, 0.25),
                "q75": _quantile(margins, 0.75),
                "q95": _quantile(margins, 0.95),
            },
            "positions": [
                {
                    "position_id": position.position_id,
                    "policy": policy,
                    "policy_entropy": entropy,
                    "policy_argmax": max(range(len(policy)), key=policy.__getitem__),
                    "policy_top1_margin": margin,
                    "value": float(value),
                }
                for position, policy, entropy, margin, value in zip(
                    positions, policy_rows, entropies, margins, value_rows
                )
            ],
        }

    pairwise = {}
    names = list(models)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left_positions = outputs[left_name]["positions"]
            right_positions = outputs[right_name]["positions"]
            js_values = [
                _jensen_shannon(left["policy"], right["policy"])
                for left, right in zip(left_positions, right_positions)
            ]
            kl_left_right = [
                _kl_divergence(left["policy"], right["policy"])
                for left, right in zip(left_positions, right_positions)
            ]
            kl_right_left = [
                _kl_divergence(right["policy"], left["policy"])
                for left, right in zip(left_positions, right_positions)
            ]
            absolute_values = [
                abs(left["value"] - right["value"])
                for left, right in zip(left_positions, right_positions)
            ]
            sign_changes = [
                (left["value"] > 0.0) != (right["value"] > 0.0)
                for left, right in zip(left_positions, right_positions)
            ]
            disagreements = [
                left["policy_argmax"] != right["policy_argmax"]
                for left, right in zip(left_positions, right_positions)
            ]
            left_argmax_probability_changes = [
                abs(
                    left["policy"][left["policy_argmax"]]
                    - right["policy"][left["policy_argmax"]]
                )
                for left, right in zip(left_positions, right_positions)
            ]
            pairwise[f"{left_name}__vs__{right_name}"] = {
                "policy_jensen_shannon_mean": statistics.fmean(js_values),
                "policy_jensen_shannon_max": max(js_values),
                "policy_kl_left_to_right_mean": statistics.fmean(kl_left_right),
                "policy_kl_right_to_left_mean": statistics.fmean(kl_right_left),
                "policy_argmax_disagreements": sum(disagreements),
                "policy_argmax_disagreement_rate": sum(disagreements) / len(disagreements),
                "left_argmax_probability_absolute_change_mean": statistics.fmean(
                    left_argmax_probability_changes
                ),
                "value_absolute_change_mean": statistics.fmean(absolute_values),
                "value_absolute_change_max": max(absolute_values),
                "value_sign_changes": sum(sign_changes),
                "value_sign_change_rate": sum(sign_changes) / len(sign_changes),
                "per_position": [
                    {
                        "position_id": position.position_id,
                        "jensen_shannon": js,
                        "argmax_disagreement": disagreement,
                        "left_argmax": left["policy_argmax"],
                        "right_argmax": right["policy_argmax"],
                        "left_argmax_probability_absolute_change": probability_change,
                    }
                    for position, left, right, js, disagreement, probability_change in zip(
                        positions,
                        left_positions,
                        right_positions,
                        js_values,
                        disagreements,
                        left_argmax_probability_changes,
                    )
                ],
            }

    position_records = [
        {
            "position_id": position.position_id,
            "state": {
                "board": list(position.state.board),
                "player_to_move": position.state.player_to_move,
            },
            "legal_mask": list(position.legal_mask),
            "legal_actions": [
                action for action, legal in enumerate(position.legal_mask) if legal
            ],
            "legal_count": position.legal_count,
            "seeds_in_play": position.seeds_in_play,
            "stores": list(position.state.board[14:16]),
            "source_trajectory_id": position.source_trajectory_id,
            "move_number": position.move_number,
            "phase_proxy": position.phase_proxy,
        }
        for position in positions
    ]
    concrete = []
    for index in range(min(5, len(positions))):
        concrete.append(
            {
                **position_records[index],
                "models": {
                    name: outputs[name]["positions"][index] for name in models
                },
            }
        )
    return {
        "positions": position_records,
        "selection": {
            "rule": "one minimum seeded SHA-256 record per non-empty phase_proxy x legal_count stratum",
            "phase_proxy": {"opening": "move_number <= 15", "midgame": "16..60", "late": "> 60"},
            "physical_player_assignment": "alternating P1/P2 after deterministic stratum sort",
            "uses_model_outputs": False,
            "uses_teacher_labels": False,
        },
        "models": outputs,
        "pairwise": pairwise,
        "concrete_subset": concrete,
    }
