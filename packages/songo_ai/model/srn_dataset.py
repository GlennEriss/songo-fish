"""Dataset et batching PyTorch dedies aux exemples ``D_RL`` du SRN."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import torch
from torch.utils.data import Dataset

from .srn_graph import SongoGraphBatch, SongoGraphBuilder

if TYPE_CHECKING:
    from songo_ai.dataset.selfplay_schema import RLTrainingExample


class DRLDataset(Dataset):
    """Dataset conserve sous forme d'etats bruts, jamais de features figees."""

    def __init__(self, source: str | Path | Sequence["RLTrainingExample"]) -> None:
        if isinstance(source, (str, Path)):
            from songo_ai.dataset.selfplay_io import read_d_rl_jsonl

            examples = read_d_rl_jsonl(source)
            self.source_path = Path(source)
        else:
            examples = tuple(source)
            self.source_path = None
        if not examples:
            raise ValueError("D_RL dataset must contain at least one example")
        for index, example in enumerate(examples):
            if example.policy_target is None:
                raise ValueError(f"example {index} has no policy_target")
            if example.visit_counts is None:
                raise ValueError(f"example {index} has no visit_counts")
            game_id = example.metadata.get("game_id")
            if not isinstance(game_id, str) or not game_id:
                raise ValueError(f"example {index} has no valid metadata.game_id")
        self.examples = tuple(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> "RLTrainingExample":
        return self.examples[index]

    @property
    def game_ids(self) -> tuple[str, ...]:
        return tuple(sorted({str(example.metadata["game_id"]) for example in self.examples}))


@dataclass(frozen=True)
class DRLSplit:
    train_examples: tuple["RLTrainingExample", ...]
    validation_examples: tuple["RLTrainingExample", ...]
    train_game_ids: tuple[str, ...]
    validation_game_ids: tuple[str, ...]


def split_examples_by_game_id(
    examples: Sequence["RLTrainingExample"],
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> DRLSplit:
    """Split deterministe sans jamais separer les positions d'une partie."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    groups: dict[str, list["RLTrainingExample"]] = {}
    for example in examples:
        game_id = example.metadata.get("game_id")
        if not isinstance(game_id, str) or not game_id:
            raise ValueError("every D_RL example must contain metadata.game_id")
        groups.setdefault(game_id, []).append(example)
    if len(groups) < 2:
        raise ValueError("at least two game_ids are required for a train/validation split")

    game_ids = sorted(groups)
    random.Random(seed).shuffle(game_ids)
    validation_count = round(len(game_ids) * validation_fraction)
    validation_count = min(len(game_ids) - 1, max(1, validation_count))
    validation_ids = tuple(sorted(game_ids[:validation_count]))
    train_ids = tuple(sorted(game_ids[validation_count:]))
    validation_set = set(validation_ids)
    train_examples = tuple(
        example for example in examples if example.metadata["game_id"] not in validation_set
    )
    validation_examples = tuple(
        example for example in examples if example.metadata["game_id"] in validation_set
    )
    return DRLSplit(
        train_examples=train_examples,
        validation_examples=validation_examples,
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
    )


def split_examples_by_fixed_game_ids(
    examples: Sequence["RLTrainingExample"],
    *,
    train_game_ids: Sequence[str],
    validation_game_ids: Sequence[str],
) -> DRLSplit:
    """Reconstruit exactement un split déjà publié, sans nouveau tirage."""

    train_ids = tuple(sorted(str(value) for value in train_game_ids))
    validation_ids = tuple(sorted(str(value) for value in validation_game_ids))
    train_set, validation_set = set(train_ids), set(validation_ids)
    if not train_set or not validation_set or train_set & validation_set:
        raise ValueError("fixed train/validation game ids must be non-empty and disjoint")
    observed = {str(example.metadata.get("game_id")) for example in examples}
    if train_set | validation_set != observed:
        missing = observed - (train_set | validation_set)
        extra = (train_set | validation_set) - observed
        raise ValueError(f"fixed split does not cover dataset exactly; missing={missing}, extra={extra}")
    return DRLSplit(
        train_examples=tuple(e for e in examples if e.metadata["game_id"] in train_set),
        validation_examples=tuple(e for e in examples if e.metadata["game_id"] in validation_set),
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
    )


@dataclass(frozen=True)
class SRNTrainingBatch:
    graph: SongoGraphBatch
    legal_mask: torch.Tensor  # [B, 7]
    policy_target: torch.Tensor  # [B, 7]
    value_target: torch.Tensor  # [B], placeholder masque si cible absente
    value_mask: torch.Tensor  # [B]
    game_ids: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return self.graph.batch_size

    @property
    def node_features(self) -> torch.Tensor:
        return self.graph.node_features

    @property
    def global_features(self) -> torch.Tensor:
        return self.graph.global_features

    def to(self, device: torch.device | str) -> "SRNTrainingBatch":
        return SRNTrainingBatch(
            graph=self.graph.to(device),
            legal_mask=self.legal_mask.to(device),
            policy_target=self.policy_target.to(device),
            value_target=self.value_target.to(device),
            value_mask=self.value_mask.to(device),
            game_ids=self.game_ids,
        )


class SRNBatchCollator:
    def __init__(self, graph_builder: SongoGraphBuilder | None = None) -> None:
        self.graph_builder = graph_builder or SongoGraphBuilder()

    def __call__(self, examples: Iterable["RLTrainingExample"]) -> SRNTrainingBatch:
        examples = tuple(examples)
        if not examples:
            raise ValueError("cannot collate an empty D_RL batch")
        if any(example.policy_target is None for example in examples):
            raise ValueError("all batch examples must have a policy_target")
        graph = self.graph_builder.build_batch([example.state for example in examples])
        value_mask = torch.tensor(
            [example.value_target is not None for example in examples], dtype=torch.bool
        )
        # Le zero est uniquement un placeholder tensoriel. value_mask garantit
        # qu'il ne contribue jamais a la Value loss.
        value_target = torch.tensor(
            [example.value_target if example.value_target is not None else 0.0 for example in examples],
            dtype=torch.float32,
        )
        return SRNTrainingBatch(
            graph=graph,
            legal_mask=torch.tensor([example.legal_mask for example in examples], dtype=torch.bool),
            policy_target=torch.tensor(
                [example.policy_target for example in examples], dtype=torch.float32
            ),
            value_target=value_target,
            value_mask=value_mask,
            game_ids=tuple(str(example.metadata["game_id"]) for example in examples),
        )
