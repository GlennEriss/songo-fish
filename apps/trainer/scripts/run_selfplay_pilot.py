#!/usr/bin/env python3
"""Petite experience locale de validation du pipeline self-play D_RL."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from songo_ai.dataset import write_d_rl_jsonl
from songo_ai.generation import SelfPlayConfig, SelfPlayRunner
from songo_ai.model import SRNConfig, SongoRelationalNetwork
from songo_ai.search import MCTSConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=2)
    parser.add_argument("--max-game-plies", type=int, default=300)
    parser.add_argument("--repetition-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    model = SongoRelationalNetwork(
        SRNConfig(hidden_dim=args.hidden_dim, num_relational_blocks=args.blocks)
    )
    config = SelfPlayConfig(
        games=args.games,
        max_game_plies=args.max_game_plies,
        repetition_limit=args.repetition_limit,
        seed=args.seed,
        checkpoint_id="random-srn-pilot",
        mcts=MCTSConfig(
            num_simulations=args.simulations,
            add_root_noise=True,
            seed=args.seed,
        ),
    )
    result = SelfPlayRunner(model, config).generate()
    payload = asdict(result.statistics)
    payload["status_counts"] = {
        status: sum(game.status.value == status for game in result.games)
        for status in sorted({game.status.value for game in result.games})
    }
    if args.output is not None:
        payload["examples_written"] = write_d_rl_jsonl(args.output, result.examples)
        payload["output"] = str(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
