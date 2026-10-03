from __future__ import annotations

import torch

from songo_ai.dataset import RLTrainingExample, RawSongoState
from songo_ai.evaluation import analyze_policy_mcts_signal
from songo_ai.model import SRNConfig, SongoRelationalNetwork


def _examples():
    rows = []
    for index, ply in enumerate((0, 45, 120)):
        rows.append(RLTrainingExample(
            state=RawSongoState((5,) * 14 + (0, 0), 1 if index % 2 == 0 else 2),
            legal_mask=(True,) * 7,
            visit_counts=(32, 16, 8, 4, 2, 1, 1),
            policy_target=(0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.015625),
            value_target=1.0,
            metadata={"game_id": f"g{index}", "ply": ply},
        ))
    return rows


def test_policy_mcts_signal_is_reproducible_and_stratified():
    torch.manual_seed(15)
    model = SongoRelationalNetwork(SRNConfig(hidden_dim=8, num_relational_blocks=1))
    first = analyze_policy_mcts_signal(model, _examples(), batch_size=2)
    second = analyze_policy_mcts_signal(model, _examples(), batch_size=3)
    assert first == second
    assert first["examples"] == 3
    assert set(first["subgroups"]["phase"]) == {"ply_0_30", "ply_31_90", "ply_91_plus"}
    assert first["global"]["js"]["minimum"] >= 0.0
