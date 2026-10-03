import json

import torch
from torch import nn

from songo_ai.evaluation import (
    HybridPolicyValueEvaluator,
    discover_real_dataset_files,
    iter_real_records,
    overlap_report,
    position_digest,
    position_key,
    record_integrity,
)


class Model(nn.Module):
    def __init__(self, policy, value):
        super().__init__(); self.register_buffer("policy", torch.tensor(policy, dtype=torch.float32)); self.value = value

    def forward(self, graph):
        return self.policy.repeat(graph.batch_size, 1), torch.full((graph.batch_size,), self.value)


class Graph:
    batch_size = 2


def record():
    return {
        "schema_version": "songo-real-move-v1", "board_before": [5] * 14 + [0, 0],
        "player_position": 1, "legal_mask": [True] * 7, "action_local": 0,
        "match_id": "m1", "ply": 0,
    }


def test_hybrid_uses_exact_policy_and_value_without_parameters():
    policy = Model(range(7), 0.25); value = Model(range(7, 14), -0.75)
    hybrid = HybridPolicyValueEvaluator(policy, value)
    logits, result = hybrid(Graph())
    assert torch.equal(logits, policy(Graph())[0]); assert torch.equal(result, value(Graph())[1])
    assert list(hybrid.parameters()) == list(policy.parameters()) + list(value.parameters())


def test_real_parser_identity_integrity_and_overlap(tmp_path):
    path = tmp_path / "real_matches" / "moves.jsonl"; path.parent.mkdir()
    path.write_text(json.dumps(record()) + "\n")
    assert discover_real_dataset_files(tmp_path) == [path]
    parsed = list(iter_real_records(path))[0]
    assert position_key(parsed) == (tuple([5] * 14 + [0, 0]), 1)
    assert position_digest(parsed) == position_digest(record())
    assert record_integrity(parsed) == []
    report = overlap_report({"a": {position_key(parsed)}, "b": {position_key(parsed)}})
    assert report["a"]["b"] == {"intersection": 1, "percent_of_left": 100.0, "percent_of_right": 100.0}


def test_discovery_excludes_d_rl(tmp_path):
    path = tmp_path / "d_rl" / "fake.jsonl"; path.parent.mkdir()
    path.write_text(json.dumps(record()) + "\n")
    assert discover_real_dataset_files(tmp_path) == []
