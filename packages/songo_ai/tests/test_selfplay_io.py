import json

import pytest

from songo_ai.dataset import (
    D_RL_FORMAT,
    D_RL_FORMAT_VERSION,
    RLTrainingExample,
    RawSongoState,
    read_d_rl_jsonl,
    write_d_rl_jsonl,
)
from songo_ai.songo.rules import PLAYER_ONE, PLAYER_TWO


def _examples():
    return (
        RLTrainingExample(
            state=RawSongoState(tuple([5] * 14 + [0, 0]), PLAYER_ONE),
            legal_mask=(True, True, False, True, True, False, True),
            visit_counts=(3, 7, 0, 55, 24, 0, 11),
            policy_target=(0.03, 0.07, 0.0, 0.55, 0.24, 0.0, 0.11),
            value_target=1.0,
            metadata={
                "game_id": "g-1",
                "ply": 7,
                "final_score": [40, 30],
                "play_policy": [0, 0, 0, 1, 0, 0, 0],
            },
        ),
        RLTrainingExample(
            state=RawSongoState(tuple([5] * 14 + [0, 0]), PLAYER_TWO),
            legal_mask=(True,) * 7,
            visit_counts=(1, 1, 1, 1, 1, 1, 1),
            policy_target=(1 / 7,) * 7,
            value_target=None,
            metadata={"game_id": "g-2", "status": "TRUNCATED_MAX_PLIES"},
        ),
    )


def test_d_rl_jsonl_round_trip_preserves_the_complete_contract(tmp_path):
    path = tmp_path / "d_rl" / "shard-00001.jsonl"
    expected = _examples()

    count = write_d_rl_jsonl(path, expected)
    loaded = read_d_rl_jsonl(path)

    assert count == 2
    assert loaded == expected
    header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert header == {
        "record_type": "manifest",
        "format": D_RL_FORMAT,
        "version": D_RL_FORMAT_VERSION,
        "dataset_family": "D_RL",
    }


def test_d_rl_reader_rejects_unknown_format_version(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        json.dumps(
            {
                "record_type": "manifest",
                "format": D_RL_FORMAT,
                "version": 999,
                "dataset_family": "D_RL",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported"):
        read_d_rl_jsonl(path)


def test_d_rl_reader_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_d_rl_jsonl(path)
