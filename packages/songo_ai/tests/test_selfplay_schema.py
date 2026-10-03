import pytest

from songo_ai.dataset import D_LAB, D_RL, DatasetFamily, RLTrainingExample, RawSongoState
from songo_ai.songo.rules import PLAYER_ONE, PLAYER_TWO, SongoLegacyGame, State


def test_dataset_families_are_explicit_and_distinct():
    assert D_LAB is DatasetFamily.LAB
    assert D_RL is DatasetFamily.RL
    assert D_LAB != D_RL


def test_raw_state_round_trip_keeps_physical_board_and_player():
    board = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5)
    raw = RawSongoState.from_engine_state(State(board, PLAYER_TWO))

    assert raw.board == board
    assert raw.player_to_move == PLAYER_TWO
    assert raw.to_engine_state() == State(board, PLAYER_TWO)


def test_raw_state_can_be_created_from_existing_game_without_canonicalization():
    game = SongoLegacyGame.from_board([5] * 14 + [0, 0], turn=PLAYER_TWO)
    raw = RawSongoState.from_game(game)

    assert raw.board == tuple([5] * 14 + [0, 0])
    assert raw.player_to_move == PLAYER_TWO


def test_rl_example_accepts_pending_search_and_terminal_labels():
    raw = RawSongoState(tuple([5] * 14 + [0, 0]), PLAYER_ONE)
    example = RLTrainingExample(
        state=raw,
        legal_mask=(True,) * 7,
        metadata={"game_id": "future-self-play-game"},
    )

    assert example.dataset_family is D_RL
    assert example.policy_target is None
    assert example.value_target is None
    assert example.visit_counts is None
    assert not example.is_search_complete
    assert not example.is_terminally_labeled


def test_rl_example_accepts_complete_s_m_pi_z_contract():
    raw = RawSongoState(tuple([5] * 14 + [0, 0]), PLAYER_ONE)
    example = RLTrainingExample(
        state=raw,
        legal_mask=(True, True, False, True, False, True, True),
        visit_counts=(3, 7, 0, 55, 0, 24, 11),
        policy_target=(0.03, 0.07, 0.0, 0.55, 0.0, 0.24, 0.11),
        value_target=1.0,
        metadata={"generation": 0},
    )

    assert example.is_search_complete
    assert example.is_terminally_labeled
    assert example.metadata == {"generation": 0}


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"legal_mask": (True,) * 6}, "legal_mask"),
        (
            {
                "legal_mask": (True, True, False, True, True, True, True),
                "policy_target": (0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1),
            },
            "illegal",
        ),
        (
            {
                "legal_mask": (True,) * 7,
                "policy_target": (0.1,) * 7,
            },
            "sum to 1",
        ),
        (
            {
                "legal_mask": (True, False, True, True, True, True, True),
                "visit_counts": (1, 1, 1, 1, 1, 1, 1),
            },
            "illegal",
        ),
        ({"legal_mask": (True,) * 7, "value_target": 1.1}, "[-1, 1]"),
    ],
)
def test_rl_example_rejects_contract_violations(kwargs, message):
    raw = RawSongoState(tuple([5] * 14 + [0, 0]), PLAYER_ONE)
    with pytest.raises(ValueError, match=message):
        RLTrainingExample(state=raw, **kwargs)


def test_raw_state_rejects_invalid_engine_invariants():
    with pytest.raises(ValueError, match="16 counters"):
        RawSongoState((5,) * 14, PLAYER_ONE)
    with pytest.raises(ValueError, match="conserve 70"):
        RawSongoState((0,) * 16, PLAYER_ONE)
    with pytest.raises(ValueError, match="PLAYER_ONE"):
        RawSongoState(tuple([5] * 14 + [0, 0]), 3)

