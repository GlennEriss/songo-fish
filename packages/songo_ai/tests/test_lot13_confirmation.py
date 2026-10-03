from __future__ import annotations

import hashlib

from songo_ai.evaluation import (
    checkpoint_identity,
    confirmation_verdict,
    generate_unique_deterministic_openings,
)


def _summary(*, score=0.7, low=0.6, p1=(7, 1, 2), p2=(6, 2, 2)):
    def side(values):
        wins, draws, losses = values
        return {
            "games": wins + draws + losses,
            "terminal_games": wins + draws + losses,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "truncated_repetition": 0,
            "truncated_max_plies": 0,
        }
    return {
        "score_rate_a_terminal": score,
        "paired_bootstrap_ci": [low, 0.8],
        "by_a_side": {"P1": side(p1), "P2": side(p2)},
    }


def test_checkpoint_identity_uses_full_sha256(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checkpoint_identity(checkpoint, expected_sha256=expected)["matches"]
    assert not checkpoint_identity(checkpoint, expected_sha256="0" * 64)["matches"]


def test_confirmation_requires_ci_both_sides_and_no_anomaly():
    assert confirmation_verdict(_summary(), anomalies=[])["CONFIRMED_G2"]
    assert not confirmation_verdict(_summary(low=0.49), anomalies=[])["CONFIRMED_G2"]
    assert not confirmation_verdict(
        _summary(p2=(4, 0, 6)), anomalies=[]
    )["CONFIRMED_G2"]
    assert not confirmation_verdict(_summary(), anomalies=["bad pairing"])["CONFIRMED_G2"]


def test_unique_openings_are_reproducible_and_distinct():
    first = generate_unique_deterministic_openings(count=128, seed=20261313)
    second = generate_unique_deterministic_openings(count=128, seed=20261313)
    assert first == second
    states = {(opening.state.board, opening.state.player_to_move) for opening in first}
    assert len(states) == 128
