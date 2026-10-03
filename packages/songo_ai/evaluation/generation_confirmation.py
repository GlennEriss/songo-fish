"""Contrats minimaux de confirmation d'une génération SRN."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping


def checkpoint_identity(path: str | Path, *, expected_sha256: str) -> dict:
    path = Path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "matches": actual == expected_sha256,
    }


def confirmation_verdict(mcts64_summary: Mapping, *, anomalies: list[str]) -> dict:
    """Applique le critère Lot 13 sans optimiser un seuil sur les résultats."""

    score = mcts64_summary["score_rate_a_terminal"]
    interval = mcts64_summary["paired_bootstrap_ci"]
    sides = mcts64_summary["by_a_side"]

    def side_score(side: Mapping) -> float | None:
        terminal = int(side["terminal_games"])
        if not terminal:
            return None
        return (int(side["wins"]) + 0.5 * int(side["draws"])) / terminal

    p1_score = side_score(sides["P1"])
    p2_score = side_score(sides["P2"])
    checks = {
        "score_above_half": score is not None and score > 0.5,
        "paired_ci_entirely_above_half": interval is not None and interval[0] > 0.5,
        "positive_from_both_sides": (
            p1_score is not None and p1_score > 0.5
            and p2_score is not None and p2_score > 0.5
        ),
        "no_major_anomaly": not anomalies,
    }
    confirmed = all(checks.values())
    return {
        "CONFIRMED_G2": confirmed,
        "G2_GENERATOR_AUTHORIZED": confirmed,
        "checks": checks,
        "side_score_terminal": {"P1": p1_score, "P2": p2_score},
        "anomalies": list(anomalies),
    }
