"""Règles pré-enregistrées de confirmation et promotion du Lot 29."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def side_score(side: Mapping[str, int]) -> float | None:
    games = int(side.get("games", int(side["wins"]) + int(side["draws"]) + int(side["losses"])))
    return (int(side["wins"]) + .5 * int(side["draws"])) / games if games else None


def side_gap(summary: Mapping) -> float:
    scores = [side_score(summary["by_a_side"][side]) for side in ("P1", "P2")]
    if any(x is None for x in scores):
        raise ValueError("both sides need terminal games")
    return abs(scores[0] - scores[1])


def independent_seed_set(candidate: Sequence[int], forbidden: Sequence[int]) -> bool:
    values = tuple(int(x) for x in candidate)
    return len(values) == len(set(values)) and not (set(values) & set(map(int, forbidden)))


def promotion_rule(
    score64: float,
    score128: float,
    *,
    side_gap64: float,
    side_gap128: float,
    score256: float | None = None,
    max_scaling_drop: float = .05,
    max_side_gap: float = .15,
    valid: bool = True,
) -> dict[str, object]:
    values=(score64,score128,side_gap64,side_gap128)
    if any(not math.isfinite(float(x)) for x in values):raise ValueError("metrics must be finite")
    combined=(score64+score128)/2
    delta64_128=score128-score64
    delta128_256=None if score256 is None else score256-score128
    scaling=delta64_128>=-max_scaling_drop and (delta128_256 is None or delta128_256>=-max_scaling_drop)
    sides=max(side_gap64,side_gap128)<=max_side_gap
    promoted=valid and score64>.5 and score128>.5 and combined>.5 and scaling and sides
    return {"promoted":promoted,"combined_score":combined,"delta_64_128":delta64_128,"delta_128_256":delta128_256,"scaling_healthy":scaling,"side_balance_acceptable":sides}
