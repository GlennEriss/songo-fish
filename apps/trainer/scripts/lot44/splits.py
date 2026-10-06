"""Split TRAIN/CALIBRATION/TEST par partie et verrou du TEST (sections 43-46)."""
from __future__ import annotations

from .artifacts import canonical_hash
from .config import PARTITIONS, SPLIT_FRACTIONS

SPLIT_METHOD = "deterministic_game_grouped_hash_order"


def group_split(rows: list[dict], *, seed: int, fractions: dict[str, float] = SPLIT_FRACTIONS) -> dict[str, list[dict]]:
    """Toutes les positions d'une partie vont dans la meme partition."""

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["game_id"], []).append(row)
    ordered = sorted(groups, key=lambda g: canonical_hash({"seed": seed, "split_group": g}))
    total = len(rows)
    bounds = {"train": fractions["train"] * total, "calibration": (fractions["train"] + fractions["calibration"]) * total}
    parts: dict[str, list[dict]] = {name: [] for name in PARTITIONS}
    assigned = 0
    for group in ordered:
        name = "train" if assigned < bounds["train"] else "calibration" if assigned < bounds["calibration"] else "test"
        parts[name].extend(groups[group])
        assigned += len(groups[group])
    for name in PARTITIONS:
        parts[name].sort(key=lambda r: r["fingerprint"])
    return parts


def leakage_report(parts: dict[str, list[dict]]) -> dict:
    games = {k: {r["game_id"] for r in v} for k, v in parts.items()}
    fps = {k: {r["fingerprint"] for r in v} for k, v in parts.items()}
    pairs = [(a, b) for i, a in enumerate(PARTITIONS) for b in PARTITIONS[i + 1:]]
    return {
        "same_game_across_partitions": sum(len(games[a] & games[b]) for a, b in pairs),
        "same_state_across_partitions": sum(len(fps[a] & fps[b]) for a, b in pairs),
        "within_partition_duplicates": sum(len(v) - len(fps[k]) for k, v in parts.items()),
    }


def partition_payload(name: str, rows: list[dict]) -> dict:
    fps = [r["fingerprint"] for r in rows]
    return {"partition": name, "count": len(fps), "fingerprints": fps, "sha256": canonical_hash(fps)}
