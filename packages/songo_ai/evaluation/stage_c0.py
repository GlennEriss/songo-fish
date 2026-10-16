"""Lot46B Stage C0: pure preregistration helpers (never runs games or MCTS)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

B2_COMMIT = "34ba99a55cb2debbf37e9f5eb2f7038573327a15"
B3_COMMIT = "e804a41b80c6b81380b5fe26138c907308362227"
B2_SHA256 = "11e77ced813e8d71b2eb73dc2e8cb703dcc1b0dd5bba3b6c65d3c474879aea39"
B3_SHA256 = "4c03c46c99bd5d1238dac15988bd0c02f577ca54d73a90a5239718c7d56c9809"
MINIMAX_BIDOUA_SHA256 = "70f3b5632da10cfa9df4c6f28c19f8c8b8927d514b515d1c31984243d61e334c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArenaPair:
    phase: str
    candidate_a: str
    candidate_b: str
    budget_a: int
    budget_b: int
    opening_clusters: int

    def __post_init__(self) -> None:
        if self.candidate_a == self.candidate_b or min(self.budget_a, self.budget_b) <= 0:
            raise ValueError("invalid arena pair")
        if self.opening_clusters <= 0:
            raise ValueError("opening_clusters must be positive")


def preregistered_pairs() -> tuple[ArenaPair, ...]:
    candidates = ("POOL_G4R", "DEEP_POLICY_STANDARD", "DEEP_POLICY_UNCONSTRAINED")
    pairs = ((candidates[0], candidates[1]), (candidates[0], candidates[2]), (candidates[1], candidates[2]))
    c1 = tuple(ArenaPair("C1", a, b, budget, budget, 400) for budget in (512, 4096) for a, b in pairs)
    c2 = (ArenaPair("C2", "DEEP_POLICY_UNCONSTRAINED", "POOL_G4R", 512, 4096, 400),)
    return c1 + c2


def paired_seed(base_seed: int, opening_index: int, side: int) -> int:
    if opening_index < 0 or side not in (1, 2):
        raise ValueError("invalid seed coordinates")
    payload = f"lot46b-c0:{base_seed}:{opening_index}:{side}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


REQUIRED_MATCH_FIELDS = frozenset({
    "experiment_id", "source_commit", "engine_fingerprint", "candidate_checkpoint_sha256",
    "value_checkpoint_sha256", "mcts_config", "random_seed", "opening_id", "starting_side",
    "search_budget", "device", "precision", "software_versions", "terminal_result",
    "match_duration_s", "game_trace_sha256",
})


def validate_match_manifest(record: dict) -> None:
    missing = REQUIRED_MATCH_FIELDS - record.keys()
    if missing:
        raise ValueError(f"missing match manifest fields: {sorted(missing)}")
    if record["starting_side"] not in (1, 2) or int(record["search_budget"]) <= 0:
        raise ValueError("invalid side or search budget")
    for name in ("engine_fingerprint", "candidate_checkpoint_sha256", "value_checkpoint_sha256", "game_trace_sha256"):
        value = record[name]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid SHA256 field: {name}")


def verify_archive(path: Path, expected: str) -> dict:
    if not path.is_file():
        return {"status": "NOT_VERIFIED", "path": str(path), "reason": "archive unavailable in this environment"}
    actual = sha256(path)
    return {"status": "PASS" if actual == expected else "FAIL", "path": str(path), "expected_sha256": expected, "actual_sha256": actual}


def write_preregistration(path: Path) -> None:
    payload = {
        "schema": "songo-lot46b-stage-c0-v1",
        "execution_authorized": False,
        "pairs": [asdict(pair) for pair in preregistered_pairs()],
        "side_balance": "two games per opening with reversed physical sides",
        "paired_unit": "opening_id",
        "root_noise": False,
        "temperature": 0.0,
        "c_puct": 1.5,
        "primary_endpoint": "paired opening-cluster score, win=1 draw=0.5 loss=0",
        "confidence_interval": "cluster bootstrap over opening_id, 10000 resamples, 95%",
        "multiplicity": "Holm correction across the six C1 primary comparisons",
        "c2_non_inferiority_margin": -0.05,
        "c2_rule": "lower bound of paired 95% CI for score difference must exceed -0.05",
        "stopping": "fixed 400 opening clusters per comparison; no optional stopping",
        "invalid_games": "reported separately; never converted to draws",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
