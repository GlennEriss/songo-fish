"""Référence externe historique Minimax-Bidoua, gelée pour les générations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from songo_ai.search.negamax import SearchLimits, default_evaluate, iterative_deepening
from songo_ai.songo.rules import SongoLegacyGame

from .srn_arena import ArenaDecision


@dataclass(frozen=True)
class MinimaxBidouaReferenceConfig:
    reference_id: str = "MINIMAX_BIDOUA_REFERENCE_V1"
    algorithm: str = "negamax_alpha_beta_iterative_deepening_pvs_aspiration_tt"
    max_depth: int = 14
    max_nodes: int = 300_000
    max_time_s: float = 5.0
    evaluation_fn: str = "songo_ai.search.negamax.default_evaluate"
    include_bidoua: bool = True
    safe_accumulation_threshold: int = 5
    safe_accumulation_weight: float = 3.0
    store_weight: float = 10.0
    mobility_weight: float = 1.0
    transposition_table: bool = True
    pvs: bool = True
    aspiration_delta: float = 50.0
    quiescence_depth: int = 0
    move_ordering: str = "TT move first, then immediate captures descending"

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MinimaxBidouaReferenceAgent:
    """Agent d'arène uniquement ; aucune méthode ne produit de label D_RL."""

    def __init__(self, config: MinimaxBidouaReferenceConfig | None = None) -> None:
        self.config = config or MinimaxBidouaReferenceConfig()
        self.name = self.config.reference_id

    def select_action(self, state, *, seed: int) -> ArenaDecision:
        del seed  # Recherche déterministe ; interface commune de l'arène.
        game = SongoLegacyGame.from_state(state.to_engine_state())
        result = iterative_deepening(
            game.clone_for_search(),
            SearchLimits(
                max_depth=self.config.max_depth,
                max_nodes=self.config.max_nodes,
                max_time_s=self.config.max_time_s,
                quiescence_depth=self.config.quiescence_depth,
            ),
            default_evaluate,
        )
        return ArenaDecision(
            action=result.local_action,
            search_nodes=result.nodes,
            search_time_s=result.time_s,
        )


def assert_no_minimax_labels(examples) -> None:
    forbidden = ("minimax", "teacher", "bidoua", "yinda", "negamax", "pv")
    for example in examples:
        keys = {str(key).lower() for key in example.metadata}
        matches = sorted(key for key in keys if any(token in key for token in forbidden))
        if matches:
            raise ValueError(f"benchmark/teacher metadata leaked into D_RL: {matches}")
