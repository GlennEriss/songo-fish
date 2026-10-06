"""Routeur conservateur de reference (Lot43 CONSERVATIVE_MULTI_SIGNAL, 32768 -> 65536).

Seuils importes de Lot43 sans modification : ils ont ete fixes sur les 256
positions d'origine, donc leur application au corpus Lot44 est hors echantillon.
L'arret a 8192 est interdit (EARLY_STOP_8192_ALLOWED = NO).
"""
from __future__ import annotations

from run_srn_lot43 import ULTRA_JS_TRIGGER, ULTRA_Q_GAP_TRIGGER

from .config import PROTOCOL

# Litteral de run_srn_lot43.decide_router (``d32.get("visit_margin", 1.0) < 0.15``).
LOT43_MARGIN_TRIGGER = 0.15


def conservative_route(features: dict[str, float]) -> bool:
    return (
        features["js_l2_l3"] > ULTRA_JS_TRIGGER
        or features["q_gap_l3"] < ULTRA_Q_GAP_TRIGGER
        or features["margin_l3"] < LOT43_MARGIN_TRIGGER
    )


def conservative_thresholds() -> dict:
    observed = {"js_trigger": ULTRA_JS_TRIGGER, "q_gap_trigger": ULTRA_Q_GAP_TRIGGER, "margin_trigger": LOT43_MARGIN_TRIGGER}
    expected = {"js_trigger": PROTOCOL["conservative_js_trigger"], "q_gap_trigger": PROTOCOL["conservative_q_gap_trigger"], "margin_trigger": PROTOCOL["conservative_margin_trigger"]}
    return {"observed_from_lot43": observed, "protocol": expected, "consistent": observed == expected, "early_stop_8192": False}
