"""Label MATERIAL_LATE_BIFURCATION_32768_65536 (gele avant entrainement)."""
from __future__ import annotations

import math

from run_srn_lot43 import jensen_shannon, normalize

from .config import LABEL_DEFINITION


def severity(regret: float | None) -> str:
    if regret is None:
        return "UNKNOWN"
    bins = LABEL_DEFINITION["severity_bins"]
    if regret <= bins["NEAR_TIE"]:
        return "NEAR_TIE"
    if regret <= bins["LOW_SEVERITY"]:
        return "LOW_SEVERITY"
    if regret <= bins["MEDIUM_SEVERITY"]:
        return "MEDIUM_SEVERITY"
    return "HIGH_SEVERITY"


def deeper_regret(shallow: dict, reference: dict) -> float | None:
    """Regret profond identique a Lot42 : Q_ref[a_ref] - Q_ref[a_shallow]."""

    new, old = reference["selected_action"], shallow["selected_action"]
    visits, q = reference["visit_counts"], reference["root_q_values"]
    if new is None or old is None or visits[new] <= 0 or visits[old] <= 0:
        return None
    value = float(q[new]) - float(q[old])
    return value if math.isfinite(value) else None


def material_label(shallow: dict, reference: dict) -> dict:
    flip = shallow["selected_action"] != reference["selected_action"]
    js = jensen_shannon(shallow.get("visit_distribution") or normalize(shallow), reference.get("visit_distribution") or normalize(reference))
    regret = deeper_regret(shallow, reference)
    regret_fires = regret is not None and regret > LABEL_DEFINITION["regret_threshold"]
    positive = flip and (js > LABEL_DEFINITION["js_threshold"] or regret_fires)
    if not flip:
        reason = "NO_TOP1_CHANGE"
    elif regret_fires:
        reason = "HIGH_REGRET_TOP1_CHANGE"
    elif positive:
        reason = "MATERIAL_DISTRIBUTION_TOP1_CHANGE"
    else:
        reason = "IMMATERIAL_TOP1_CHANGE"
    return {
        "label": int(positive),
        "label_reason": reason,
        "top1_flip": flip,
        "js_l3_ref": js,
        "deeper_regret": regret,
        "flip_severity": severity(regret) if flip else "NONE",
        "high_severity": bool(positive and regret_fires),
        "action_l3": shallow["selected_action"],
        "action_ref": reference["selected_action"],
    }
