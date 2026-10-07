import inspect
import json

import pytest

from lot44.artifacts import Lot44FatalError
from lot44.config import LABEL_DEFINITION, PROTOCOL
from lot44.features import CLASSIC, COMBINED, FORBIDDEN_COLUMNS, TRAJECTORY, compute_features, feature_schema
from lot44.labels import deeper_regret, material_label, severity
from lot44.routers import conservative_route, conservative_thresholds

from lot44_test_support import drive_experiments

STATE = {"board": [5] * 14 + [0, 0], "player_to_move": 1}


def row(visits, q=None, value=0.0):
    total = sum(visits)
    best = max(range(7), key=lambda a: (visits[a], -a))
    return {"legal_mask": [True] * 7, "visit_counts": list(visits), "visit_distribution": [v / total for v in visits], "selected_action": best, "root_q_values": list(q or [0.0] * 7), "root_value": value}


def test_label_rule_cases():
    base = row([10, 50, 5, 5, 5, 5, 20], [0, 0.2, 0, 0, 0, 0, 0.1])
    assert material_label(base, row([10, 52, 5, 5, 5, 5, 18], [0, 0.2, 0, 0, 0, 0, 0.1]))["label_reason"] == "NO_TOP1_CHANGE"
    tiny_flip = row([10, 40, 5, 5, 5, 5, 41], [0, 0.20, 0, 0, 0, 0, 0.21])
    out = material_label(base, tiny_flip)
    assert out["top1_flip"] and out["label"] == 0 and out["label_reason"] == "IMMATERIAL_TOP1_CHANGE" and out["flip_severity"] == "NEAR_TIE"
    big_regret = row([1, 30, 1, 1, 1, 1, 33], [0, 0.0, 0, 0, 0, 0, 0.15])
    out = material_label(base, big_regret)
    assert out["label"] == 1 and out["high_severity"] and out["label_reason"] == "HIGH_REGRET_TOP1_CHANGE"
    distribution = row([60, 10, 5, 5, 5, 5, 10], [0.25, 0.2, 0, 0, 0, 0, 0])
    out = material_label(base, distribution)
    assert out["label"] == 1 and not out["high_severity"] and out["label_reason"] == "MATERIAL_DISTRIBUTION_TOP1_CHANGE"


def test_regret_unavailable_when_old_action_unvisited():
    shallow = row([0, 0, 0, 0, 0, 0, 1])
    reference = row([0, 9, 0, 0, 0, 0, 0], [0, 0.5, 0, 0, 0, 0, 0])
    assert deeper_regret(shallow, reference) is None
    assert severity(None) == "UNKNOWN" and severity(0.2) == "HIGH_SEVERITY"


def test_label_is_not_any_top1_change():
    assert LABEL_DEFINITION["any_top1_change_is_positive"] is False
    assert LABEL_DEFINITION["js_threshold"] == 0.05 and LABEL_DEFINITION["regret_threshold"] == 0.10


def test_label_reproduces_lot42_ultra_hard_set_exactly():
    root = drive_experiments()
    if root is None or not (root / "lot42_hard_position_65536/search_65536_results.json").is_file():
        pytest.skip("Lot41/Lot42 artifacts not available (Drive not mounted)")
    r32 = {r["state_fingerprint"]: r for r in json.loads((root / "lot41_deep_mcts_convergence/search/budget_32768.json").read_text())["rows"]}
    r65 = {r["state_fingerprint"]: r for r in json.loads((root / "lot42_hard_position_65536/search_65536_results.json").read_text())["rows"]}
    lot42 = {r["fingerprint"]: r["ultra_hard"] for r in json.loads((root / "lot42_hard_position_65536/comparison_32768_65536.json").read_text())["rows"]}
    ours = {fp: bool(material_label(r32[fp], r65[fp])["label"]) for fp in r65}
    assert ours == lot42
    assert sum(ours.values()) == 3


def test_features_have_no_reference_input_and_are_finite():
    params = list(inspect.signature(compute_features).parameters)
    assert params == ["l1", "l2", "l3", "state"]
    assert not [n for n in COMBINED if "ref" in n or "65536" in n]
    assert not set(COMBINED) & set(FORBIDDEN_COLUMNS)
    assert set(CLASSIC) | set(TRAJECTORY) <= set(COMBINED)
    l1 = row([30, 20, 10, 10, 10, 10, 10], [0.1, 0.05, 0, 0, 0, 0, 0], 0.1)
    l2 = row([20, 40, 10, 10, 10, 5, 5], [0.05, 0.12, 0, 0, 0, 0, 0], 0.11)
    l3 = row([20, 60, 5, 5, 5, 3, 2], [0.04, 0.15, 0, 0, 0, 0, 0], 0.12)
    f = compute_features(l1, l2, l3, STATE)
    assert set(f) == set(COMBINED)
    assert f["action_changes"] == 1.0 and f["action_changed_l2_l3"] == 0.0
    assert f["legal_actions"] == 7.0 and f["board_seeds_remaining"] == 70.0
    assert f["margin_l3"] == pytest.approx((60 - 20) / 100)
    assert f["q_gap_l3"] == pytest.approx(0.15 - 0.04)
    l3_bad = dict(l3, root_value=float("nan"))
    with pytest.raises(Lot44FatalError):
        compute_features(l1, l2, l3_bad, STATE)


def test_feature_schema_declares_roles_without_reference():
    schema = feature_schema({"l1": 4096, "l2": 8192, "l3": 32768, "ref": 65536})
    assert schema["roles"] == {"l1": 4096, "l2": 8192, "l3": 32768}
    assert all(not f["uses_reference_search"] for f in schema["features"])
    assert schema["compute_features_parameters"] == ["l1", "l2", "l3", "state"]


def test_conservative_router_uses_lot43_frozen_thresholds():
    thresholds = conservative_thresholds()
    assert thresholds["consistent"] and thresholds["early_stop_8192"] is False
    assert PROTOCOL["early_stop_8192_allowed"] is False
    calm = {"js_l2_l3": 0.001, "q_gap_l3": 0.2, "margin_l3": 0.9}
    assert not conservative_route(calm)
    assert conservative_route({**calm, "js_l2_l3": 0.03})
    assert conservative_route({**calm, "q_gap_l3": 0.01})
    assert conservative_route({**calm, "margin_l3": 0.1})
