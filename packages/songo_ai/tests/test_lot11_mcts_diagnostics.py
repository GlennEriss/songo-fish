from __future__ import annotations

import math

import torch

from songo_ai.dataset import RawSongoState
from songo_ai.evaluation import (
    ScaledValueEvaluator,
    choose_smallest_mcts_budget,
    final_g2_readiness_gate,
    interseed_stability_report,
    model_parameter_fingerprint,
)
from songo_ai.model import SRNConfig, SongoRelationalNetwork
from songo_ai.search import MCTSConfig, SongoMCTS


def _model() -> SongoRelationalNetwork:
    torch.manual_seed(17)
    return SongoRelationalNetwork(
        SRNConfig(hidden_dim=16, num_relational_blocks=1)
    )


def _initial_state() -> RawSongoState:
    return RawSongoState((5,) * 14 + (0, 0), 1)


def test_scaled_value_preserves_policy_and_scales_value() -> None:
    model = _model()
    graph = SongoMCTS(model).graph_builder.build(_initial_state()).as_batch()
    with torch.no_grad():
        expected_policy, expected_value = model(graph)
        policy, value = ScaledValueEvaluator(model, 0.5)(graph)
    assert torch.equal(policy, expected_policy)
    assert torch.allclose(value, expected_value * 0.5)


def test_mcts_trace_is_reproducible_complete_and_read_only() -> None:
    model = _model()
    before = model_parameter_fingerprint(model)
    config = MCTSConfig(
        num_simulations=8,
        add_root_noise=True,
        collect_simulation_trace=True,
        seed=91,
    )
    first = SongoMCTS(model, config=config).search(_initial_state())
    second = SongoMCTS(model, config=config).search(_initial_state())
    assert first.simulation_trace == second.simulation_trace
    assert len(first.simulation_trace) == 8
    assert first.simulation_trace[-1]["visit_counts_after_backup"] == list(
        first.visit_counts
    )
    assert all(row["root_action"] is not None for row in first.simulation_trace)
    assert model_parameter_fingerprint(model) == before


def test_interseed_report_excludes_forced_positions() -> None:
    report = interseed_stability_report(
        [
            [[0.5, 0.5, 0, 0, 0, 0, 0], [0.25, 0.75, 0, 0, 0, 0, 0]],
            [[1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]],
        ],
        [2, 1],
    )
    assert report["states"] == 1
    assert report["runs"] == 2
    assert report["support"]["mean"] == 2.0
    assert math.isfinite(report["pairwise_js"]["mean"])


def _curve(js=(0.50, 0.25, 0.24, 0.23)) -> dict:
    result = {}
    for budget, value in zip((8, 16, 32, 64), js):
        result[budget] = {
            "pairwise_js": {"mean": value},
            "one_hot_fraction": 0.05,
            "support": {"mean": 3.5},
            "entropy": {"mean": 1.0},
            "pairwise_argmax_agreement": {"mean": 0.60 if budget == 8 else 0.70},
        }
    return result


def test_budget_selection_uses_smallest_quality_plateau() -> None:
    selection = choose_smallest_mcts_budget(_curve())
    assert selection["selected_budget"] == 16
    assert not selection["selected_by_fallback_best_js"]


def test_final_gate_requires_every_quality_condition() -> None:
    baseline = {
        "policy": {"one_hot_fraction": 0.4, "support_mean": 2.0, "entropy_mean": 0.5},
        "repeated": {"initial_state": {"pairwise_jensen_shannon_mean": 0.4}},
    }
    pilot = {
        "policy": {"one_hot_fraction": 0.2, "support_mean": 2.6, "entropy_mean": 0.7},
        "repeated": {"initial_state": {"pairwise_jensen_shannon_mean": 0.2}},
    }
    assert final_g2_readiness_gate(baseline, pilot, valid=True)["ready_for_g2"]
    assert not final_g2_readiness_gate(baseline, pilot, valid=False)["ready_for_g2"]
