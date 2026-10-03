import pytest

from songo_ai.evaluation import (
    action_regrets, correlations, legal_ranking, ranking_category,
    stratified_sample_indices, top_margin,
)
from songo_ai.search import convert_value_perspective


def test_ranking_metrics_and_categories_respect_legality():
    mask = [True, True, False, True, False, False, False]
    g2 = [.6, .3, 0, .1, 0, 0, 0]; g3 = [.2, .7, 0, .1, 0, 0, 0]
    target = [.2, .75, 0, .05, 0, 0, 0]
    assert legal_ranking(g2, mask) == (0, 1, 3)
    assert top_margin(target, mask) == pytest.approx(.55)
    category = ranking_category(g2, g3, target, mask)
    assert category["primary"] == "B_change_to_mcts_top1"
    assert category["separation"] == "F_strongly_separated"


def test_regret_and_perspective_are_parent_relative():
    regrets = action_regrets([.2, -.4, 0, .1, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0])
    assert regrets == pytest.approx((0.0, .6, None, .1, None, None, None))
    assert convert_value_perspective(.4, 2, 1) == pytest.approx(-.4)
    assert convert_value_perspective(.4, 1, 1) == pytest.approx(.4)


def test_stratified_sampling_is_reproducible_and_spans_strata():
    rows = [
        {"ply": i % 120, "legal_count": 1 + i % 7, "g2_top1": 0, "g3a_top1": i % 2, "g3b_top1": 0}
        for i in range(100)
    ]
    first = stratified_sample_indices(rows, 50, seed=17)
    assert first == stratified_sample_indices(rows, 50, seed=17)
    assert len(first) == len(set(first)) == 50


def test_correlations_reports_pearson_and_spearman():
    report = correlations({"a": [1, 2, 3], "b": [3, 2, 1]})
    assert report["a"]["b"]["pearson"] == pytest.approx(-1)
    assert report["a"]["b"]["spearman"] == pytest.approx(-1)
