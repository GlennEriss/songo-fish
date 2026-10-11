import pytest

from songo_ai.training.lot46 import Lot46Error
from songo_ai.training.lot46b_b0 import support_change_counts, value_baselines


def test_support_change_rates_use_changed_denominator_and_keep_counts():
    report = support_change_counts([.2, .051, .05, 0, -.05, -.051, -.4])
    assert report["N_changed"] == 7
    assert report["N_potential_correction"] == 2
    assert report["N_potential_regression"] == 2
    assert report["N_uncertain"] == 3
    assert report["rates_on_changed"]["potential_correction"] == pytest.approx(2/7)
    assert report["rates_on_changed"]["uncertain"] == pytest.approx(3/7)


def test_support_change_rejects_posthoc_invalid_thresholds_and_nonfinite_values():
    with pytest.raises(Lot46Error): support_change_counts([0.1], 0)
    with pytest.raises(Lot46Error): support_change_counts([float("nan")])


def test_value_baselines_are_paired_and_use_terminal_labels_only():
    result = value_baselines([-1, 0, 1, 1], [-.8, 0, .7, .9])
    assert result["model"]["sign_accuracy"] == 1
    assert result["constant_mean"]["prediction"] == pytest.approx(.25)
    assert sum(result["result_frequencies"].values()) == pytest.approx(1)
    with pytest.raises(Lot46Error): value_baselines([.5], [0])


def test_b0_module_has_no_training_or_arena_entrypoint():
    import songo_ai.training.lot46b_b0 as module
    assert not any(name in vars(module) for name in ("run_training", "pilot_run", "arena", "promote"))
