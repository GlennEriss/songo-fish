from __future__ import annotations

import pytest

from songo_ai.dataset import RLTrainingExample, RawSongoState
from songo_ai.evaluation import (
    MinimaxBidouaReferenceConfig,
    assert_no_minimax_labels,
)


def _example(metadata):
    return RLTrainingExample(
        state=RawSongoState((5,) * 14 + (0, 0), 1),
        legal_mask=(True,) * 7,
        visit_counts=(1, 1, 1, 1, 1, 1, 1),
        policy_target=(1 / 7,) * 7,
        value_target=1.0,
        metadata=metadata,
    )


def test_reference_fingerprint_is_stable_and_configuration_sensitive():
    first = MinimaxBidouaReferenceConfig()
    second = MinimaxBidouaReferenceConfig()
    weaker = MinimaxBidouaReferenceConfig(max_depth=13)
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != weaker.fingerprint
    assert first.reference_id == "MINIMAX_BIDOUA_REFERENCE_V1"


def test_drl_contract_rejects_minimax_or_teacher_metadata():
    assert_no_minimax_labels([_example({"generation_lineage": "G2_to_G3"})])
    with pytest.raises(ValueError, match="leaked"):
        assert_no_minimax_labels([_example({"minimax_score": 12})])
    with pytest.raises(ValueError, match="leaked"):
        assert_no_minimax_labels([_example({"teacher": "bidoua"})])
