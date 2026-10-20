import pytest

from songo_ai.evaluation.opening_amendment import (
    amendment_sha256,
    validate_pending_amendment_approval,
    verify_amendment_bytes,
)


def pending_record(payload: bytes) -> dict:
    return {
        "AMENDMENT_SHA256": amendment_sha256(payload),
        "STATUS": "PENDING",
        "APPROVAL_DECISION": "PENDING",
        "APPROVAL_AUTHORITY": None,
        "APPROVAL_TIMESTAMP": None,
        "APPROVAL_EVIDENCE": None,
        "FINAL_OPENING_REGISTRY_CREATED": False,
    }


def test_amendment_hash_is_bound_to_exact_bytes():
    payload = b'{"status":"PROPOSED"}\n'
    record = pending_record(payload)
    validate_pending_amendment_approval(record, payload)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_pending_amendment_approval(record, payload + b" ")


def test_incorrect_hash_is_rejected():
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_amendment_bytes(b"amendment", "0" * 64)


@pytest.mark.parametrize("field,value", [
    ("APPROVAL_DECISION", "APPROVED"),
    ("APPROVAL_AUTHORITY", "invented"),
    ("FINAL_OPENING_REGISTRY_CREATED", True),
])
def test_missing_decisions_remain_pending_and_no_registry(field, value):
    payload = b"prospective amendment"
    record = pending_record(payload)
    record[field] = value
    with pytest.raises(ValueError):
        validate_pending_amendment_approval(record, payload)


def test_module_exposes_no_game_mcts_or_checkpoint_operation():
    import songo_ai.evaluation.opening_amendment as module
    assert all(not hasattr(module, name) for name in ("game", "mcts", "checkpoint", "registry_payload"))
