"""Static integrity helpers for a prospective opening-protocol amendment.

This module cannot approve an amendment or construct an opening registry.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping


def amendment_sha256(amendment_bytes: bytes) -> str:
    if not amendment_bytes:
        raise ValueError("amendment bytes must not be empty")
    return hashlib.sha256(amendment_bytes).hexdigest()


def verify_amendment_bytes(amendment_bytes: bytes, expected_sha256: str) -> str:
    actual = amendment_sha256(amendment_bytes)
    if len(expected_sha256) != 64 or actual != expected_sha256:
        raise ValueError("amendment SHA256 mismatch")
    return actual


def validate_pending_amendment_approval(record: Mapping[str, Any], amendment_bytes: bytes) -> None:
    """Require an unsigned, hash-bound record and reject inferred approval."""
    verify_amendment_bytes(amendment_bytes, str(record.get("AMENDMENT_SHA256", "")))
    if record.get("STATUS") != "PENDING" or record.get("APPROVAL_DECISION") != "PENDING":
        raise ValueError("approval record must remain pending")
    if any(record.get(field) is not None for field in
           ("APPROVAL_AUTHORITY", "APPROVAL_TIMESTAMP", "APPROVAL_EVIDENCE")):
        raise ValueError("pending approval cannot carry approval metadata")
    if record.get("FINAL_OPENING_REGISTRY_CREATED") is not False:
        raise ValueError("pending approval cannot construct a final registry")
