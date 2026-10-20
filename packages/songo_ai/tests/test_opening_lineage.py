import pytest

from songo_ai.evaluation.opening_lineage import *
from songo_ai.evaluation.opening_registry import physical_state_fingerprint, raw_state_from_record, sha256


def state(player=1, delta=0):
    board=[5]*14+[0,0]
    if delta:
        board[0]-=delta; board[14]+=delta
    return {"board":board,"player_to_move":player}


def test_exact_join_complete_recovery_and_overlap():
    sources=[{"fingerprint":"a","state":state()},{"fingerprint":"b","state":state(delta=1)}]
    mappings=join_historical_identifiers([{"position_id":"a"},{"position_id":"b"}],sources,
        record_field="position_id",source_field="fingerprint",record_namespace="lot45",source_namespace="lot45")
    assert lineage_summary(mappings)["physical_coverage_percent"]==100.0
    fp=physical_state_fingerprint(raw_state_from_record(sources[0]))
    assert exact_overlap(mappings,{fp})==[fp]


def test_namespace_mismatch_rejected():
    with pytest.raises(ValueError,match="namespace"):
        join_historical_identifiers([],[],record_field="id",source_field="id",record_namespace="b3",source_namespace="other")


def test_ambiguous_one_to_many_join():
    result=join_historical_identifiers([{"id":"x"}],[{"id":"x","state":state()},{"id":"x","state":state(delta=1)}],
        record_field="id",source_field="id",record_namespace="n",source_namespace="n")
    assert result[0].confidence==AMBIGUOUS and result[0].state_fingerprint is None


def test_missing_and_no_speculative_reconstruction():
    result=join_historical_identifiers([{"id":"missing"},{"id":"metadata"}],[{"id":"metadata","logits":[1,2]}],
        record_field="id",source_field="id",record_namespace="n",source_namespace="n")
    assert [x.confidence for x in result]==[UNRESOLVED,UNRESOLVED]
    assert all(x.state_fingerprint is None for x in result)


def test_duplicate_occurrence_ids_rejected():
    with pytest.raises(ValueError,match="duplicate"):
        require_unique_occurrence_ids([{"occurrence_id":"1"},{"occurrence_id":"1"}],"occurrence_id")


def test_many_to_one_mapping_is_deterministic():
    mappings=join_historical_identifiers([{"id":"x"},{"id":"x"}],[{"id":"x","state":state()}],
        record_field="id",source_field="id",record_namespace="n",source_namespace="n")
    assert [x.confidence for x in mappings]==[EXACT_VERIFIED,EXACT_VERIFIED]
    assert mappings[0].state_fingerprint==mappings[1].state_fingerprint


def test_partial_recovery_never_reports_clear():
    mappings=join_historical_identifiers([{"id":"x"},{"id":"y"}],[{"id":"x","state":state()}],
        record_field="id",source_field="id",record_namespace="n",source_namespace="n")
    summary=lineage_summary(mappings)
    assert summary["exactly_mapped_records"]==1 and summary["validation_leakage_status"]=="PARTIALLY_VERIFIED"


def test_physical_schema_validation_and_stability():
    assert physical_state_fingerprint(raw_state_from_record({"state":state()}))==physical_state_fingerprint(raw_state_from_record({"state":state()}))
    with pytest.raises(ValueError): raw_state_from_record({"state":{"board":[0]*16,"player_to_move":1}})


def test_source_hash_and_provenance_classification():
    data=b"derived"
    report=classify_provenance(derived_bytes=data,derived_sha256=sha256(data),original_sha256="a"*64,original_bytes=None)
    assert not report["original_bytes_recovered"] and "UNAVAILABLE" in report["status"]
    with pytest.raises(ValueError): classify_provenance(derived_bytes=data,derived_sha256="0"*64,original_sha256=None,original_bytes=None)


def test_original_source_bytes_are_verified_when_available():
    derived=b"derived"; original=b"original"
    report=classify_provenance(derived_bytes=derived,derived_sha256=sha256(derived),
        original_sha256=sha256(original),original_bytes=original)
    assert report["original_bytes_recovered"] is True
    assert report["status"]=="VERIFIED_DERIVATION_CHAIN"


def test_incorrect_original_source_sha_is_rejected():
    with pytest.raises(ValueError,match="source SHA256 mismatch"):
        classify_provenance(derived_bytes=b"derived",derived_sha256=sha256(b"derived"),
            original_sha256="0"*64,original_bytes=b"original")


def test_b3_evidence_summary_requires_complete_exact_mapping_for_clearance():
    complete=join_historical_identifiers([{"id":"a"}],[{"id":"a","state":state()}],
        record_field="id",source_field="id",record_namespace="b3",source_namespace="b3")
    assert lineage_summary(complete)["validation_leakage_status"]=="CLEAR"
    incomplete=join_historical_identifiers([{"id":"a"},{"id":"b"}],[{"id":"a","state":state()}],
        record_field="id",source_field="id",record_namespace="b3",source_namespace="b3")
    assert lineage_summary(incomplete)["validation_leakage_status"]!="CLEAR"


def test_forensic_module_has_no_execution_or_registry_api():
    import songo_ai.evaluation.opening_lineage as module
    forbidden=("registry_payload","SongoLegacyGame","mcts","checkpoint","train")
    assert all(not hasattr(module,name) for name in forbidden)


def test_pending_approval_is_fail_closed():
    record=pending_approval_record(amendment_id="LOT46B_STAGE_C0_OPENING_PROTOCOL_AMENDMENT_V1",
        source_id="LOT45_D_REAL_HUMAN_SUBSET",source_sha256="a"*64,selection_seed=3538971373371044805)
    validate_approval_record(record)
    assert record["approval_decision"]=="PENDING"
    assert not any(record[k] for k in ("source_approved","provenance_risk_accepted","seed_approved","registry_approved","final_registry_created"))


@pytest.mark.parametrize("missing",["approval_authority","approval_timestamp","approval_evidence"])
def test_non_pending_decision_requires_complete_authorization(missing):
    record=pending_approval_record(amendment_id="a",source_id="s",source_sha256="a"*64,selection_seed=1)
    record.update(approval_decision="APPROVED",approval_authority="authority",
                  approval_timestamp="2026-10-20T00:00:00Z",approval_evidence="signed-record")
    record[missing]=None
    with pytest.raises(ValueError,match=missing.replace("_"," ")):
        validate_approval_record(record)


def test_pending_record_rejects_silent_source_or_seed_approval():
    record=pending_approval_record(amendment_id="a",source_id="s",source_sha256="a"*64,selection_seed=1)
    record["seed_approved"]=True
    with pytest.raises(ValueError,match="cannot approve"):
        validate_approval_record(record)
