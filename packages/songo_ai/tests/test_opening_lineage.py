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


def test_forensic_module_has_no_execution_or_registry_api():
    import songo_ai.evaluation.opening_lineage as module
    forbidden=("registry_payload","SongoLegacyGame","mcts","checkpoint","train")
    assert all(not hasattr(module,name) for name in forbidden)
