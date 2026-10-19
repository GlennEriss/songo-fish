import pytest
from songo_ai.dataset.selfplay_schema import RawSongoState
from songo_ai.evaluation.opening_registry import *

STATE=RawSongoState(tuple([5]*14+[0,0]),1)
SRC="a"*64
def rec(i="o",state=STATE,source=SRC,split="independent"):
 return OpeningRecord(i,state,physical_state_fingerprint(state),"synthetic",source,split,i)

def test_canonical_and_fingerprint_stable():
 assert canonical({"b":1,"a":2})==canonical({"a":2,"b":1})
 assert physical_state_fingerprint(STATE)==physical_state_fingerprint(STATE)
def test_schema_and_required_fields():
 assert physical_state_payload(STATE)["schema"]==SCHEMA
 with pytest.raises(ValueError):OpeningRecord("",STATE,physical_state_fingerprint(STATE),"s",SRC,"x","r").validate()
def test_invalid_states_rejected_by_raw_contract():
 with pytest.raises(ValueError):RawSongoState((1,)*15,1)
 with pytest.raises(ValueError):RawSongoState(tuple([5]*14+[0,0]),3)
 with pytest.raises(ValueError):RawSongoState(tuple([0]*16),1)
def test_fingerprint_and_source_mismatch():
 with pytest.raises(ValueError):OpeningRecord("o",STATE,"0"*64,"s",SRC,"x","r").validate()
 with pytest.raises(ValueError):rec(source="bad").validate()
def test_duplicate_detection_and_registry_hash():
 with pytest.raises(ValueError):registry_payload((rec("o"),rec("o")),{"status":"HISTORICAL"})
 p={"status":"HISTORICAL","evidence":"synthetic-test"}
 assert registry_fingerprint((rec(),),p)==registry_fingerprint((rec(),),p)
def test_missing_provenance_and_no_silent_substitution():
 with pytest.raises(ValueError):registry_payload((rec(),),{"status":"PROPOSED"})
 with pytest.raises(ValueError):registry_payload((),{"status":"HISTORICAL"})
def test_overlap_audit():
 fp=physical_state_fingerprint(STATE);a=overlap_audit((rec(),),training={fp},validation={fp})
 assert a["training_overlap"]==[fp] and a["validation_overlap"]==[fp]
def test_deterministic_selection_spec_is_proposed_only():
 s=prospective_selection_spec(source_sha256=SRC,seed=20261019)
 assert s["required_openings"]==400 and s["status"]=="PROPOSED" and not s["candidate_outputs_used"]
 with pytest.raises(ValueError):prospective_selection_spec(source_sha256=SRC,seed=1,count=399)
