import json

from songo_ai.evaluation import (
    annotation_conflict_summary, discover_teacher_corpora,
    iter_internal_teacher_records, overlap_summary, position_fingerprint,
    teacher_record_integrity,
)


def row(best=0):
    return {"state": [5] * 14 + [0, 0], "legal_mask": [True] * 7, "best_action": best, "action_values": [1, 0, 0, 0, 0, 0, 0], "action_value_depths": [2] * 7, "policy_target": [1, 0, 0, 0, 0, 0, 0], "principal_variation": [best], "teacher": {"depth": 2, "tier": "standard"}, "move_number": 0}


def test_discovery_excludes_drl_and_real(tmp_path):
    for root in ("teacher", "d_rl", "real_matches"):
        directory = tmp_path/root; directory.mkdir(); (directory/"manifest.json").write_text("{}")
        (directory/"train.jsonl").write_text(json.dumps(row()) + "\n")
    assert list(discover_teacher_corpora(tmp_path)) == ["teacher"]


def test_internal_records_keep_position_and_annotation_separate(tmp_path):
    path=tmp_path/"train.jsonl"; path.write_text(json.dumps(row())+"\n")
    record=next(iter_internal_teacher_records(path,"x"))
    assert record.position_key == (tuple([5]*14+[0,0]),1)
    assert not record.physical_player_recoverable
    assert record.annotation["best_action"] == 0
    assert teacher_record_integrity(record) == []


def test_annotation_conflicts_are_detected(tmp_path):
    path=tmp_path/"train.jsonl"; path.write_text(json.dumps(row(0))+"\n"+json.dumps(row(1))+"\n")
    report=annotation_conflict_summary(list(iter_internal_teacher_records(path,"x")))
    assert report["positions_with_best_action_conflict"] == 1
    assert report["positions_with_annotation_conflict"] == 1


def test_position_hash_includes_player_without_canonicalizing():
    board = tuple([5] * 14 + [0, 0])
    assert position_fingerprint((board, 1)) == position_fingerprint((board, 1))
    assert position_fingerprint((board, 1)) != position_fingerprint((board, 2))
    assert position_fingerprint((board, 1)) != position_fingerprint((tuple(reversed(board)), 1))


def test_overlap_and_deduplication_are_exact():
    board = tuple([5] * 14 + [0, 0])
    other = tuple([4] + [5] * 12 + [6, 0, 0])
    left = {(board, 1), (board, 1), (other, 2)}
    right = {(board, 1)}
    assert len(left) == 2
    assert overlap_summary(left, right) == {
        "intersection": 1,
        "percent_left_in_right": 50.0,
        "percent_right_in_left": 100.0,
    }


def test_streaming_reader_skips_blank_lines_and_reports_integrity(tmp_path):
    invalid = row()
    invalid["state"][0] = -1
    invalid["state"][1] = 11
    path = tmp_path / "train.jsonl"
    path.write_text("\n" + json.dumps(row()) + "\n\n" + json.dumps(invalid) + "\n")
    records = list(iter_internal_teacher_records(path, "x"))
    assert len(records) == 2
    assert teacher_record_integrity(records[0]) == []
    assert "board_negative" in teacher_record_integrity(records[1])


def test_integrity_rejects_mask_mismatch_and_illegal_teacher_action(tmp_path):
    value = row(best=0)
    value["legal_mask"][0] = False
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(value) + "\n")
    errors = teacher_record_integrity(next(iter_internal_teacher_records(path, "x")))
    assert "legal_mask_engine_mismatch" in errors
    assert "best_action_illegal" in errors
