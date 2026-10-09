"""Dataset, garde-fous et integration reelle (moteur, POOL_G4R, SongoMCTS)."""
import ast
import builtins
import copy
import dataclasses
import json
import re

import pytest

from lot44.artifacts import Lot44FatalError, read_json, verify_bundle
from lot44.paths import REPO_ROOT
from lot45.config import FORBIDDEN_TARGET_FIELDS
from lot45.dataset import dataset_row, old_vs_deep, policy_from_visits, read_dataset, target_audit, target_statistics, validate_row, write_dataset
from lot45.finalize import DATASET_FILE, finalize, minimax_audit
from lot45.generation import generate, prepare, shard_plan
from lot45.preflight import run_preflight
from lot45.run_lock import RunLock
from lot45.selection import write_selection
from lot45.smoke import REQUIRED_DATASET_KEYS, run_smoke, smoke_selection
from run_lot45_g5_target_generation import STAGES, build_parser

from lot45_test_support import PROBE, make_context, require_inputs

NOTEBOOK = REPO_ROOT / "notebooks/lot45_g5_target_generation.ipynb"
LOT45_SOURCES = sorted((REPO_ROOT / "apps/trainer/scripts/lot45").rglob("*.py")) + [REPO_ROOT / "apps/trainer/scripts/run_lot45_g5_target_generation.py"]
DISTRIBUTED_NOTEBOOK = REPO_ROOT / "notebooks/lot45_distributed_worker.ipynb"


def selection(tmp_path, n=16):
    require_inputs()
    rows = smoke_selection(PROBE, n)
    write_selection(tmp_path / "selection.jsonl.gz", rows)
    return rows


def search_row(out):
    """Une vraie recherche MCTS (budget 64) sur le premier shard de la selection."""
    ctx = make_context(out, target_positions=8)
    prepare(ctx)
    generate(ctx, owner="test")
    shard = next(p for p in (ctx.out / "search/shard_00000/64").glob("shard_*.json") if not p.name.endswith(".sha256.json"))
    payload = read_json(shard)
    return ctx, payload


@pytest.fixture
def computed(tmp_path):
    rows = selection(tmp_path, 8)
    ctx, payload = search_row(tmp_path)
    return rows, ctx, payload


def test_validate_row_accepts_real_rows_and_rejects_corruptions(computed):
    rows, ctx, payload = computed
    by_fp = {r["fingerprint"]: r for r in rows}
    row = payload["rows"][0]
    sel = by_fp[row["state_fingerprint"]]
    assert validate_row(row, sel, 64) == []
    legal = row["legal_mask"]
    illegal = next((a for a in range(7) if not legal[a]), None)
    cases = {
        "simulation_count_63": dict(actual_simulations=63),
        "fingerprint_not_reproducible": dict(state_fingerprint="0" * 64),
        "legal_mask_differs_from_engine": dict(legal_mask=[not x for x in legal]),
        "policy_differs_from_normalized_visits": dict(visit_distribution=[1.0 if i == row["selected_action"] else 0.0 for i in range(7)]),
        "top_action_illegal_or_not_argmax": dict(selected_action=min(row["ranking"][-1], 6) if len(row["ranking"]) > 1 else row["selected_action"]),
        "root_q_not_finite": dict(root_value=float("nan")),
    }
    for problem, change in cases.items():
        bad = {**copy.deepcopy(row), **change}
        if problem == "fingerprint_not_reproducible":
            assert problem in validate_row(bad, sel, 64)
            continue
        assert problem in validate_row(bad, sel, 64), problem
    if illegal is not None:
        visits = list(row["visit_counts"])
        visits[illegal] += 1
        visits[row["selected_action"]] -= 1
        assert "visits_on_illegal_action" in validate_row({**row, "visit_counts": visits}, sel, 64)


def test_dataset_rows_never_contain_teacher_fields_and_roundtrip(computed, tmp_path):
    rows, ctx, payload = computed
    by_fp = {r["fingerprint"]: r for r in rows}
    poisoned = {fp: {**r, "best_action": 3, "action_values": [1] * 7, "teacher": {"depth": 9}} for fp, r in by_fp.items()}
    data = [dataset_row(r, poisoned[r["state_fingerprint"]], payload, teacher={"model_id": "POOL_G4R"}, identity_fp="x") for r in payload["rows"]]
    for row in data:
        assert not set(row) & set(FORBIDDEN_TARGET_FIELDS)
        assert REQUIRED_DATASET_KEYS <= set(row)
        assert row["policy_target"] == policy_from_visits(row["visit_counts"], row["legal_mask"])
        assert row["root_noise"] is False and "root_q_values_diagnostic" in row
    path = tmp_path / "d.jsonl.gz"
    write_dataset(path, data)
    first = path.read_bytes()
    write_dataset(path, data)
    assert path.read_bytes() == first and read_dataset(path) == data
    stats = target_statistics(data)
    assert stats["overall"]["positions"] == len(data) and 0 <= stats["overall"]["one_hot_like_rate"] <= 1
    assert target_audit(data, seed=1)["ALL_CHECKS_PASS"]
    with_old = {fp: {**r, "old_targets": [{"kind": "SELFPLAY_SEARCH", "budget": 64, "root_noise": True, "visit_counts": [1] * 7}]} for fp, r in by_fp.items()}
    comparison = old_vs_deep(data, with_old)["comparisons"]
    assert list(comparison) == ["SELFPLAY_SEARCH_MCTS64_noise_True"] and comparison["SELFPLAY_SEARCH_MCTS64_noise_True"]["positions"] == len(data)


def test_shard_plan_prefix_extension_and_incompatible_change(tmp_path):
    rows = selection(tmp_path, 24)
    plan16 = shard_plan(rows, 16, 8)
    plan24 = shard_plan(rows, 24, 8)
    assert plan24[:2] == plan16 and len(plan24) == 3
    ctx = make_context(tmp_path, target_positions=16)
    prepare(ctx)
    prepare(dataclasses.replace(ctx, target_positions=24, _selection=None))
    with pytest.raises(Lot44FatalError) as err:
        prepare(dataclasses.replace(ctx, shard_size=4, _selection=None))
    assert err.value.code == "SPLIT_MODIFIED"
    with pytest.raises(Lot44FatalError):
        prepare(dataclasses.replace(ctx, target_positions=10_000, _selection=None))


def test_generation_refuses_second_writer_and_resumes(tmp_path):
    selection(tmp_path, 16)
    ctx = make_context(tmp_path, target_positions=16)
    prepare(ctx)
    other = RunLock(tmp_path, owner="mac", settle_s=0)
    other.acquire()
    with pytest.raises(Lot44FatalError) as err:
        generate(ctx, owner="colab")
    assert err.value.code == "RUN_LOCKED"
    assert not (tmp_path / "search").exists()
    other.release()
    first = generate(ctx, owner="colab", max_shards=1)
    assert first["status"] == "PARTIAL" and first["positions_computed"] == 8
    second = generate(ctx, owner="colab")
    assert second["status"] == "COMPLETE" and second["skipped_shards"] == 1 and second["positions_computed"] == 8
    assert generate(ctx, owner="colab")["positions_computed"] == 0


def test_lock_loss_stops_writing(tmp_path, monkeypatch):
    selection(tmp_path, 8)
    ctx = make_context(tmp_path, target_positions=8)
    prepare(ctx)
    import lot45.generation as generation

    original = generation.RunLock.assert_owner

    def stolen(self):
        (tmp_path / "run_lock.json").write_text(json.dumps({**read_json(tmp_path / "run_lock.json"), "run_id": "someone-else"}))
        return original(self)

    monkeypatch.setattr(generation.RunLock, "assert_owner", stolen)
    with pytest.raises(Lot44FatalError) as err:
        generate(ctx, owner="colab")
    assert err.value.code == "LOCK_LOST"
    assert not [p for p in (tmp_path / "search").rglob("shard_*.json")]


def test_finalize_refuses_incomplete_dataset(tmp_path):
    rows = selection(tmp_path, 16)
    ctx = make_context(tmp_path, target_positions=16)
    for name in ("candidate_source_audit.json", "source_selection.json", "source_manifest.json", "deduplication_report.json"):
        (tmp_path / name).write_text("{}")
    from lot45.selection import coverage_report

    (tmp_path / "coverage_report.json").write_text(json.dumps(coverage_report(rows)))
    prepare(ctx)
    generate(ctx, owner="t", max_shards=1)
    with pytest.raises(Lot44FatalError) as err:
        finalize(ctx)
    assert err.value.code == "INCOMPLETE_SEARCH"


def test_smoke_end_to_end(tmp_path):
    require_inputs()
    ctx = make_context(tmp_path / "exp")
    report = run_smoke(ctx, tmp_path / "lot45_smoke")
    assert report["SMOKE_TEST"] == "PASS" and report["steps"]["second_writer_refused"]
    data = read_dataset(tmp_path / "lot45_smoke" / DATASET_FILE)
    assert len(data) == 24 and all(r["simulations"] == 64 for r in data)
    assert verify_bundle(tmp_path / "lot45_smoke_results.tar.gz", "lot45_g5_uniform_mcts32768_targets")["bundle_valid"]
    assert {"ACQUIRE", "RELEASE"} <= set(report["steps"]["lock_history"])


def test_preflight_passes_with_run_lock_check(tmp_path):
    require_inputs()
    ctx = dataclasses.replace(make_context(tmp_path), smoke=False)
    report = run_preflight(ctx, require_inputs=False)
    assert report["PREFLIGHT_STATUS"] == "PASS", report["critical_failures"]
    assert report["summary"]["RUN_LOCK"] == "PASS" and report["summary"]["RESUME"] == "PASS"


def test_minimax_audit_and_source_guards():
    assert minimax_audit()["MINIMAX_LABELS_USED"] == "NO"
    from lot45.config import BUDGET

    assert BUDGET == 32768
    forbidden = (".backward(", "torch.optim", "optimizer.step", "add_root_noise=True", "search_many(", "ROUTER_VALIDATED ==", "budget=65536")
    for path in LOT45_SOURCES:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{token} in {path.name}"
        tree = ast.parse(text)
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not [m for m in imported if "negamax" in m or "minimax" in m or m.startswith("songo_ai.teachers")], path.name
        assert not re.search(r"except[^\n]*:\s*\n\s*pass\b", text), path.name


def assert_notebook(path, steps):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    cells = [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]
    titles = [s.splitlines()[0] for s, k in cells if k == "markdown" and s.startswith("## STEP")]
    assert titles == [f"## STEP {i} — {n}" for i, n in enumerate(steps, 1)]
    return [s for s, k in cells if k == "code"]


def test_distributed_notebook_asks_only_the_required_parameters():
    code = assert_notebook(DISTRIBUTED_NOTEBOOK, ["Mount Drive", "Locate project", "Environment", "Preflight", "Migrate (idempotent)", "Worker", "Finalize", "Export"])
    config = ast.parse(code[0])
    assigned = [t.id for n in config.body if isinstance(n, ast.Assign) for t in n.targets]
    assert assigned == ["WORKER_ID", "COORDINATOR_CONFIG", "DRIVE_ROOT", "PROJECT_ROOT", "DEVICE"]
    joined = "\n".join(code)
    assert "shard_0" not in joined and "--takeover-stale-lock" not in joined
    for stage in ("dist-preflight", "migrate", "worker", "dist-status", "finalize", "export"):
        assert f"'{stage}'" in joined
    assert_cells_defined(code)


def test_cli_and_notebook_are_consistent():
    parser = build_parser()
    for stage in STAGES:
        parser.parse_args(["--stage", stage])
    code = assert_notebook(NOTEBOOK, ["Mount Drive", "Locate project", "Environment", "Preflight", "Smoke test", "Prepare", "Pilot", "Generate / Resume", "Finalize", "Export"])
    assert_cells_defined(code)
    stages = set(re.findall(r"lot45\('([a-z-]+)'", "\n".join(code)))
    assert {"selftest", "preflight", "smoke", "prepare", "pilot", "generate", "finalize", "export"} <= stages
    assert "CONFIRM_LONG_COMPUTE = False" in code[0]


def assert_cells_defined(code):
    defined = set(dir(builtins))
    for index, source in enumerate(code):
        tree = ast.parse(source)
        local = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                local |= {node.name, *(a.arg for a in node.args.args)} | ({node.args.vararg.arg} if node.args.vararg else set())
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                local |= {(a.asname or a.name).split(".")[0] for a in node.names}
            elif isinstance(node, ast.comprehension):
                local |= {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
        undefined = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - defined - local
        assert not undefined, f"code cell {index}: {undefined}"
        defined |= local
    joined = "\n".join(code)
    assert set(re.findall(r"lot45\('([a-z-]+)'", joined)) <= set(STAGES)
    flags = set(re.findall(r"'(--[a-z-]+)'", joined)) - {"--branch", "--ff-only"}
    parser_flags = {o for a in build_parser()._actions for o in a.option_strings}
    assert flags <= parser_flags, flags - parser_flags
    assert "PREFLIGHT FAIL" in joined
