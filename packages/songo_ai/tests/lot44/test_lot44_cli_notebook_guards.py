import ast
import builtins
import json
import re
import subprocess

import pytest

from lot44.api_audit import build_api_audit
from lot44.paths import REPO_ROOT, resolve_drive_root, resolve_paths
from run_lot44_router_oos_validation import STAGES, build_parser

NOTEBOOK = REPO_ROOT / "notebooks/lot44_router_oos_validation.ipynb"
LOT44_SOURCES = sorted((REPO_ROOT / "apps/trainer/scripts/lot44").glob("*.py")) + [REPO_ROOT / "apps/trainer/scripts/run_lot44_router_oos_validation.py"]
FROZEN_SCIENCE_FILES = ("packages/songo_ai/search/mcts.py", "packages/songo_ai/songo/rules.py", "packages/songo_ai/songo/fast_rules.py", "packages/songo_ai/model/srn_network.py", "packages/songo_ai/model/srn_graph.py")


def notebook_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    return nb, [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]


def test_parser_accepts_documented_commands():
    parser = build_parser()
    for argv in (
        ["--stage", "preflight"], ["--stage", "smoke"], ["--stage", "prepare", "--max-positions", "768"],
        ["--stage", "run", "--confirm", "--concurrency", "64", "--max-shards", "3"], ["--stage", "finalize"],
        ["--stage", "export", "--bundle", "x.tar.gz"], ["--stage", "prepare-inputs", "--sync-to-drive"], ["--stage", "status"],
    ):
        parser.parse_args(argv)
    with pytest.raises(SystemExit):
        parser.parse_args(["--stage", "evaluate"])


def test_notebook_steps_are_ordered_and_gated():
    _, cells = notebook_cells()
    markdown = [s for s, kind in cells if kind == "markdown"]
    titles = [m.splitlines()[0] for m in markdown if m.startswith("## STEP")]
    assert titles == [f"## STEP {i} — {name}" for i, name in enumerate(["Mount Drive", "Locate project", "Environment", "Preflight", "Smoke test", "Prepare Lot44", "Execute / Resume", "Finalize", "Export"], 1)]
    code = [s for s, kind in cells if kind == "code"]
    assert "CONFIGURATION" in code[0] and "CONFIRM_LONG_COMPUTE = False" in code[0]
    joined = "\n".join(code)
    assert "PREFLIGHT_STATUS'] != 'PASS'" in joined and "raise RuntimeError('PREFLIGHT FAIL" in joined
    assert "'--branch', REPO_BRANCH" in joined and "REPO_BRANCH = 'dev'" in code[0]
    assert "file_sha256(INPUT_BUNDLE) != expected" in joined


def test_notebook_code_compiles_and_names_are_defined_before_use():
    _, cells = notebook_cells()
    defined = set(dir(builtins)) | {"__file__"}
    for index, (source, kind) in enumerate(cells):
        if kind != "code":
            continue
        tree = ast.parse(source, filename=f"cell{index}")
        local_defs = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                local_defs.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                local_defs.add(node.name)
                local_defs.update(a.arg for a in node.args.args + node.args.kwonlyargs) if isinstance(node, ast.FunctionDef) else None
                if isinstance(node, ast.FunctionDef) and node.args.vararg:
                    local_defs.add(node.args.vararg.arg)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                local_defs.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, ast.comprehension):
                local_defs.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
            elif isinstance(node, ast.ExceptHandler) and node.name:
                local_defs.add(node.name)
        loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        undefined = loads - defined - local_defs
        assert not undefined, f"cell {index} uses undefined names {sorted(undefined)}"
        defined |= local_defs


def test_notebook_stages_and_flags_exist_in_the_real_parser():
    _, cells = notebook_cells()
    joined = "\n".join(s for s, kind in cells if kind == "code")
    stages = set(re.findall(r"lot44\('([a-z-]+)'", joined))
    assert stages == {"selftest", "preflight", "smoke", "prepare", "plan", "run", "status", "finalize", "export"}
    assert stages <= set(STAGES)
    flags = set(re.findall(r"'(--[a-z-]+)'", joined)) - {"--branch", "--ff-only"}
    parser_flags = {opt for action in build_parser()._actions for opt in action.option_strings}
    assert flags <= parser_flags, flags - parser_flags
    parser = build_parser()
    base = ["--drive-root", "/x", "--device", "auto", "--seed", "1", "--max-positions", "768"]
    for stage in stages:
        parser.parse_args(["--stage", stage, *base])
    parser.parse_args(["--stage", "run", *base, "--confirm", "--concurrency", "64", "--max-shards", "2"])
    parser.parse_args(["--stage", "export", *base, "--bundle", "/x/lot44_results.tar.gz"])


def test_path_resolution_order(tmp_path, monkeypatch):
    monkeypatch.setenv("SONGO_DRIVE_ROOT", str(tmp_path))
    assert resolve_drive_root(None) == (tmp_path.resolve(), "ENV_SONGO_DRIVE_ROOT")
    cli = tmp_path / "cli"
    cli.mkdir()
    assert resolve_drive_root(str(cli)) == (cli.resolve(), "CLI")
    with pytest.raises(FileNotFoundError):
        resolve_drive_root(str(tmp_path / "missing"))
    paths = resolve_paths(drive_root=str(tmp_path), output=None, lot41=None, lot42=None, lot43=None, exports=None)
    assert paths.output == tmp_path.resolve() / "songo-ai/experiments/lot44_router_out_of_sample_validation"
    assert paths.exports == tmp_path.resolve() / "songo-ai/exports"


def test_api_audit_resolves_every_used_api():
    audit = build_api_audit()
    assert audit["all_resolved"] and len(audit["apis"]) >= 25
    by_name = {a["name"]: a for a in audit["apis"]}
    assert "seeds" in by_name["SongoMCTS.search_many"]["signature"]
    assert {f["name"] for f in by_name["MCTSResult"]["fields"]} >= {"visit_counts", "policy", "root_q_values", "selected_action", "num_simulations", "legal_mask"}


def test_lot44_code_never_trains_or_uses_minimax():
    forbidden = (".backward(", "torch.optim", "optimizer.step", "negamax", "iterative_deepening", "minimax_reference", "load_state_dict(", "add_root_noise=True")
    for path in LOT44_SOURCES:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{token} in {path.name}"
        assert not re.search(r"except[^\n]*:\s*\n\s*pass\b", text), f"silent except in {path.name}"


def test_engine_mcts_and_model_code_untouched_by_lot44():
    try:
        diff = subprocess.run(["git", "diff", "--stat", "HEAD", "--", *FROZEN_SCIENCE_FILES], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git unavailable: {exc}")
    assert diff.strip() == "", diff
