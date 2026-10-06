"""Audit des APIs reellement utilisees, genere par introspection (section 7)."""
from __future__ import annotations

import dataclasses
import importlib
import inspect
from pathlib import Path

from colab_drive import sha256

from .paths import REPO_ROOT

# (module, attribut, role dans Lot44)
USED_APIS = (
    ("songo_ai.dataset", "RawSongoState", "physical state (board[16], player_to_move); validates conservation of 70 seeds"),
    ("songo_ai.songo.rules", "SongoLegacyGame.from_state", "engine construction from State"),
    ("songo_ai.songo.rules", "SongoLegacyGame.normalize_terminal", "terminal detection without playing"),
    ("songo_ai.songo.rules", "SongoLegacyGame.legal_local_actions", "legal local actions 0..6"),
    ("songo_ai.songo.rules", "SongoLegacyGame.play_local", "apply a local action (preflight only)"),
    ("songo_ai.songo.rules", "SongoLegacyGame.clone", "engine copy (preflight only)"),
    ("songo_ai.search", "MCTSConfig", "frozen PUCT configuration"),
    ("songo_ai.search", "SongoMCTS", "validated MCTS, constructor flags from LOT40_FLAGS"),
    ("songo_ai.search", "SongoMCTS.search", "sequential search (preflight MCTS16/64)"),
    ("songo_ai.search", "SongoMCTS.search_many", "batched search used by Lot44 shards"),
    ("songo_ai.search", "MCTSResult", "search result fields consumed by result_row"),
    ("songo_ai.evaluation", "model_parameter_fingerprint", "weights fingerprint before/after"),
    ("songo_ai.model", "SongoGraphBuilder.build_batch", "graph batch for the model forward check"),
    ("songo_ai.model", "load_srn_checkpoint", "strict state_dict loading (via run_srn_lot39.load_model)"),
    ("run_srn_lot39", "fingerprint_state", "state identity sha256({board, player_to_move})"),
    ("run_srn_lot39", "load_model", "POOL_G4R HybridPolicyValueEvaluator loading"),
    ("run_srn_lot41", "LOT40_FLAGS", "validated SongoMCTS performance flags"),
    ("run_srn_lot43", "jensen_shannon", "JS divergence (natural log)"),
    ("run_srn_lot43", "kendall_agreement", "ranking pair agreement"),
    ("run_srn_lot43", "rank_actions", "visit ranking of legal actions"),
    ("run_srn_lot43", "entropy", "visit entropy"),
    ("run_srn_lot43", "normalize", "visit counts -> distribution"),
    ("run_srn_lot43", "ULTRA_JS_TRIGGER", "Lot43 conservative router threshold"),
    ("run_srn_lot43", "ULTRA_Q_GAP_TRIGGER", "Lot43 conservative router threshold"),
    ("run_srn_colab_benchmark", "pool_fingerprints", "checkpoint sha256 verification against identity manifest"),
    ("run_srn_colab_benchmark", "engine_fingerprint", "sha256 of songo/rules.py"),
    ("run_srn_colab_benchmark", "git_commit", "code commit"),
    ("colab_drive", "sha256", "chunked file sha256"),
    ("colab_drive", "detect_google_drive_root", "Drive desktop detection"),
)


def _resolve(module_name: str, attribute: str):
    obj = importlib.import_module(module_name)
    for part in attribute.split("."):
        obj = getattr(obj, part)
    return obj


def describe(module_name: str, attribute: str, role: str) -> dict:
    obj = _resolve(module_name, attribute)
    entry: dict = {"module": module_name, "name": attribute, "role": role, "kind": type(obj).__name__}
    target = obj
    if dataclasses.is_dataclass(obj) and isinstance(obj, type):
        entry["fields"] = [{"name": f.name, "type": str(f.type)} for f in dataclasses.fields(obj)]
        entry["frozen"] = obj.__dataclass_params__.frozen
    if callable(target) and not isinstance(target, type):
        entry["signature"] = str(inspect.signature(target))
    elif isinstance(target, type):
        entry["signature"] = str(inspect.signature(target.__init__))
    else:
        entry["value"] = target if isinstance(target, (int, float, str, bool, dict, list, tuple)) else repr(target)
    try:
        source_file = Path(inspect.getsourcefile(obj if not isinstance(obj, (int, float, str, dict, tuple)) else importlib.import_module(module_name)))
        entry["file"] = str(source_file.resolve().relative_to(REPO_ROOT))
        entry["file_sha256"] = sha256(source_file)
        if callable(obj):
            entry["line"] = inspect.getsourcelines(obj)[1]
    except (TypeError, OSError, ValueError) as exc:
        entry["source_lookup"] = f"unavailable: {type(exc).__name__}"
    return entry


def build_api_audit(runtime_observations: dict | None = None) -> dict:
    entries = [describe(m, a, r) for m, a, r in USED_APIS]
    return {
        "lot": 44,
        "generated_by": "lot44.api_audit.build_api_audit (inspect on the real repository)",
        "apis": entries,
        "all_resolved": True,
        "runtime_observations": runtime_observations or {},
        "notes": {
            "model_forward": "HybridPolicyValueEvaluator(graph) -> (policy_logits[B,7], value[B]); verified at runtime by preflight",
            "mcts_root_q_perspective": "MCTSNode.q_values = W/N on root edges, root player perspective; unvisited actions report Q=0",
            "terminal_root": "search_many returns selected_action=None and num_simulations=0 for a terminal root; Lot44 corpus excludes terminal and single-legal-action states",
            "mcts_resume": "search/search_many always rebuild the root with _root_node(state) and return no tree",
        },
    }
