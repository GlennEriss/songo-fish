"""Preflight Lot45 = preflight Lot44 valide + inputs Lot45 + RUN_LOCK."""
from __future__ import annotations

from lot44.artifacts import Lot44FatalError, read_json
from lot44.preflight import Preflight

from .config import SELECTION_FILE
from .generation import Context
from .inputs import install_reports
from .run_lock import RunLock


class Lot45Preflight(Preflight):
    def __init__(self, ctx: Context, *, require_inputs: bool) -> None:
        super().__init__(ctx, require_lot_inputs=require_inputs)

    def inputs(self) -> dict:
        details = {"probe_positions": str(self.ctx.original_positions), "probe_exists": self.ctx.original_positions.is_file()}
        if not details["probe_exists"]:
            raise FileNotFoundError(self.ctx.original_positions)
        if self.require_lot_inputs:
            rows = self.ctx.selection()
            details.update({"selection_rows": len(rows), "selection_file": SELECTION_FILE})
            if self.ctx.lot44 is None or not (self.ctx.lot44 / "decision.json").is_file():
                raise FileNotFoundError(f"Lot44 decision.json not found under {self.ctx.lot44}")
            details["lot44"] = str(self.ctx.lot44)
        return details

    def run_lock(self) -> dict:
        root = self.tmp / "lock"
        first = RunLock(root, owner="preflight-a", settle_s=0)
        first.acquire()
        second = RunLock(root, owner="preflight-b", settle_s=0)
        try:
            second.acquire()
            refused = False
        except Lot44FatalError as exc:
            refused = exc.code == "RUN_LOCKED"
        first.assert_owner()
        first.release()
        stale = RunLock(root, owner="preflight-stale", settle_s=0)
        stale.acquire()
        later = RunLock(root, owner="preflight-takeover", settle_s=0, stale_after_s=0.0)
        try:
            later.acquire()
            takeover_without_flag = True
        except Lot44FatalError as exc:
            takeover_without_flag = exc.code != "STALE_LOCK"
        later.acquire(takeover_stale=True, reason="preflight stale-lock simulation")
        try:
            stale.assert_owner()
            old_owner_blocked = False
        except Lot44FatalError:
            old_owner_blocked = True
        later.release()
        history = [e["event"] for e in read_json(root / "run_lock_history.json")["events"]]
        ok = refused and not takeover_without_flag and old_owner_blocked and history.count("TAKEOVER") == 1
        if not ok:
            raise RuntimeError(f"run lock semantics broken: refused={refused} takeover_without_flag={takeover_without_flag} old_owner_blocked={old_owner_blocked} history={history}")
        return {"second_writer_refused": refused, "stale_takeover_requires_flag": True, "previous_owner_blocked_after_takeover": old_owner_blocked, "history": history}

    def extra_checks(self):
        return [("RUN_LOCK", self.run_lock, True)]


def run_preflight(ctx: Context, *, require_inputs: bool, repo=None) -> dict:
    ctx.out.mkdir(parents=True, exist_ok=True)
    (ctx.out / "errors.jsonl").touch()
    if require_inputs and repo is not None:
        install_reports(repo, ctx.out)
    return Lot45Preflight(ctx, require_inputs=require_inputs).run()
