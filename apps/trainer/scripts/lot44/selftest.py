"""Execute la suite pytest Lot44 et ecrit ``test_report.json`` (sections 93-94)."""
from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from .artifacts import utc_now, write_json
from .paths import REPO_ROOT

TEST_DIR = "packages/songo_ai/tests/lot44"


def run_selftest(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    junit = out / ".lot44_junit.xml"
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", TEST_DIR, f"--junitxml={junit}"]
    proc = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if not junit.is_file():
        raise RuntimeError(f"pytest produced no junit report (rc={proc.returncode}):\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")
    root = ET.parse(junit).getroot()
    cases = []
    for case in root.iter("testcase"):
        outcome, message = "passed", None
        for tag in ("failure", "error", "skipped"):
            node = case.find(tag)
            if node is not None:
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
                message = (node.get("message") or "")[:2000]
        cases.append({"test": f"{case.get('classname')}::{case.get('name')}", "outcome": outcome, "seconds": float(case.get("time") or 0), "message": message})
    junit.unlink()
    count = lambda o: sum(c["outcome"] == o for c in cases)  # noqa: E731
    report = {
        "command": " ".join(command),
        "cwd": str(REPO_ROOT),
        "return_code": proc.returncode,
        "timestamp_utc": utc_now(),
        "total": len(cases),
        "passed": count("passed"),
        "failed": count("failed") + count("error"),
        "skipped": count("skipped"),
        "failures": [c for c in cases if c["outcome"] in ("failed", "error")],
        "skips": [c for c in cases if c["outcome"] == "skipped"],
        "cases": cases,
        "stdout_tail": proc.stdout[-3000:],
    }
    write_json(out / "test_report.json", report)
    return report
