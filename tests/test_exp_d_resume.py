"""Resume test for experiment D's audit `--resume` flag.

Drives the real `exp_d audit` entry point as a subprocess (the module runs
its main flow at import time) with W bound to the dev-only scripted provider
via NSE_W_PROVIDER=scripted, so no API calls happen.  The second invocation
passes `--resume` and must reuse the per-run audit result persisted by the
first (`<workspace>/audit/<view>/<run>.json`) instead of re-judging,
reproducing the report exactly.  A persisted result whose `run_id` does not
match the run dir must be rejected and re-judged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# a real recorded run: one window, so the scripted verdict is deterministic
RUN_DIR = (
    Path("/Users/huangbohan/network-security-experiment")
    / "outputs" / "runs" / "main-M1-I-06-s0"
)


# a long negative-control run: many windows, so concurrency is exercised
LONG_RUN_DIR = (
    Path("/Users/huangbohan/network-security-experiment")
    / "outputs" / "runs" / "negctrl-M1-I-06-s0"
)


def _run_audit(workspace: Path, extra: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["NSE_W_PROVIDER"] = "scripted"
    env.pop("NSE_API_KEY", None)
    env.pop("NSE_MODEL", None)
    return subprocess.run(
        [
            sys.executable, "-m", "tasks.exp_d", "audit",
            "--workspace", str(workspace),
            str(RUN_DIR),
            "--views", "content",
            *extra,
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env, capture_output=True, text=True, timeout=300,
    )


class ExpDAuditResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(
            (RUN_DIR / "measurement.json").exists(),
            f"recorded run fixture missing: {RUN_DIR}",
        )

    def test_resume_reuses_persisted_audit_and_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td) / "ws"

            first = _run_audit(ws, [])
            self.assertEqual(
                first.returncode, 0,
                f"first pass failed:\nstdout:\n{first.stdout}\nstderr:\n{first.stderr}",
            )
            saved = ws / "audit" / "content" / f"{RUN_DIR.name}.json"
            self.assertTrue(saved.exists(), first.stdout)
            rep1 = json.loads((ws / "reports" / "exp_d_audit.json").read_text())

            # second pass reuses the persisted judgments verbatim
            second = _run_audit(ws, ["--resume"])
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn("resumed", second.stdout)
            rep2 = json.loads((ws / "reports" / "exp_d_audit.json").read_text())
            self.assertEqual(rep1["views"], rep2["views"])

            # a persisted result claiming a different run is not trusted
            tampered = json.loads(saved.read_text())
            tampered["run_id"] = "not-this-run"
            saved.write_text(json.dumps(tampered))
            third = _run_audit(ws, ["--resume"])
            self.assertEqual(third.returncode, 0, third.stdout + third.stderr)
            self.assertNotIn("resumed", third.stdout)
            # and the re-judged result was written back with the right id
            self.assertEqual(
                json.loads(saved.read_text())["run_id"], RUN_DIR.name
            )

    def test_workers_match_serial_judgments_in_window_order(self) -> None:
        self.assertTrue((LONG_RUN_DIR / "measurement.json").exists())
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            serial = _run_audit_over(base / "serial", LONG_RUN_DIR, [])
            self.assertEqual(serial.returncode, 0, serial.stderr)
            parallel = _run_audit_over(
                base / "parallel", LONG_RUN_DIR, ["--workers", "8"]
            )
            self.assertEqual(parallel.returncode, 0, parallel.stderr)

            def judged(ws: Path) -> list:
                rep = json.loads(
                    (ws / "audit" / "content" / f"{LONG_RUN_DIR.name}.json")
                    .read_text()
                )
                return rep["judged"]

            got = judged(base / "parallel")
            # the scripted provider yields one CLEAN verdict per window, so
            # any reordering or dropped window would show up here
            self.assertEqual(got, judged(base / "serial"))
            self.assertGreater(len(got), 1)
            self.assertTrue(all(v is False for v in got))


def _run_audit_over(
    workspace: Path, run_dir: Path, extra: list[str]
) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["NSE_W_PROVIDER"] = "scripted"
    env.pop("NSE_API_KEY", None)
    env.pop("NSE_MODEL", None)
    return subprocess.run(
        [
            sys.executable, "-m", "tasks.exp_d", "audit",
            "--workspace", str(workspace),
            str(run_dir),
            "--views", "content",
            *extra,
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env, capture_output=True, text=True, timeout=300,
    )


if __name__ == "__main__":
    unittest.main()
