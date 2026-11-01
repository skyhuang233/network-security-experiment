"""Resume test for experiment A's `--resume` flag.

Runs the full exp_a entry point twice in the SAME workspace with the
dev-only scripted provider (no API calls).  The second invocation passes
`--resume`: every already-completed run dir (result.json + measurement.json)
must be reused from disk instead of re-executed, and the re-emitted report
must contain the SAME rows (identical wall_seconds, rounds and metrics).
A run dir left incomplete (no result.json) must be re-executed, not resumed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class ExpAResumeTest(unittest.TestCase):
    def test_resume_reuses_completed_runs_and_reruns_partial_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # 1. a legal temporary registry
            from benchmark_generator.generator import run_precheck, save_registry

            report = run_precheck(target_per_class=1, class_ids=["I-06"])
            self.assertGreaterEqual(report["classes"]["I-06"]["valid_count"], 1)
            registry = tmp / "seed-registry.json"
            save_registry(report, registry)

            env = dict(os.environ)
            env.pop("NSE_API_KEY", None)
            env.pop("NSE_MODEL", None)
            env.setdefault("NSE_M3_K", "64")

            def run_exp_a(extra_args: list[str]) -> subprocess.CompletedProcess:
                return subprocess.run(
                    [
                        sys.executable, "-m", "tasks.exp_a",
                        "--workspace", str(tmp / "ws"),
                        "--registry", str(registry),
                        "--classes", "I-06",
                        "--seeds-per-class", "1",
                        "--methods", "M1",
                        "--max-rounds", "10",
                        "--provider", "scripted",
                        *extra_args,
                    ],
                    cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                    timeout=600,
                )

            # 2. first pass: full scripted run (main + negative control)
            first = run_exp_a([])
            self.assertEqual(
                first.returncode, 0,
                f"exp_a first pass failed:\nstdout:\n{first.stdout}\nstderr:\n{first.stderr}",
            )
            report_path = tmp / "ws" / "reports" / "exp_a.json"
            rep1 = json.loads(report_path.read_text())
            self.assertGreaterEqual(len(rep1["runs"]), 1)
            self.assertGreaterEqual(len(rep1["negative_control"]), 1)

            # 3. simulate an interrupted run: wipe one dir's result.json
            #    (a partial dir must never be resumed)
            victim = sorted((tmp / "ws" / "runs").glob("negctrl-M1-*/result.json"))[0]
            victim_rid = victim.parent.name
            victim.unlink()

            # 4. second pass with --resume: completed runs reused from disk,
            #    the partial dir re-executed
            second = run_exp_a(["--resume"])
            self.assertEqual(
                second.returncode, 0,
                f"exp_a resume pass failed:\nstdout:\n{second.stdout}\nstderr:\n{second.stderr}",
            )
            self.assertIn("resumed from disk", second.stdout)
            self.assertNotIn("resume artifact mismatch", second.stdout)

            rep2 = json.loads(report_path.read_text())

            def key_rows(rep: dict) -> dict:
                return {
                    r["case_run_id"]: (
                        r["rounds"], r["wall_seconds"],
                        r["submissions"], r["metrics"],
                    )
                    for r in rep["runs"] + rep["negative_control"]
                }

            rows1, rows2 = key_rows(rep1), key_rows(rep2)
            self.assertEqual(set(rows1), set(rows2))
            for rid in rows1:
                if rid == victim_rid:
                    continue  # re-executed victim, values legitimately differ
                self.assertEqual(
                    rows1[rid], rows2[rid],
                    f"resumed row {rid} must reproduce the first-pass values exactly",
                )
            # the partial dir WAS re-executed and now has both artifacts again
            self.assertTrue(victim.exists())

            # 5. a --resume pass over a fully-complete workspace reuses
            #    every run without re-execution (victim's on-disk values
            #    now come from its pass-2 re-execution, so compare to rows2)
            third = run_exp_a(["--resume"])
            self.assertEqual(third.returncode, 0, third.stdout + third.stderr)
            n_jobs = len(json.loads(report_path.read_text())["runs"]) + len(
                json.loads(report_path.read_text())["negative_control"]
            )
            self.assertEqual(third.stdout.count("resumed from disk"), n_jobs)
            self.assertEqual(key_rows(json.loads(report_path.read_text())), rows2)


if __name__ == "__main__":
    unittest.main()