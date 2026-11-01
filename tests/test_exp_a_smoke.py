"""Entry smoke test for experiment A (issue 1).

Verifies the FULL exp_a entry point end to end with NO real API calls:
a temporary valid seed registry is generated, the model provider is
replaced with the dev-only scripted replay (`--provider scripted`), and the
experiment must complete at least one run per condition and write both
`measurement.json` (per run) and the report.
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


class ExpAScriptedSmokeTest(unittest.TestCase):
    def test_exp_a_completes_run_and_writes_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # 1. a legal temporary registry (dependency precheck path)
            from benchmark_generator.generator import run_precheck, save_registry

            report = run_precheck(target_per_class=1, class_ids=["I-06"])
            self.assertGreaterEqual(report["classes"]["I-06"]["valid_count"], 1)
            registry = tmp / "seed-registry.json"
            save_registry(report, registry)

            # 2. run exp_a as a subprocess with the scripted provider (no API)
            workspace = tmp / "ws"
            env = dict(os.environ)
            env.pop("NSE_API_KEY", None)  # scripted provider must not need it
            env.pop("NSE_MODEL", None)
            env.setdefault("NSE_M3_K", "64")
            proc = subprocess.run(
                [
                    sys.executable, "-m", "tasks.exp_a",
                    "--workspace", str(workspace),
                    "--registry", str(registry),
                    "--classes", "I-06",
                    "--seeds-per-class", "1",
                    "--methods", "M1",
                    "--max-rounds", "10",
                    "--provider", "scripted",
                ],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                timeout=600,
            )
            self.assertEqual(
                proc.returncode, 0,
                f"exp_a failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
            )

            # 3. at least one MAIN run completed and wrote measurement.json
            report_path = workspace / "reports" / "exp_a.json"
            self.assertTrue(report_path.exists(), proc.stdout)
            rep = json.loads(report_path.read_text())
            self.assertGreaterEqual(len(rep["runs"]), 1)
            self.assertEqual(rep["provider"], "scripted")
            self.assertIn("warning", rep)  # scripted runs flagged non-usable

            run_dirs = sorted(
                (workspace / "runs").glob("main-M1-*/measurement.json")
            )
            self.assertTrue(run_dirs, proc.stdout)
            meas = json.loads(run_dirs[0].read_text())
            self.assertGreaterEqual(meas["rounds_completed"], 1)
            # the run recorded the configuration it actually consumed
            run_cfg = [
                e for e in meas["events"] if e.get("event") == "run_config"
            ]
            self.assertTrue(run_cfg)
            self.assertEqual(run_cfg[0]["provider_name"], "scripted")
            self.assertEqual(run_cfg[0]["key_set_size"], 64)  # NSE_M3_K consumed

            # 4. the negative control also completed (separated backends)
            self.assertGreaterEqual(len(rep["negative_control"]), 1)
            neg_dirs = sorted(
                (workspace / "runs").glob("negctrl-M1-*/measurement.json")
            )
            self.assertTrue(neg_dirs, proc.stdout)
            neg = json.loads(neg_dirs[0].read_text())
            neg_cfg = [
                e for e in neg["events"] if e.get("event") == "run_config"
            ]
            self.assertEqual(neg_cfg[0]["policy_mode"], "separated")

            # 5. the recorded config matches what was actually consumed
            cfg = json.loads((workspace / "reports" / "config.json").read_text())
            self.assertEqual(cfg["max_rounds"], 10)
            self.assertEqual(cfg["m3_key_set_size_default"], 64)
            self.assertEqual(cfg["provider"], "scripted")


if __name__ == "__main__":
    unittest.main()